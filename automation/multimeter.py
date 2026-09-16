"""
Keithley 2701 multimeter driver for AutomateFTIR.

Extracted from ``AutomateFTIR.pyw`` (v0.11.0) where it lived as a top-level
class alongside the GUI.  The controller was already GUI-free and took its
address by injection, so the move is a relocation rather than a rewrite.

Scope
-----
Acquisition side only.  Nothing in the analysis half of the package
(``functions``, ``utils``, ``plot``, ``EmissionLWIR``) may import this module:
those must run on a machine with no instrument attached and no ``pyvisa``
installed.

The channel map below is a data contract, not configuration -- see the note on
``CHANNEL_MODES``.
"""

import logging
import threading

import pyvisa


# =============================================================================
# ============================= channel metadata ==============================
# =============================================================================
# The physical wiring of the Keithley multiplexer card.  Deliberately not
# configurable: these channel numbers are the measurement-info CSV schema and
# the science pipeline binds roles to specific numbers.  Labels live in
# utils.CHANNEL_LABELS, shared with EmissionLWIR so the two cannot drift.
# Panel display order stays in AutomateFTIR.pyw -- it is a GUI concern.

# Channels 101–102 use 4-wire resistance; 103–107 use temperature.
CHANNEL_MODES: dict[int, str] = {
    101: 'FRES',
    102: 'FRES',
    103: 'Temperature',
    104: 'Temperature',
    105: 'Temperature',
    106: 'Temperature',
    107: 'Temperature',
}

# Display unit follows from the measurement function, so it is derived rather
# than declared: a parallel dict could silently disagree with the instrument.
MODE_UNITS: dict[str, str] = {'FRES': 'Ω', 'Temperature': '°C'}
CHANNEL_UNITS: dict[int, str] = {
    ch: MODE_UNITS[mode] for ch, mode in CHANNEL_MODES.items()
}


# =============================================================================
# ============================ MultimeterController ===========================
# =============================================================================
class MultimeterController:
    """
    Manages the PyVISA connection and channel reads for the Keithley 2701.

    Thread safety
    -------------
    ``read_channel`` acquires ``_lock`` around each VISA query to prevent
    interleaving of writes from the live-poll thread and collection threads.
    ``disconnect`` takes the same lock, so closing from the GUI thread waits for
    an in-flight channel read instead of splicing ``:SYSTem:LOCal`` into the
    middle of it.  The lock is re-entrant because ``read_channel`` calls
    ``disconnect`` on its error path while already holding it.

    Parameters
    ----------
    address : str
        VISA resource string (e.g. ``'TCPIP::192.0.2.10::1394::SOCKET'``).
        The live value comes from ``instrument_config.yaml``.
    """

    # Trailing chars to strip from the SCPI response token.
    # Empirical from ftir-automation-v4.py: FRES strips 5, Temperature strips 2.
    _STRIP: dict[str, int] = {'FRES': 5, 'Temperature': 2}

    def __init__(self, address: str) -> None:
        self._address: str                     = address
        self._resource: pyvisa.Resource | None = None
        self._lock: threading.RLock            = threading.RLock()

    @property
    def connected(self) -> bool:
        """True if a VISA resource is currently open."""
        return self._resource is not None

    @property
    def address(self) -> str:
        """Current VISA resource string."""
        return self._address

    @address.setter
    def address(self, value: str) -> None:
        """
        Re-point the controller at a different instrument.

        Does not reconnect: call ``connect`` afterwards, which closes any open
        session first.  Used by the settings dialog so an address change takes
        effect without restarting the GUI.
        """
        self._address = value

    def connect(self) -> bool:
        """
        Open the VISA resource at the configured address.

        Returns
        -------
        bool
            True on success, False on any connection error.
        """
        self.disconnect()   # close any existing session before opening a new one
        try:
            rm  = pyvisa.ResourceManager()
            res = rm.open_resource(self._address)
            res.read_termination = '\n'
            self._resource = res
            logging.info("Multimeter connected: %s", self._address)
            return True
        except Exception as exc:
            logging.error("Multimeter connect failed: %s", exc)
            self._resource = None
            return False

    def disconnect(self) -> None:
        """
        Return the instrument to local control and close the VISA resource.

        Notes
        -----
        Any SCPI command puts the instrument into REMOTE, which locks out its
        front panel.  Without ``:SYSTem:LOCal`` the panel stays locked after
        the GUI exits, and the operator has to press LOCAL on the instrument
        to get it back.

        Holds ``_lock`` for the duration, so a concurrent sweep either finishes
        its current channel first or sees the resource gone and stops -- it can
        never send a command after ``:SYSTem:LOCal`` and re-lock the panel.

        The command is best-effort: ``disconnect`` is also called from the
        error path in :meth:`read_channel` after a ``VisaIOError``, where the
        socket is already dead and the write cannot succeed.  A failure there
        is expected and must not mask the original error or prevent the
        resource from being closed.
        """
        with self._lock:
            if self._resource is None:
                return
            try:
                self._resource.write(':SYSTem:LOCal')
            except Exception:
                pass    # socket already gone; nothing to restore
            try:
                self._resource.close()
            except Exception:
                pass
            self._resource = None

    def read_channel(self, channel: int) -> float:
        """
        Read one Keithley channel.

        Closes the relay for *channel*, sets the appropriate measurement
        function, and queries a fresh reading via ``:SENSe:DATA:FRESh?``.

        Parameters
        ----------
        channel : int
            Channel number (101–107).

        Returns
        -------
        float
            Measured value in Ω for channels 101–102, °C for 103–107.

        Raises
        ------
        RuntimeError
            If the multimeter is not connected.
        KeyError
            If *channel* is not in the channel map.
        pyvisa.errors.VisaIOError
            On instrument communication failure.
        """
        mode  = CHANNEL_MODES[channel]
        strip = self._STRIP[mode]
        with self._lock:
            # Checked under the lock: a concurrent disconnect() may have closed
            # the resource while this thread was waiting to acquire it.
            if self._resource is None:
                raise RuntimeError("Multimeter not connected")
            try:
                self._resource.write(f':ROUTe:CLOSe (@{channel})')
                self._resource.write(f":SENSe:FUNCtion '{mode}'")
                raw = self._resource.query(':SENSe:DATA:FRESh?')
            except pyvisa.errors.VisaIOError as exc:
                # TCP connection lost; tear down so the next connect() starts clean.
                self.disconnect()
                raise
        return float(raw.split(',')[0][:-strip])

    def read_all_channels(self, channels: list[int]) -> dict[int, float]:
        """
        Read multiple channels sequentially.

        Individual channel failures are logged and stored as ``float('nan')``
        so a partial result is always returned rather than raising.

        Parameters
        ----------
        channels : list of int
            Channel numbers to read.

        Returns
        -------
        dict[int, float]
            Channel number → measured value; failed channels contain NaN.
        """
        readings: dict[int, float] = {}
        for ch in channels:
            try:
                readings[ch] = self.read_channel(ch)
            except Exception as exc:
                logging.error("Channel %d read failed: %s", ch, exc)
                readings[ch] = float('nan')
        return readings
