"""
OMNIC spectrometer DDE driver for AutomateFTIR.

Extracted from ``AutomateFTIR.pyw`` (v0.11.0).  The controller was already
GUI-free and took the DDE service identifiers by injection, so the move is a
relocation rather than a rewrite.

Scope
-----
Acquisition side only.  Nothing in the analysis half of the package may import
this module: ``win32ui``/``dde`` are Windows-only and absent on analysis
machines.
"""

import logging

import win32ui  # initialises the win32 OLE layer required by dde  # noqa: F401
import dde


# DDE protocol identifiers, not settings -- these name the OMNIC DDE service
# itself and are fixed by the protocol, so they stay in the source rather than
# in instrument_config.yaml.
OMNIC_SERVER_NAME = 'OMNIC'
OMNIC_TOPIC_NAME  = 'SPECTRA'


# =============================================================================
# =========================== SpectrometerController ==========================
# =============================================================================
class SpectrometerController:
    """
    Manages DDE communication with the OMNIC spectrometer application.

    OMNIC must be open and fully loaded before ``connect`` is called.
    All DDE commands execute synchronously; ``collect`` blocks its calling
    thread until the scan reaches 100 %.

    Parameters
    ----------
    server_name : str
        DDE server name (``'OMNIC'``).
    topic_name : str
        DDE topic (``'SPECTRA'``).
    """

    def __init__(self, server_name: str, topic_name: str) -> None:
        self._server_name: str = server_name
        self._topic_name: str  = topic_name
        self._server           = None
        self._conv             = None

    @property
    def connected(self) -> bool:
        """True if a DDE conversation is open."""
        return self._conv is not None

    def connect(self) -> bool:
        """
        Create a DDE server and open a conversation with OMNIC.

        Returns
        -------
        bool
            True on success.
        """
        srv = None
        try:
            srv  = dde.CreateServer()
            srv.Create(self._server_name)
            conv = dde.CreateConversation(srv)
            conv.ConnectTo(self._server_name, self._topic_name)
            self._server = srv
            self._conv   = conv
            logging.info("Spectrometer DDE connected: %s / %s",
                         self._server_name, self._topic_name)
            return True
        except Exception as exc:
            logging.error("Spectrometer connect failed: %s", exc)
            # Destroy the server if it was created but ConnectTo failed;
            # leaving it alive prevents a clean retry.
            if srv is not None:
                try:
                    srv.Destroy()
                except Exception:
                    pass
            self._server = None
            self._conv   = None
            return False

    def disconnect(self) -> None:
        """Drop DDE references without closing OMNIC."""
        self._conv = None
        if self._server is not None:
            try:
                self._server.Destroy()
            except Exception:
                pass
            self._server = None

    def _exec(self, cmd: str) -> None:
        """Send a DDE execute command; raises RuntimeError if not connected."""
        if self._conv is None:
            raise RuntimeError("Spectrometer not connected")
        self._conv.Exec(cmd)

    def set_option(self, name: str, value: str) -> None:
        """Set an OMNIC Options parameter via DDE.

        Uses the DDE ``Set`` command syntax::

            [Set Options <name> <value>]

        Parameters
        ----------
        name : str
            Option parameter name (e.g. ``'CollectPrompt'``).
        value : str
            New value (e.g. ``'True'`` or ``'False'``).
        """
        self._exec(f'[Set Options {name} {value}]')

    def load_experiment(self, exp_path: str) -> None:
        """
        Load an OMNIC experiment parameter file.

        Parameters
        ----------
        exp_path : str
            Full path to the ``.exp`` parameter file.
        """
        self._exec(f'[LoadParameters "{exp_path}"]')

    def bench_align(self) -> None:
        """Trigger the OMNIC bench alignment routine."""
        self._exec('[Invoke StartBenchAlign]')

    def start_collect(self, name: str) -> None:
        """
        Send the CollectSample DDE execute command.

        Uses ``Auto Polling`` for all modes: no OMNIC prompts, no collection
        window; spectrum is placed directly in the active spectral window.
        Shutters are left in manual/always-open mode in the ``.exp`` file.
        T/R purge equilibration is handled by the GUI before this call.

        Must be called from the main thread (Win32 DDE requires a message pump).
        Returns immediately; use :meth:`poll_collect_status` to track progress.

        Parameters
        ----------
        name : str
            Spectrum label passed to OMNIC.
        """
        self._exec(f'[CollectSample "{name}" Auto Polling]')

    def poll_collect_status(self) -> tuple[int, int]:
        """
        Request the current collection progress from OMNIC.

        Must be called from the main thread (Win32 DDE requires a message pump).

        Returns
        -------
        tuple[int, int]
            ``(n_scans_completed, pct_complete)`` where *pct_complete* is 0–100.
        """
        if self._conv is None:
            raise RuntimeError("Spectrometer not connected")
        status = self._conv.Request('Collect Status')
        parts  = status.split(',')
        return int(parts[0]), int(parts[7])

    def export_csv(self, out_path: str) -> None:
        """
        Export the currently displayed spectrum to a CSV file.

        Parameters
        ----------
        out_path : str
            Destination file path (OMNIC requires the ``.CSV`` extension).
        """
        self._exec(f'[Export "{out_path}"]')

    def display(self) -> None:
        """Display the most recently collected spectrum in OMNIC."""
        self._exec('[Display]')

    def hide_selected(self) -> None:
        """Hide the selected spectrum in the OMNIC spectral window."""
        self._exec('[HideSelectedSpectra]')

    def query_exp_params(self) -> dict:
        """
        Query current experiment parameters from OMNIC via DDE Request.

        Must be called from the main thread.  Failures on individual
        parameters are silently stored as empty strings.

        Returns
        -------
        dict
            Mapping of parameter key → string value.
        """
        if self._conv is None:
            return {}
        requests = [
            ('resolution',   'Collect Resolution'),
            ('num_scans',    'Collect NumScans'),
            ('apodization',  'Collect ApodizationFunction'),
            ('zero_fill',    'Collect ZeroFill'),
            ('high_cutoff',  'Bench HighCutoff'),
            ('low_cutoff',   'Bench LowCutoff'),
            ('gain',         'Bench Gain'),
            ('beamsplitter', 'Bench BeamSplitter'),
            ('velocity',     'Bench Velocity'),
        ]
        params: dict = {}
        for key, dde_param in requests:
            try:
                params[key] = self._conv.Request(dde_param).strip()
            except Exception:
                params[key] = ''
        return params
