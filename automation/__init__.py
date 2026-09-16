"""
speclab.automation — instrument acquisition layer for AutomateFTIR.

Hardware drivers and the machine-specific configuration loader, separated from
the GUI in the v0.11.x refactor.  Each controller is named for the box it
drives (``MultimeterController``, ``SpectrometerController``), and returns
domain-named data.

Scope boundary
--------------
This subpackage serves ``AutomateFTIR.pyw`` only.  **Nothing in the analysis
half of the package** — ``functions``, ``utils``, ``plot``, ``EmissionLWIR``,
``ReflectanceVSWIR``, ``SpeclibViewer`` — may import from it, and
``speclab/__init__.py`` deliberately does not re-export it.  Those modules must
run on a machine with no instrument attached, no ``instrument_config.yaml``
present, and without ``pyvisa`` or ``pywin32`` installed; importing anything
here would break that.

The rule was previously a note in a docstring.  Making it a directory makes it
visible.

Configuration and mutable state
-------------------------------
Controllers take their settings by **injection** (an address, a DDE service
name), never by reading module globals.  That is deliberate: ``AutomateFTIR``
rebinds its config globals at runtime when the settings dialog saves, and a
``from ... import NAME`` elsewhere would capture a stale copy that never
updates.  Keep it that way — immutable constants may be imported by name,
anything that can be rebound must not be.
"""
