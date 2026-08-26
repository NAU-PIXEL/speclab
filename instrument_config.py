"""
Instrument configuration loader for AutomateFTIR.

Site-specific and operational settings live in ``instrument_config.yaml`` at
the package root.  That file is untracked; ``instrument_config.example.yaml``
is tracked and is copied into place the first time the GUI runs on a machine.

Rationale for the split
-----------------------
Settings that vary per machine (multimeter address, OMNIC install paths, data
directory) previously lived as constants in ``AutomateFTIR.pyw``.  Editing them
dirtied the working tree on every deployment and conflicted on pull.  Keeping
the live file untracked removes both problems and keeps the lab's internal IP
out of the repository.

The file is machine-scoped rather than user-scoped, and therefore sits at the
package root rather than in ``%APPDATA%``: it describes the instrument in the
room, not the person logged in.  Two accounts on the lab PC must not need to
configure the same Keithley separately.

Scope
-----
This module serves ``AutomateFTIR.pyw`` only.  Nothing in the analysis side of
the package (``EmissionLWIR``, ``functions``, ``plot``, ``utils``) imports it,
and it should stay that way: those modules must run on a machine with no
instrument attached and no ``instrument_config.yaml`` present.

What is deliberately *not* configurable here
--------------------------------------------
Multimeter channel numbers, labels, units, and the resistance/temperature split
are a data contract, not configuration, and live in code.  The channel map is
``utils.CHANNEL_LABELS`` — placed there because it is shared with the analysis
GUI and because ``utils`` already owns the instrument-specific PRT conversions
(``r2t_nau`` and friends) that consume those same channels.

DDE protocol identifiers (OMNIC server and topic names) are protocol constants
and likewise stay in the source.

Usage
-----
>>> from speclab.instrument_config import load_instrument_config
>>> cfg, first_run = load_instrument_config()
>>> cfg['multimeter']['address']
'TCPIP::10.11.100.182::1394::SOCKET'
"""

import logging
import shutil
from pathlib import Path

import yaml

_ROOT         = Path(__file__).resolve().parent
_LIVE_PATH    = _ROOT / 'instrument_config.yaml'
_EXAMPLE_PATH = _ROOT / 'instrument_config.example.yaml'

# Required scalar keys per section, with accepted types.  ``experiments`` is
# nested and validated separately by _validate_experiments.
_SCHEMA: dict[str, dict[str, type | tuple[type, ...]]] = {
    'multimeter': {
        'address':                str,
        'poll_interval_s':        (int, float),
        'measurement_interval_s': (int, float),
    },
    'omnic': {
        'param_dir':             str,
        'exe':                   str,
        'autoconnect_poll_ms':   (int, float),
        'autoconnect_timeout_s': (int, float),
        'collect_max_retries':   int,
        'collect_retry_delay_s': (int, float),
    },
    'blackbody': {
        'warm_min_c': (int, float),
        'hot_min_c':  (int, float),
    },
    'collection': {
        'purge_delay_s': (int, float),
    },
    'data': {
        'default_dir': str,
    },
}

# Keys coerced from str to Path after validation.
_PATH_KEYS: tuple[tuple[str, str], ...] = (
    ('omnic', 'param_dir'),
    ('omnic', 'exe'),
    ('data',  'default_dir'),
)

# Path keys warned about (not failed) when absent from disk, so the GUI still
# launches on a machine without OMNIC installed.
_EXISTENCE_CHECKED: tuple[tuple[str, str], ...] = (
    ('omnic', 'param_dir'),
    ('omnic', 'exe'),
)

# Numeric keys that must be strictly positive; a zero poll interval would spin.
_STRICTLY_POSITIVE: tuple[tuple[str, str], ...] = (
    ('multimeter', 'poll_interval_s'),
    ('multimeter', 'measurement_interval_s'),
    ('omnic',      'autoconnect_poll_ms'),
    ('omnic',      'autoconnect_timeout_s'),
)

# Numeric keys that must be non-negative; zero is meaningful (skip the wait).
_NON_NEGATIVE: tuple[tuple[str, str], ...] = (
    ('omnic',      'collect_max_retries'),
    ('omnic',      'collect_retry_delay_s'),
    ('collection', 'purge_delay_s'),
)

_REQUIRED_MODES: tuple[str, ...] = ('Emission', 'Transmission', 'Reflectance')


class InstrumentConfigError(Exception):
    """Raised when the instrument configuration file is missing or malformed."""


def load_instrument_config(path: Path | None = None) -> tuple[dict, bool]:
    """
    Load, validate, and return the instrument configuration.

    If the live configuration file does not exist, the tracked example is
    copied into place and ``first_run`` is returned True so the caller can
    open the settings dialog.

    Parameters
    ----------
    path : Path or None, optional
        Explicit path to the configuration file.  Defaults to
        ``instrument_config.yaml`` at the package root.  Intended for tests.

    Returns
    -------
    cfg : dict
        Parsed and validated configuration.  Path-valued keys are coerced to
        ``pathlib.Path``.
    first_run : bool
        True if the file was just created from the example and still holds
        placeholder values.

    Raises
    ------
    InstrumentConfigError
        If the file is absent and no example exists to seed it, if it is not
        valid YAML, or if any required key is missing, mistyped, or out of
        range.  The message names both the offending key and the file path.
    """
    target    = Path(path) if path is not None else _LIVE_PATH
    first_run = False

    if not target.exists():
        if not _EXAMPLE_PATH.exists():
            raise InstrumentConfigError(
                f"No instrument configuration at {target}, and no template at "
                f"{_EXAMPLE_PATH} to create one from. Restore "
                f"instrument_config.example.yaml from the repository."
            )
        shutil.copyfile(_EXAMPLE_PATH, target)
        first_run = True
        logging.info("Created %s from template — configuration required", target)

    # encoding is explicit: the Windows default (cp1252) would mangle the file.
    try:
        with open(target, encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise InstrumentConfigError(f"{target}: not valid YAML — {exc}") from exc
    except OSError as exc:
        raise InstrumentConfigError(f"{target}: could not be read — {exc}") from exc

    if not isinstance(cfg, dict):
        raise InstrumentConfigError(
            f"{target}: expected a mapping at the top level, got "
            f"{type(cfg).__name__}"
        )

    _validate(cfg, target)
    _coerce_paths(cfg)
    _warn_missing_paths(cfg)

    return cfg, first_run


def update_instrument_config(updates: dict[tuple[str, str], str],
                             path: Path | None = None) -> None:
    """
    Rewrite individual scalar values in place, preserving comments and layout.

    A whole-file ``yaml.safe_dump`` would discard every comment in the file.
    Those comments carry the reasoning behind the values — most importantly the
    relay-wear budget on ``poll_interval_s`` — and are the main reason this is
    YAML rather than JSON.  So only the matched value lines are touched; all
    formatting, comments and key order survive untouched.

    The rewritten file is re-parsed and validated before the change is
    committed.  If it no longer loads, the original is restored and the error
    is raised, so a failed save cannot leave an unusable configuration behind.

    Parameters
    ----------
    updates : dict of (str, str) -> str
        Mapping of ``(section, key)`` to the new value.  Only scalar
        (non-nested) keys are supported.
    path : Path or None, optional
        Target file.  Defaults to ``instrument_config.yaml`` at the package
        root.

    Raises
    ------
    InstrumentConfigError
        If a requested key is not found in the file, or if the rewritten file
        fails validation (in which case the original has been restored).
    """
    target = Path(path) if path is not None else _LIVE_PATH

    try:
        original = target.read_text(encoding='utf-8')
    except OSError as exc:
        raise InstrumentConfigError(f"{target}: could not be read — {exc}") from exc

    lines     = original.splitlines(keepends=True)
    remaining = dict(updates)
    section   = None

    for i, line in enumerate(lines):
        stripped = line.rstrip('\n')
        if not stripped or stripped.lstrip().startswith('#'):
            continue

        # A top-level key with no indentation opens a new section.
        if not line[:1].isspace():
            section = stripped.split(':', 1)[0].strip()
            continue

        for (sec, key) in list(remaining):
            if sec != section:
                continue
            head, sep, _ = stripped.partition(':')
            if sep and head.strip() == key:
                indent   = line[:len(line) - len(line.lstrip())]
                lines[i] = f'{indent}{key}: {_yaml_scalar(remaining[(sec, key)])}\n'
                del remaining[(sec, key)]
                break

    if remaining:
        missing = ', '.join(f"'{s}.{k}'" for s, k in sorted(remaining))
        raise InstrumentConfigError(
            f"{target}: could not locate key(s) {missing} to update"
        )

    target.write_text(''.join(lines), encoding='utf-8')

    # Prove the result still loads before letting the change stand.
    try:
        load_instrument_config(target)
    except InstrumentConfigError:
        target.write_text(original, encoding='utf-8')
        logging.error("Rejected config update; restored previous %s", target)
        raise


def _yaml_scalar(value: str) -> str:
    """
    Render *value* as a single-quoted YAML scalar.

    Single quotes are used deliberately: in double-quoted YAML the backslashes
    in Windows paths would be read as escape sequences, so ``C:\\my documents``
    would silently corrupt.  Embedded single quotes are doubled per the spec.

    Parameters
    ----------
    value : str
        Raw value to encode.

    Returns
    -------
    str
        Quoted scalar suitable for direct substitution into the file.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _validate(cfg: dict, path: Path) -> None:
    """
    Check that every required section, key, type, and range constraint holds.

    Parameters
    ----------
    cfg : dict
        Freshly parsed configuration.
    path : Path
        Source file, named in error messages so the user knows what to edit.

    Raises
    ------
    InstrumentConfigError
        On the first violation found, naming the dotted key path.
    """
    for section, keys in _SCHEMA.items():
        if section not in cfg:
            raise InstrumentConfigError(
                f"{path}: missing required section '{section}'"
            )
        if not isinstance(cfg[section], dict):
            raise InstrumentConfigError(
                f"{path}: '{section}' must be a mapping, got "
                f"{type(cfg[section]).__name__}"
            )
        for key, expected in keys.items():
            if key not in cfg[section]:
                raise InstrumentConfigError(
                    f"{path}: missing required key '{section}.{key}'"
                )
            value = cfg[section][key]
            # bool subclasses int, so `poll_interval_s: true` would otherwise pass.
            if isinstance(value, bool) or not isinstance(value, expected):
                names = (expected.__name__ if isinstance(expected, type)
                         else ' or '.join(t.__name__ for t in expected))
                raise InstrumentConfigError(
                    f"{path}: '{section}.{key}' must be {names}, got "
                    f"{type(value).__name__}"
                )

    for section, key in _STRICTLY_POSITIVE:
        if cfg[section][key] <= 0:
            raise InstrumentConfigError(
                f"{path}: '{section}.{key}' must be greater than zero, got "
                f"{cfg[section][key]}"
            )

    for section, key in _NON_NEGATIVE:
        if cfg[section][key] < 0:
            raise InstrumentConfigError(
                f"{path}: '{section}.{key}' must not be negative, got "
                f"{cfg[section][key]}"
            )

    if not cfg['multimeter']['address'].strip():
        raise InstrumentConfigError(f"{path}: 'multimeter.address' is empty")

    # Reversed thresholds would break BB auto-selection silently.
    if cfg['blackbody']['warm_min_c'] >= cfg['blackbody']['hot_min_c']:
        raise InstrumentConfigError(
            f"{path}: 'blackbody.warm_min_c' "
            f"({cfg['blackbody']['warm_min_c']}) must be below "
            f"'blackbody.hot_min_c' ({cfg['blackbody']['hot_min_c']})"
        )

    _validate_experiments(cfg, path)


def _validate_experiments(cfg: dict, path: Path) -> None:
    """
    Check the per-mode OMNIC experiment mapping.

    Parameters
    ----------
    cfg : dict
        Configuration containing ``omnic.experiments``.
    path : Path
        Source file, named in error messages.

    Raises
    ------
    InstrumentConfigError
        If a mode is missing, or an entry lacks a usable ``file`` or
        ``keywords`` field.
    """
    experiments = cfg['omnic'].get('experiments')
    if not isinstance(experiments, dict):
        raise InstrumentConfigError(
            f"{path}: 'omnic.experiments' must be a mapping of mode to "
            f"{{file, keywords}}"
        )

    for mode in _REQUIRED_MODES:
        if mode not in experiments:
            raise InstrumentConfigError(
                f"{path}: 'omnic.experiments' is missing mode '{mode}'"
            )
        entry = experiments[mode]
        if not isinstance(entry, dict):
            raise InstrumentConfigError(
                f"{path}: 'omnic.experiments.{mode}' must be a mapping with "
                f"'file' and 'keywords'"
            )

        exp_file = entry.get('file')
        if not isinstance(exp_file, str) or not exp_file.strip():
            raise InstrumentConfigError(
                f"{path}: 'omnic.experiments.{mode}.file' must be a non-empty "
                f"string"
            )

        keywords = entry.get('keywords')
        if (not isinstance(keywords, list) or not keywords
                or not all(isinstance(k, str) and k.strip() for k in keywords)):
            raise InstrumentConfigError(
                f"{path}: 'omnic.experiments.{mode}.keywords' must be a "
                f"non-empty list of strings"
            )


def _coerce_paths(cfg: dict) -> None:
    """
    Convert path-valued strings to ``pathlib.Path`` in place.

    Parameters
    ----------
    cfg : dict
        Validated configuration, modified in place.
    """
    for section, key in _PATH_KEYS:
        cfg[section][key] = Path(cfg[section][key])


def _warn_missing_paths(cfg: dict) -> None:
    """
    Log a warning for configured paths that do not exist on disk.

    Deliberately non-fatal: the GUI should still launch on a machine without
    OMNIC installed, and Transmission/Reflectance work without the multimeter.

    Parameters
    ----------
    cfg : dict
        Validated configuration with paths already coerced.
    """
    for section, key in _EXISTENCE_CHECKED:
        value = cfg[section][key]
        if not value.exists():
            logging.warning(
                "Configured path '%s.%s' does not exist: %s", section, key, value
            )
