"""
User-level configuration for speclab.

Settings are persisted to ``~/.config/speclab/config.json`` and loaded
automatically on import.

Usage
-----
>>> import speclab
>>> speclab.configure(spectral_libraries_dir='/path/to/spectral_libraries')
>>> speclab.get_config()
{'spectral_libraries_dir': '/path/to/spectral_libraries'}
"""

import json
import logging
from pathlib import Path

_CONFIG_FILE = Path.home() / '.config' / 'speclab' / 'config.json'

_CONFIG: dict = {}

# Valid configuration keys and their descriptions.
_VALID_KEYS = {
    'spectral_libraries_dir': 'Path to the directory containing spectral library HDF files.',
    'default_library':        'Path to the HDF file loaded as the Full Library on startup.',
}


def configure(**kwargs) -> None:
    """
    Set one or more configuration values and persist them to disk.

    Parameters
    ----------
    **kwargs
        Key-value pairs to update.  Valid keys: ``spectral_libraries_dir``.

    Raises
    ------
    KeyError
        If an unrecognised configuration key is passed.
    """
    unknown = set(kwargs) - set(_VALID_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown config key(s): {unknown}. "
            f"Valid keys: {list(_VALID_KEYS)}"
        )
    _CONFIG.update(kwargs)
    _save()
    logging.info("speclab config updated: %s", list(kwargs.keys()))


def get_config() -> dict:
    """Return a copy of the current configuration."""
    return dict(_CONFIG)


def _save() -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, 'w') as fh:
        json.dump(_CONFIG, fh, indent=2)


def _load() -> None:
    global _CONFIG
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE) as fh:
                _CONFIG = json.load(fh)
        except Exception as exc:
            logging.warning("speclab: could not read config file: %s", exc)
            _CONFIG = {}


# Auto-load on import.
_load()
