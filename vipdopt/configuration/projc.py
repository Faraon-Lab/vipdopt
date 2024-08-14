"""Configuration manager for SonyBayerFilter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

import numpy as np
import yaml
from overrides import override

from vipdopt.configuration.config import Config
from vipdopt.configuration.template import TemplateRenderer
from vipdopt.utils import ensure_path

class ProjectConfig(Config):
    """Config object used to save and load Project classes (see vipdopt/project.py)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def __setitem__(self, name: str, value: Any) -> None:
        super().__setitem__(name, value)

    @ensure_path
    @override
    def read_file(self, fname: Path, cfg_format: str = 'auto') -> None:
        super().read_file(fname, cfg_format=cfg_format)

    @overload
    def update(self, __m: SupportsKeysAndGetItem, **kwargs: Any) -> None: ...

    @overload
    def update(self, __m: Iterable[tuple[Any, Any]], **kwargs) -> None: ...

    @overload
    def update(self, **kwargs: Any) -> None: ...

    def update(self, *args, **kwargs: Any) -> None:
        """Update self with values from another dictionary-like object."""
        self._do_validation = False
        if len(args) == 0:
            super().update(**kwargs)
        else:
            super().update(args[0], **kwargs)
