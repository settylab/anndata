"""
Jinja2 Environment for the AnnData HTML repr.

The repr renders by feeding structured values (plus pre-produced HTML
fragments wrapped in ``markupsafe.Markup``) into templates loaded from
``anndata._repr.templates``. Autoescape is on by default:

- Plain ``str`` values are escaped (``<``, ``>``, ``&``, ``"``, ``'``).
- ``Markup`` values pass through verbatim because Jinja recognises the type.

The trust contract is therefore typed rather than conventional: data arrives
as ``str`` and gets escaped; HTML arrives as ``Markup`` and is trusted.
"""

from __future__ import annotations

from functools import cache

from jinja2 import Environment, PackageLoader, select_autoescape

from .._repr_constants import (
    CSS_TEXT_ERROR,
    CSS_TEXT_MUTED,
    NOT_SERIALIZABLE_MSG,
    STYLE_HIDDEN,
)


@cache
def get_env() -> Environment:
    env = Environment(
        loader=PackageLoader("anndata._repr", "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(
        CSS_TEXT_ERROR=CSS_TEXT_ERROR,
        CSS_TEXT_MUTED=CSS_TEXT_MUTED,
        NOT_SERIALIZABLE_MSG=NOT_SERIALIZABLE_MSG,
        STYLE_HIDDEN=STYLE_HIDDEN,
    )
    return env
