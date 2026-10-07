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
from typing import overload

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

from .._repr_constants import (
    CSS_COLORS,
    CSS_COLORS_SWATCH,
    CSS_COLORS_SWATCH_INVALID,
    CSS_DTYPE_UNKNOWN,
    CSS_NESTED_ANNDATA,
    CSS_TEXT_ERROR,
    CSS_TEXT_MUTED,
    CSS_TEXT_WARNING,
    NOT_SERIALIZABLE_MSG,
    STYLE_HIDDEN,
)


def _scrub_nulls(value):
    # Null bytes in user data break HTML parsers; replace pre-escape.
    # Markup values pass through unchanged.
    if isinstance(value, str) and not hasattr(value, "__html__"):
        return value.replace("\x00", "\ufffd")
    return value


@cache
def get_env() -> Environment:
    env = Environment(
        loader=PackageLoader("anndata._repr", "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
        finalize=_scrub_nulls,
        # Templates ship inside the package and never change at runtime;
        # skip the per-lookup mtime check of the default ``auto_reload=True``.
        auto_reload=False,
        undefined=StrictUndefined,
    )
    env.globals.update(
        CSS_COLORS=CSS_COLORS,
        CSS_COLORS_SWATCH=CSS_COLORS_SWATCH,
        CSS_COLORS_SWATCH_INVALID=CSS_COLORS_SWATCH_INVALID,
        CSS_DTYPE_UNKNOWN=CSS_DTYPE_UNKNOWN,
        CSS_NESTED_ANNDATA=CSS_NESTED_ANNDATA,
        CSS_TEXT_ERROR=CSS_TEXT_ERROR,
        CSS_TEXT_MUTED=CSS_TEXT_MUTED,
        CSS_TEXT_WARNING=CSS_TEXT_WARNING,
        NOT_SERIALIZABLE_MSG=NOT_SERIALIZABLE_MSG,
        STYLE_HIDDEN=STYLE_HIDDEN,
    )
    return env


@overload
def as_markup(html: str | Markup) -> Markup: ...
@overload
def as_markup(html: None) -> None: ...
def as_markup(html: str | Markup | None) -> Markup | None:
    """Mark extension-supplied ``*_html`` as trusted HTML.

    This is the single boundary where a plain ``str`` passed in a ``*_html``
    field (``preview_html``, ``type_html``, ``expanded_html``) or as
    ``entries_html`` becomes ``Markup``. By contract such strings are trusted:
    the producer escapes any user data it interpolates (``escape_html``,
    ``Markup.format``). Everything else reaching a template as ``str`` is
    autoescaped.
    """
    if html is None or isinstance(html, Markup):
        return html
    return Markup(html)  # noqa: S704 - documented trust boundary for *_html


def render_template(name: str, /, **context: object) -> Markup:
    """Render a packaged template to trusted ``Markup``.

    Template output is autoescaped (plain ``str`` values are escaped, ``Markup``
    values pass through), so the result is safe to mark as trusted.
    """
    return Markup(get_env().get_template(name).render(**context))  # noqa: S704 - autoescaped template output


@cache
def get_macros():
    """Cached handle to the macros module from ``_macros.j2``.

    Extension packages can invoke any macro as ``get_macros().name(args)``.
    Macro calls render through this module's Jinja environment, so
    arguments flow through autoescape + the NUL-scrub finalize hook — the
    safest way to build custom HTML from user data.
    """
    return get_env().get_template("_macros.j2").module
