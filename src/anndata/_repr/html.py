"""
Main HTML generator for AnnData representation.

This module generates the complete HTML representation by:
1. Building the header with badges
2. Rendering the search box
3. Generating metadata (version, memory)
4. Rendering each section (X, obs, var, uns, etc.)
5. Handling nested objects recursively
"""

from __future__ import annotations

import contextlib
import re
import uuid
from typing import TYPE_CHECKING

from markupsafe import Markup

from .._core.anndata import AnnData
from .._repr_constants import (
    CHAR_WIDTH_PX,
    COPY_BUTTON_PADDING_PX,
    CSS_BADGE_EXTENSION,
    DEFAULT_FIELD_WIDTH_PX,
    MIN_FIELD_WIDTH_PX,
    TOOLTIP_TRUNCATE_LENGTH,
)
from .._settings import settings
from .._types import AnnDataElem
from ..utils import get_literal_members
from . import (
    formatters as _formatters,  # noqa: F401  # side-effect: register built-in formatters
)
from .components import (
    render_badge,
    render_header_badges,
    render_search_box,
)
from .core import (
    render_error_section,
    render_formatted_entry,
    render_index_preview,
    render_section,
    render_truncation_indicator,
    render_x_entry,
)
from .css import get_css
from .environment import get_env, get_macros
from .javascript import get_javascript
from .lazy import get_lazy_backing_info, is_lazy_adata
from .registry import (
    FormatterContext,
    formatter_registry,
)
from .sections import (
    _detect_unknown_sections,
    _render_dataframe_section,
    _render_mapping_section,
    _render_raw_section,
    _render_unknown_sections,
    _render_uns_section,
)
from .utils import (
    format_memory_size,
    format_number,
    get_anndata_version,
    get_backing_info,
    is_backed,
    is_view,
)

if TYPE_CHECKING:
    from .registry import SectionFormatter


# container_id is interpolated verbatim into a <script> block
# (see javascript.py: `getElementById('{container_id}')`), so any
# caller-supplied value must match this restrictive shape.
_CONTAINER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# Display order of the standard sections: the main matrix first, then the
# annotations in the order of :data:`AnnDataElem`.
_ELEMS: tuple[AnnDataElem, ...] = tuple(get_literal_members(AnnDataElem))
_SECTION_ORDER: tuple[AnnDataElem, ...] = ("X", *(e for e in _ELEMS if e != "X"))


def _collect_all_field_names(adata: AnnData) -> list[str]:
    """
    Collect all field names from standard and custom sections.

    Returns field names from obs/var columns and keys from mapping sections
    (uns, obsm, varm, layers, obsp, varp) plus any registered custom sections.
    """
    all_names: list[str] = []
    standard_sections: set[str] = set(_SECTION_ORDER)

    for section in _SECTION_ORDER:
        if section in {"X", "raw"}:
            continue
        try:
            attr = getattr(adata, section)
            if attr is None:
                continue
            if section in {"obs", "var"}:
                if hasattr(attr, "columns"):
                    all_names.extend(attr.columns.tolist())
            elif hasattr(attr, "keys"):
                # skip `layers[None]`, which is `.X`
                all_names.extend(k for k in attr if k is not None)
        except Exception:  # noqa: BLE001
            # Broken section — skip for width calculation, error placeholder is
            # rendered separately by _render_section.
            pass

    # Registered custom sections (e.g., TreeData's obst/vart)
    for section_name in formatter_registry.get_registered_sections():
        if section_name in standard_sections:
            continue
        try:
            attr = getattr(adata, section_name, None)
            if attr is not None and hasattr(attr, "keys"):
                all_names.extend(attr.keys())
        except Exception:  # noqa: BLE001
            pass

    return all_names


def _calculate_field_name_width(adata: AnnData, max_width: int) -> int:
    """
    Calculate the optimal field name column width based on longest field name.

    Uses _collect_all_field_names() to gather names from all sections,
    then converts the longest name to a pixel width (up to max_width).

    Uses constants from _repr_constants.py tuned for the default 13px monospace font.
    """
    all_names = _collect_all_field_names(adata)

    if not all_names:
        return DEFAULT_FIELD_WIDTH_PX

    # Find longest name and convert to pixels
    max_len = max(len(str(name)) for name in all_names)
    width_px = (max_len * CHAR_WIDTH_PX) + COPY_BUTTON_PADDING_PX

    # Clamp to reasonable range (max_width from user setting always wins)
    return min(max(MIN_FIELD_WIDTH_PX, width_px), max_width)


def _create_formatter_context(
    adata: AnnData,
    *,
    depth: int = 0,
    max_depth: int | None = None,
    fold_threshold: int | None = None,
    max_items: int | None = None,
    max_lazy_categories: int | None = None,
) -> FormatterContext:
    """Create a FormatterContext, using explicit overrides where given and settings otherwise."""

    def resolve(override: int | None, setting: int) -> int:
        return setting if override is None else override

    return FormatterContext(
        depth=depth,
        max_depth=resolve(max_depth, settings.repr_html_max_depth),
        fold_threshold=resolve(fold_threshold, settings.repr_html_fold_threshold),
        max_items=resolve(max_items, settings.repr_html_max_items),
        max_lazy_categories=resolve(
            max_lazy_categories, settings.repr_html_max_lazy_categories
        ),
        max_categories=settings.repr_html_max_categories,
        max_string_length=settings.repr_html_max_string_length,
        unique_limit=settings.repr_html_unique_limit,
        adata_ref=adata,
    )


def generate_repr_html(  # noqa: PLR0913
    adata: AnnData,
    *,
    depth: int = 0,
    max_depth: int | None = None,
    fold_threshold: int | None = None,
    max_items: int | None = None,
    max_lazy_categories: int | None = None,
    show_header: bool = True,
    show_search: bool = True,
    _container_id: str | None = None,
) -> Markup:
    """
    Generate HTML representation for an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to represent
    depth
        Current recursion depth (for nested AnnData in .uns)
    max_depth
        Maximum recursion depth. Uses settings/default if None.
    fold_threshold
        Auto-fold sections with more entries than this. Uses settings/default if None.
    max_items
        Maximum items to show per section. Uses settings/default if None.
    max_lazy_categories
        Maximum categories to load for lazy categoricals. Set to 0 to disable
        loading categories entirely (metadata-only mode). Uses settings/default if None.
    show_header
        Whether to show the header (for nested display)
    show_search
        Whether to show the search box (only at top level)
    _container_id
        Internal: container ID for scoping

    Returns
    -------
    HTML string
    """
    # Check if HTML repr is enabled
    if not settings.repr_html_enabled:
        return get_macros().pre_fallback(repr(adata))

    # Create formatter context (resolves settings)
    context = _create_formatter_context(
        adata,
        depth=depth,
        max_depth=max_depth,
        fold_threshold=fold_threshold,
        max_items=max_items,
        max_lazy_categories=max_lazy_categories,
    )

    # Check max depth
    if depth >= context.max_depth:
        return _render_max_depth_indicator(adata)

    # Generate unique container ID. container_id is interpolated verbatim into
    # a <script> block (see javascript.py), so caller-supplied values must match
    # a restrictive pattern to prevent JS injection. Auto-generated UUIDs are
    # safe by construction and skip the check.
    if _container_id is not None:
        if not _CONTAINER_ID_RE.match(_container_id):
            msg = (
                f"_container_id must match {_CONTAINER_ID_RE.pattern!r}; "
                f"got {_container_id!r}"
            )
            raise ValueError(msg)
        container_id = _container_id
    else:
        container_id = f"anndata-repr-{uuid.uuid4().hex[:8]}"

    # Calculate field name column width based on content
    field_width = _calculate_field_name_width(adata, settings.repr_html_max_field_width)
    type_width = settings.repr_html_type_width

    # Computed column widths as CSS variables. Inline font-family:monospace
    # provides a readable fallback when CSS is stripped (GitHub, untrusted
    # notebooks).
    style = (
        f"font-family: monospace; "
        f"--anndata-name-col-width: {field_width}px; "
        f"--anndata-type-col-width: {type_width}px;"
    )

    header_html: Markup | None = None
    if show_header:
        header_html = _render_header(
            adata,
            show_search=show_search and depth == 0,
            container_id=container_id,
        )

    index_preview_markup: Markup | None = None
    footer_html: Markup | None = None
    hints_html: Markup | None = None
    css_html: Markup | None = None
    javascript_html: Markup | None = None
    if depth == 0:
        index_preview_markup = render_index_preview(adata)
        footer_html = _render_footer(adata)
        hints_html = _render_hints()
        css_html = Markup(get_css())
        javascript_html = Markup(get_javascript(container_id))

    return Markup(
        get_env()
        .get_template("anndata.j2")
        .render(
            container_id=container_id,
            depth=depth,
            style=style,
            css=css_html,
            header=header_html,
            index_preview=index_preview_markup,
            sections=_render_all_sections(adata, context),
            footer=footer_html,
            hints=hints_html,
            javascript=javascript_html,
        )
    )


def _render_all_sections(
    adata: AnnData,
    context: FormatterContext,
) -> list[Markup]:
    """Render all standard and custom sections."""
    parts: list[Markup] = []
    custom_sections_after = _get_custom_sections_by_position(adata)

    for section in _SECTION_ORDER:
        parts.append(_render_section(adata, section, context))

        # Render custom sections after this section
        if section in custom_sections_after:
            parts.extend(
                _render_custom_section(adata, section_formatter, context)
                for section_formatter in custom_sections_after[section]
            )

    # Custom sections at end (no specific position)
    if None in custom_sections_after:
        parts.extend(
            _render_custom_section(adata, section_formatter, context)
            for section_formatter in custom_sections_after[None]
        )

    # Detect and show unknown sections (attributes not in AnnDataElem)
    unknown_sections = _detect_unknown_sections(adata)
    if unknown_sections:
        parts.append(_render_unknown_sections(unknown_sections))

    return parts


def _render_section(
    adata: AnnData,
    section: str,
    context: FormatterContext,
) -> Markup:
    """Render a single standard section.

    Attribute access happens inside the try/except so a broken section (one
    whose ``getattr`` raises — e.g. a corrupt aligned mapping or a subclass
    with a crashing property) renders as an error placeholder instead of
    aborting the whole repr. This is why we iterate section names directly
    via ``_SECTION_ORDER`` rather than delegating to
    ``iter_outer``, which propagates the first exception it hits.
    """
    try:
        if section == "X":
            return render_x_entry(adata, context)
        elem = getattr(adata, section)
        if section == "raw":
            return _render_raw_section(elem, context)
        if section in ("obs", "var"):
            return _render_dataframe_section(section, elem, context)
        if section == "uns":
            return _render_uns_section(elem, context)
        return _render_mapping_section(section, elem, context)
    except Exception as e:  # noqa: BLE001
        # Show error instead of hiding the section
        return render_error_section(section, f"{type(e).__name__}: {e}")


def _get_custom_sections_by_position(
    adata: object,
) -> dict[str | None, list[SectionFormatter]]:
    """
    Get registered custom section formatters grouped by their position.

    Returns a dict mapping after_section -> list of formatters.
    None key contains formatters that should appear at the end.
    """
    from collections import defaultdict

    result = defaultdict(list)
    standard_section_names: set[str] = set(_SECTION_ORDER)

    for section_name in formatter_registry.get_registered_sections():
        formatter = formatter_registry.get_section_formatter(section_name)
        if formatter is None:
            continue

        # Skip standard sections (they're handled separately)
        if section_name in standard_section_names:
            continue

        # Check if this section should be shown for this object
        try:
            if not formatter.should_show(adata):
                continue
        except Exception:  # noqa: BLE001
            # Intentional broad catch: custom formatters shouldn't break the repr
            continue

        # Group by position
        after = getattr(formatter, "after_section", None)
        result[after].append(formatter)

    return dict(result)


def _render_custom_section(
    adata: AnnData,
    formatter: SectionFormatter,
    context: FormatterContext,
) -> Markup:
    """Render a custom section using its registered formatter.

    If the formatter defines ``render_html(obj, context)``, it is tried
    first and the result is used as-is (no ``<details>`` wrapping).
    If ``render_html`` fails, falls back to the standard ``get_entries``
    path so formatters can provide both an enhanced and a safe representation.
    """
    if hasattr(formatter, "render_html"):
        try:
            return Markup(formatter.render_html(adata, context))
        except Exception as e:  # noqa: BLE001
            from .._warnings import warn

            warn(
                f"Custom section formatter '{formatter.section_name}' render_html failed, "
                f"falling back to get_entries: {e}",
                UserWarning,
            )

    try:
        entries = formatter.get_entries(adata, context)
    except Exception as e:  # noqa: BLE001
        # Intentional broad catch: custom formatters shouldn't crash the entire
        # repr. Show the failure like for built-in sections, and warn for debugging.
        from .._warnings import warn

        warn(
            f"Custom section formatter '{formatter.section_name}' failed: {e}",
            UserWarning,
        )
        return render_error_section(formatter.section_name, f"{type(e).__name__}: {e}")

    if not entries:
        return Markup("")

    n_items = len(entries)
    section_name = formatter.section_name

    # Render entries (with truncation)
    rows = []
    for i, entry in enumerate(entries):
        if i >= context.max_items:
            rows.append(render_truncation_indicator(n_items - context.max_items))
            break
        rows.append(render_formatted_entry(entry, section_name))

    return render_section(
        getattr(formatter, "display_name", section_name),
        Markup("\n").join(rows),
        n_items=n_items,
        doc_url=getattr(formatter, "doc_url", None),
        tooltip=getattr(formatter, "tooltip", ""),
        should_collapse=n_items > context.fold_threshold,
        section_id=section_name,
    )


def _build_readme_icon(adata: AnnData) -> Markup | None:
    """Build the README icon Markup from ``adata.uns['README']`` if present.

    The truncation + tooltip shaping stays in Python; only the final Markup
    fragment is handed off to the template.
    """
    readme_content = adata.uns.get("README") if hasattr(adata, "uns") else None
    if not (isinstance(readme_content, str) and readme_content.strip()):
        return None

    max_readme_size = settings.repr_html_max_readme_size
    original_len = len(readme_content)
    if max_readme_size > 0 and original_len > max_readme_size:
        readme_content = readme_content[:max_readme_size]
        truncation_note = (
            f"\n\n---\n*README truncated: showing {max_readme_size:,} of "
            f"{original_len:,} characters*"
        )
        readme_content += truncation_note

    tooltip_text = readme_content[:TOOLTIP_TRUNCATE_LENGTH]
    if len(readme_content) > TOOLTIP_TRUNCATE_LENGTH:
        tooltip_text += "..."

    # Scrub NULs before the template: the previous ``.format(...)`` path only
    # HTML-escaped, so NUL bytes flowed through into the ``data-readme``
    # attribute. The macro autoescapes but doesn't scrub; Jinja's finalize
    # hook scrubs too, but being explicit here keeps the contract obvious.
    readme_content = readme_content.replace("\x00", "\ufffd")
    tooltip_text = tooltip_text.replace("\x00", "\ufffd")

    return get_macros().readme_icon(readme_content, tooltip_text)


def _render_header(
    adata: AnnData, *, show_search: bool = False, container_id: str = ""
) -> Markup:
    """Render the header with type, shape, badges, and optional search box."""
    type_name = type(adata).__name__
    shape_str = f"{format_number(adata.n_obs)} obs × {format_number(adata.n_vars)} vars"

    # View / backed / lazy badges and backing file path
    backed = is_backed(adata)
    lazy = is_lazy_adata(adata)
    backing_path = backing_format = None
    is_open = None
    if backed:
        backing = get_backing_info(adata)
        backing_path = str(backing.get("filename") or "")
        backing_format = str(backing.get("format") or "")
        is_open = bool(backing.get("is_open"))
    elif lazy:
        lazy_info = get_lazy_backing_info(adata)
        backing_path = lazy_info.get("filename", "")
        backing_format = lazy_info.get("format", "")
    extras: list[Markup] = [
        render_header_badges(
            is_view=is_view(adata),
            is_backed=backed,
            is_lazy=lazy,
            backing_path=backing_path,
            backing_format=backing_format,
            is_open=is_open,
        )
    ]

    # Mark AnnData subclasses (the type name itself is already shown above).
    # AnnData-like containers (e.g. MuData) only duck-type, so they get no badge.
    if type_name != "AnnData" and isinstance(adata, AnnData):
        cls = type(adata)
        extras.append(
            render_badge(
                "AnnData subclass",
                CSS_BADGE_EXTENSION,
                f"{cls.__module__}.{cls.__qualname__}",
            )
        )

    readme_icon = _build_readme_icon(adata)
    if readme_icon is not None:
        extras.append(readme_icon)

    search_box = render_search_box(container_id) if show_search else None

    return Markup(
        get_env()
        .get_template("header.j2")
        .render(
            type_name=type_name,
            shape_str=shape_str,
            extras=extras,
            search_box=search_box,
        )
    )


def _render_footer(adata: AnnData) -> Markup:
    """Render the footer with version and memory info.

    The memory estimate is omitted for lazy AnnData, where everything stays on
    disk and ``__sizeof__`` would only count a few in-memory wrappers.
    """
    memory_str: str | None = None
    memory_title = "Estimated memory usage"
    if not is_lazy_adata(adata):
        # __sizeof__ recurses into user data which can raise anything
        with contextlib.suppress(Exception):
            memory_str = format_memory_size(adata.__sizeof__())
        if is_backed(adata):
            memory_title = "Estimated in-memory size (data on disk not included)"

    return Markup(
        get_env()
        .get_template("footer.j2")
        .render(
            version=get_anndata_version(),
            memory_str=memory_str,
            memory_title=memory_title,
        )
    )


def _render_hints() -> Markup:
    """Render the static no-CSS / no-JS hint block."""
    return Markup(get_env().get_template("hints.j2").render())


def _render_max_depth_indicator(adata: AnnData) -> Markup:
    """Render indicator when max depth is reached."""
    n_obs = getattr(adata, "n_obs", "?")
    n_vars = getattr(adata, "n_vars", "?")
    return Markup(
        get_env()
        .get_template("max_depth_indicator.j2")
        .render(n_obs_str=format_number(n_obs), n_vars_str=format_number(n_vars))
    )
