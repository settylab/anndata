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

import uuid
from typing import TYPE_CHECKING

from .._repr_constants import (
    CSS_BADGE_BACKED,
    CSS_BADGE_EXTENSION,
    CSS_BADGE_LAZY,
    CSS_BADGE_VIEW,
    DEFAULT_MAX_README_SIZE,
    TOOLTIP_TRUNCATE_LENGTH,
)
from . import (
    DEFAULT_FOLD_THRESHOLD,
    DEFAULT_MAX_CATEGORIES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FIELD_WIDTH,
    DEFAULT_MAX_ITEMS,
    DEFAULT_MAX_LAZY_CATEGORIES,
    DEFAULT_MAX_STRING_LENGTH,
    DEFAULT_PREVIEW_ITEMS,
    DEFAULT_TYPE_WIDTH,
    DEFAULT_UNIQUE_LIMIT,
    SECTION_OBS,
    SECTION_ORDER,
    SECTION_RAW,
    SECTION_VAR,
    SECTION_X,
)
from .components import (
    render_badge,
    render_search_box,
)
from .core import (
    render_empty_section,
    render_formatted_entry,
    render_section,
    render_truncation_indicator,
    render_x_entry,
)
from .css import get_css
from .javascript import get_javascript
from .lazy import get_lazy_backing_info, is_lazy_adata
from .registry import (
    FormatterContext,
    formatter_registry,
)
from .sections import (
    _detect_unknown_sections,
    _render_entry_row,
    _render_error_entry,
    _render_raw_section,
    _render_unknown_sections,
    _render_uns_section,
)
from .utils import (
    escape_html,
    format_index_preview,
    format_memory_size,
    format_number,
    get_anndata_version,
    get_backing_info,
    get_setting,
    is_backed,
    is_view,
)

if TYPE_CHECKING:
    from anndata import AnnData

    from .registry import SectionFormatter

# Import formatters to register them (side-effect import)
from .._repr_constants import (
    CHAR_WIDTH_PX,
    COPY_BUTTON_PADDING_PX,
    DEFAULT_FIELD_WIDTH_PX,
    MIN_FIELD_WIDTH_PX,
)
from . import formatters as _formatters  # noqa: F401


def _collect_all_field_names(adata: AnnData) -> list[str]:
    """
    Collect all field names from standard and custom sections.

    Returns field names from obs/var columns and keys from mapping sections
    (uns, obsm, varm, layers, obsp, varp) plus any registered custom sections.
    """
    all_names: list[str] = []
    skip_sections = {SECTION_X, SECTION_RAW}  # Single items, not collections

    # Standard sections from SECTION_ORDER
    for section in SECTION_ORDER:
        if section in skip_sections:
            continue
        try:
            attr = getattr(adata, section, None)
            if attr is None:
                continue
            # obs/var are DataFrames - use column names
            if section in (SECTION_OBS, SECTION_VAR):
                if hasattr(attr, "columns"):
                    all_names.extend(attr.columns.tolist())
            # Other sections are mappings - use keys
            elif hasattr(attr, "keys"):
                all_names.extend(attr.keys())
        except Exception:  # noqa: BLE001
            pass

    # Registered custom sections (e.g., TreeData's obst/vart)
    for section_name in formatter_registry.get_registered_sections():
        if section_name in SECTION_ORDER:
            continue  # Already handled
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


def _resolve_setting(override: int | None, setting_name: str, default: int) -> int:
    """Resolve a setting value with priority: explicit override > anndata.settings > default.

    Parameters
    ----------
    override
        Explicit value passed to generate_repr_html (highest priority)
    setting_name
        Name of the anndata.settings attribute to check
    default
        Fallback default value (lowest priority)
    """
    if override is not None:
        return override
    return get_setting(setting_name, default=default)


def _create_formatter_context(
    adata: AnnData,
    *,
    depth: int = 0,
    max_depth: int | None = None,
    fold_threshold: int | None = None,
    max_items: int | None = None,
    max_lazy_categories: int | None = None,
) -> FormatterContext:
    """Create a FormatterContext with settings resolution.

    Parameters with function overrides use _resolve_setting() (override > settings > default).
    Settings-only parameters use get_setting() directly (settings > default).
    """
    return FormatterContext(
        depth=depth,
        # Overridable parameters (passed to generate_repr_html)
        max_depth=_resolve_setting(max_depth, "repr_html_max_depth", DEFAULT_MAX_DEPTH),
        fold_threshold=_resolve_setting(
            fold_threshold, "repr_html_fold_threshold", DEFAULT_FOLD_THRESHOLD
        ),
        max_items=_resolve_setting(max_items, "repr_html_max_items", DEFAULT_MAX_ITEMS),
        max_lazy_categories=_resolve_setting(
            max_lazy_categories,
            "repr_html_max_lazy_categories",
            DEFAULT_MAX_LAZY_CATEGORIES,
        ),
        # Settings-only parameters (not overridable at call time)
        max_categories=get_setting(
            "repr_html_max_categories", default=DEFAULT_MAX_CATEGORIES
        ),
        max_string_length=get_setting(
            "repr_html_max_string_length", default=DEFAULT_MAX_STRING_LENGTH
        ),
        unique_limit=get_setting(
            "repr_html_unique_limit", default=DEFAULT_UNIQUE_LIMIT
        ),
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
) -> str:
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
    if not get_setting("repr_html_enabled", default=True):
        return f"<pre>{escape_html(repr(adata))}</pre>"

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

    # Generate unique container ID
    container_id = _container_id or f"anndata-repr-{uuid.uuid4().hex[:8]}"

    # Build HTML parts
    parts = []

    # CSS and JS only at top level
    if depth == 0:
        parts.append(get_css())

    # Calculate field name column width based on content
    max_field_width = get_setting(
        "repr_html_max_field_width", default=DEFAULT_MAX_FIELD_WIDTH
    )
    field_width = _calculate_field_name_width(adata, max_field_width)

    # Get type column width from settings
    type_width = get_setting("repr_html_type_width", default=DEFAULT_TYPE_WIDTH)

    # Container with computed column widths as CSS variables
    style = f"--anndata-name-col-width: {field_width}px; --anndata-type-col-width: {type_width}px;"
    parts.append(
        f'<div class="anndata-repr" id="{container_id}" data-depth="{depth}" style="{style}">'
    )

    # No-CSS fallback: visible by default, hidden by CSS when styles load.
    # This helps users who open nbconvert-executed notebooks in JupyterLab,
    # where <style> tags are stripped from untrusted notebooks.
    if depth == 0:
        # Use only the first line of repr (shape info) to avoid bloating HTML
        summary = escape_html(repr(adata).split("\n", 1)[0])
        parts.append(
            '<div class="anndata-repr__nocss">'
            "<pre>"
            f"{summary}\n\n"
            "<b>Styled representation not available.</b>\n"
            "JupyterLab strips &lt;style&gt; tags from untrusted notebooks. "
            "This happens when a notebook was executed outside Jupyter "
            "(e.g. via <code>nbconvert --execute</code>) or transferred from "
            "another user.\n\n"
            "Run `jupyter trust notebook.ipynb` and reload to fix this.\n"
            "</pre>"
            "</div>"
        )

    # Header (with search box integrated on the right)
    if show_header:
        parts.append(
            _render_header(
                adata, show_search=show_search and depth == 0, container_id=container_id
            )
        )

    # Index preview (only at top level)
    if depth == 0:
        parts.append(_render_index_preview(adata))

    # Sections container
    parts.append('<div class="anndata-repr__sections">')
    parts.extend(_render_all_sections(adata, context))
    parts.append("</div>")  # anndata-repr__sections

    # Footer with metadata (only at top level)
    if depth == 0:
        parts.append(_render_footer(adata))

    parts.append("</div>")  # anndata-repr

    # JavaScript (only at top level)
    if depth == 0:
        parts.append(get_javascript(container_id))

    return "\n".join(parts)


def _render_all_sections(
    adata: AnnData,
    context: FormatterContext,
) -> list[str]:
    """Render all standard and custom sections using AnnData.reduce().

    Uses ``reduce`` with ``DFS-pre`` order to traverse the AnnData element tree.
    Standard sections are rendered inside the reduce callback; custom sections
    and unknown sections are handled separately since ``reduce`` only visits
    the hardcoded AnnData attributes and has no extension mechanism.
    """
    from dataclasses import replace as dc_replace

    from anndata._core.raw import Raw
    from anndata.acc import (
        AdRef,
        GraphAcc,
        GraphMapAcc,
        LayerAcc,
        LayerMapAcc,
        MapAcc,
        MetaAcc,
        MultiAcc,
        MultiMapAcc,
        RefAcc,
    )

    custom_sections_after = _get_custom_sections_by_position(adata)

    # ── accumulator type ──────────────────────────────────────────────
    # Without enter/exit events on reduce, we must carry open-section
    # state through the accumulator so we can finalize the previous
    # section when the next one starts (and after reduce returns).
    #
    # This 4-tuple is the "pain point" described in the PR discussion:
    # reduce already knows section boundaries internally (it's the
    # `for attr_name in [...]` loop), but doesn't surface them to the
    # callback, so we have to re-derive them here.
    #
    # accumulator = (
    #   finished_sections: list[str],   # completed section HTML strings
    #   current_rows: list[str] | None, # entry rows being collected
    #   current_section: str | None,    # section name (for header/metadata)
    #   current_n_items: int,           # total items in section (for count)
    # )

    def _section_name_from_acc(ref_acc) -> str:
        """Derive section name from accessor.

        PAIN POINT: reduce doesn't pass the section name, only the
        accessor object. We have to repr() it and strip the 'A.' prefix,
        or isinstance-check to figure out what section we're in.
        """
        return repr(ref_acc).replace("A.", "")

    def _is_section_visit(ref_acc) -> bool:
        """Detect whether this is a section-level (parent) visit.

        PAIN POINT: reduce visits both sections and their children through
        the same callback. The only way to distinguish them is by
        isinstance-checking the accessor type hierarchy. This is fragile:
        LayerAcc is used for both X (k=None) and individual layer entries
        (k='counts'), so we must also check the key to avoid treating
        layer entries as new sections.
        """
        if isinstance(ref_acc, AdRef | MultiAcc | GraphAcc):
            return False
        if isinstance(ref_acc, LayerAcc):
            # LayerAcc with k=None is X (handled by _is_x), k!=None is a leaf
            return False
        return isinstance(
            ref_acc,
            MetaAcc | LayerMapAcc | MultiMapAcc | GraphMapAcc,
        )

    def _is_x(ref_acc) -> bool:
        """Check if this is the X section (LayerAcc with k=None)."""
        return isinstance(ref_acc, LayerAcc) and ref_acc.k is None

    def _finalize_section(
        finished: list[str],
        section_name: str,
        rows: list[str],
        n_items: int,
    ) -> None:
        """Wrap up collected rows into a complete section HTML string.

        PAIN POINT: this logic must be called in two places:
        1. Inside the callback, when the next section starts
        2. After reduce returns, for the last section
        A proper enter/exit API would eliminate this duplication.
        """
        from . import get_section_doc_url
        from .core import get_section_tooltip

        doc_url = get_section_doc_url(section_name)
        tooltip = get_section_tooltip(section_name)
        if section_name == "obs":
            tooltip = "Observation annotations"
        elif section_name == "var":
            tooltip = "Variable annotations"

        if n_items == 0:
            finished.append(render_empty_section(section_name, doc_url, tooltip))
        else:
            count_str = (
                f"({n_items} columns)"
                if section_name in ("obs", "var")
                else None
            )
            finished.append(
                render_section(
                    section_name,
                    "\n".join(rows),
                    n_items=n_items,
                    doc_url=doc_url,
                    tooltip=tooltip,
                    should_collapse=n_items > context.fold_threshold,
                    count_str=count_str,
                )
            )

        # Insert custom sections after this standard section
        if section_name in custom_sections_after:
            finished.extend(
                _render_custom_section(adata, sf, context)
                for sf in custom_sections_after[section_name]
            )

    def _render_via_reduce(elem, *, accumulate, ref_acc):
        finished, current_rows, current_section, current_n_items = accumulate

        # ── uns (ref_acc=None, dict) ──────────────────────────────────
        # PAIN POINT: uns and raw are passed with ref_acc=None and are
        # not descended into by reduce. We must handle them as opaque
        # blobs and fall back to the dedicated renderers.
        if ref_acc is None and isinstance(elem, dict):
            if current_rows is not None:
                _finalize_section(finished, current_section, current_rows, current_n_items)
            try:
                finished.append(_render_uns_section(adata, context))
            except Exception as e:  # noqa: BLE001
                finished.append(_render_error_entry("uns", str(e)))
            if "uns" in custom_sections_after:
                finished.extend(
                    _render_custom_section(adata, sf, context)
                    for sf in custom_sections_after["uns"]
                )
            return finished, None, None, 0

        # ── raw (ref_acc=None, Raw or None) ───────────────────────────
        # Same issue as uns: reduce passes raw as an opaque blob.
        if ref_acc is None:
            if isinstance(elem, Raw):
                try:
                    finished.append(_render_raw_section(adata, context))
                except Exception as e:  # noqa: BLE001
                    finished.append(_render_error_entry("raw", str(e)))
                if "raw" in custom_sections_after:
                    finished.extend(
                        _render_custom_section(adata, sf, context)
                        for sf in custom_sections_after["raw"]
                    )
            # raw is None (no raw data) — nothing to render
            return finished, None, None, 0

        # ── X section (special rendering) ─────────────────────────────
        if _is_x(ref_acc):
            try:
                finished.append(render_x_entry(adata, context))
            except Exception as e:  # noqa: BLE001
                finished.append(_render_error_entry("X", str(e)))
            if "X" in custom_sections_after:
                finished.extend(
                    _render_custom_section(adata, sf, context)
                    for sf in custom_sections_after["X"]
                )
            return finished, None, None, 0

        # ── Section entry (parent visit in DFS-pre) ──────────────────
        if _is_section_visit(ref_acc):
            # Finalize previous section if one was open
            # PAIN POINT: previous section's exit is detected here,
            # inside the *next* section's entry. Mixed concerns.
            if current_rows is not None:
                _finalize_section(finished, current_section, current_rows, current_n_items)

            section_name = _section_name_from_acc(ref_acc)
            # PAIN POINT: reduce passes the container elem but we need the
            # *entry count*, which differs by section type:
            #   DataFrame → len(df.columns), not len(df) (which is n_rows)
            #   Mapping   → len(mapping) (number of keys)
            # A dedicated enter event could pass this directly.
            if isinstance(ref_acc, MetaAcc):
                import pandas as pd
                n_items = len(elem.columns) if isinstance(elem, pd.DataFrame) else 0
            else:
                try:
                    n_items = len(elem)
                except TypeError:
                    n_items = 0
            return finished, [], section_name, n_items

        # ── Leaf visit (entry within a section) ──────────────────────
        # PAIN POINT: we must derive the section name and key from
        # ref_acc. For MetaAcc children (obs/var columns), the key is
        # in the AdRef. For mapping children, it's in the RefAcc.
        if current_rows is not None and current_section is not None:
            # Determine the entry key
            if isinstance(ref_acc, AdRef):
                key = ref_acc.idx
            elif isinstance(ref_acc, (MultiAcc, GraphAcc, LayerAcc)):
                key = ref_acc.k
            else:
                key = str(ref_acc)

            # Truncation: skip entries beyond max_items
            entry_index = len(current_rows)
            if entry_index >= context.max_items:
                if entry_index == context.max_items:
                    remaining = current_n_items - context.max_items
                    current_rows.append(render_truncation_indicator(remaining))
                return accumulate

            # Format via the registry (same as the dedicated renderers)
            try:
                section_context = dc_replace(context, section=current_section)
                key_context = dc_replace(section_context, key=key)
                output = formatter_registry.format_value(elem, key_context)
                append_type = current_section not in ("obs", "var")
                current_rows.append(
                    _render_entry_row(str(key), output, append_type_html=append_type)
                )
            except Exception as e:  # noqa: BLE001
                current_rows.append(
                    f'<div class="anndata-entry--error">Error: {escape_html(str(e))}</div>'
                )

        return finished, current_rows, current_section, current_n_items

    # ── Run reduce ────────────────────────────────────────────────────
    try:
        finished, leftover_rows, last_section, last_n_items = adata.reduce(
            _render_via_reduce,
            init=([], None, None, 0),
            order="DFS-pre",
        )
    except Exception as e:  # noqa: BLE001
        # If reduce itself fails, fall back to an error message
        return [f'<div class="anndata-entry--error">reduce failed: {escape_html(str(e))}</div>']

    # PAIN POINT: the *last* section never gets a "next section entry"
    # to trigger finalization. We must duplicate the finalize call here.
    if leftover_rows is not None and last_section is not None:
        _finalize_section(finished, last_section, leftover_rows, last_n_items)

    # ── Custom sections at end (no specific position) ─────────────────
    # LIMITATION: reduce has no concept of custom/unknown sections.
    # These must be handled entirely outside reduce.
    if None in custom_sections_after:
        finished.extend(
            _render_custom_section(adata, sf, context)
            for sf in custom_sections_after[None]
        )

    # ── Unknown sections (extension attributes) ──────────────────────
    # LIMITATION: reduce hardcodes its attribute list. Extension packages
    # (TreeData adding .obst/.vart) are invisible to reduce. We must
    # detect and render them separately, same as before.
    unknown_sections = _detect_unknown_sections(adata)
    if unknown_sections:
        finished.append(_render_unknown_sections(unknown_sections))

    return finished


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

    for section_name in formatter_registry.get_registered_sections():
        formatter = formatter_registry.get_section_formatter(section_name)
        if formatter is None:
            continue

        # Skip standard sections (they're handled separately)
        if section_name in SECTION_ORDER:
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
) -> str:
    """Render a custom section using its registered formatter."""
    try:
        entries = formatter.get_entries(adata, context)
    except Exception as e:  # noqa: BLE001
        # Intentional broad catch: custom formatters shouldn't crash the entire repr
        from .._warnings import warn

        warn(
            f"Custom section formatter '{formatter.section_name}' failed: {e}",
            UserWarning,
        )
        return ""

    if not entries:
        return ""

    n_items = len(entries)
    section_name = formatter.section_name

    # Render entries (with truncation)
    rows = []
    for i, entry in enumerate(entries):
        if i >= context.max_items:
            rows.append(render_truncation_indicator(n_items - context.max_items))
            break
        rows.append(render_formatted_entry(entry, section_name))

    # Use render_section for consistent structure
    return render_section(
        getattr(formatter, "display_name", section_name),
        "\n".join(rows),
        n_items=n_items,
        doc_url=getattr(formatter, "doc_url", None),
        tooltip=getattr(formatter, "tooltip", ""),
        should_collapse=n_items > context.fold_threshold,
        section_id=section_name,
    )


def _render_header(
    adata: AnnData, *, show_search: bool = False, container_id: str = ""
) -> str:
    """Render the header with type, shape, badges, and optional search box."""
    parts = ['<div class="anndata-header">']

    # Type name - allow for extension types
    type_name = type(adata).__name__
    parts.append(f'<span class="anndata-header__type">{escape_html(type_name)}</span>')

    # Shape
    shape_str = f"{format_number(adata.n_obs)} obs × {format_number(adata.n_vars)} vars"
    parts.append(f'<span class="anndata-header__shape">{shape_str}</span>')

    # Badges - use render_badge() helper
    if is_view(adata):
        parts.append(render_badge("View", CSS_BADGE_VIEW))

    if is_backed(adata):
        backing = get_backing_info(adata)
        filename = backing.get("filename", "")
        format_str = backing.get("format", "")
        status = "Open" if backing.get("is_open") else "Closed"
        parts.append(render_badge(f"{format_str} ({status})", CSS_BADGE_BACKED))
        # Inline file path (full path, no truncation)
        if filename:
            parts.append(
                f'<span class="anndata-header__filepath">{escape_html(filename)}</span>'
            )

    if is_lazy_adata(adata):
        lazy_info = get_lazy_backing_info(adata)
        lazy_format = lazy_info.get("format", "")
        if lazy_format:
            parts.append(render_badge(f"Lazy ({lazy_format})", CSS_BADGE_LAZY))
        else:
            parts.append(render_badge("Lazy", CSS_BADGE_LAZY))
        # Show file path for lazy AnnData (similar to backed)
        lazy_filename = lazy_info.get("filename", "")
        if lazy_filename:
            path_style = (
                "font-family:ui-monospace,monospace;font-size:11px;"
                "color:var(--anndata-text-secondary, #6c757d);"
            )
            parts.append(
                f'<span class="anndata-header__filepath" style="{path_style}">'
                f"{escape_html(lazy_filename)}"
                f"</span>"
            )

    # Check for extension type (not standard AnnData)
    if type_name != "AnnData":
        parts.append(render_badge(type_name, CSS_BADGE_EXTENSION))

    # README icon if uns["README"] exists with a string
    readme_content = adata.uns.get("README") if hasattr(adata, "uns") else None
    if isinstance(readme_content, str) and readme_content.strip():
        # Check max README size setting (0 means no limit)
        max_readme_size = get_setting(
            "repr_html_max_readme_size", default=DEFAULT_MAX_README_SIZE
        )
        original_len = len(readme_content)
        if max_readme_size > 0 and original_len > max_readme_size:
            # Truncate and add note
            readme_content = readme_content[:max_readme_size]
            truncation_note = (
                f"\n\n---\n*README truncated: showing {max_readme_size:,} of "
                f"{original_len:,} characters*"
            )
            readme_content += truncation_note

        escaped_readme = escape_html(readme_content)
        # Truncate for no-JS tooltip (first 500 chars)
        tooltip_text = readme_content[:TOOLTIP_TRUNCATE_LENGTH]
        if len(readme_content) > TOOLTIP_TRUNCATE_LENGTH:
            tooltip_text += "..."
        escaped_tooltip = escape_html(tooltip_text)

        parts.append(
            f'<span class="anndata-readme__icon" '
            f'data-readme="{escaped_readme}" '
            f'title="{escaped_tooltip}" '
            f'role="button" tabindex="0" aria-label="View README">'
            f"ⓘ"
            f"</span>"
        )

    # Search box on the right (spacer pushes it right) - use render_search_box() helper
    if show_search:
        parts.append('<span class="anndata-spacer"></span>')
        parts.append(render_search_box(container_id))

    parts.append("</div>")
    return "\n".join(parts)


def _render_footer(adata: AnnData) -> str:
    """Render the footer with version and memory info."""
    parts = ['<div class="anndata-footer">']

    # Version
    version = get_anndata_version()
    parts.append(f"<span>anndata v{version}</span>")

    # Memory usage
    try:
        mem_bytes = adata.__sizeof__()
        mem_str = format_memory_size(mem_bytes)
        parts.append(f'<span title="Estimated memory usage">~{mem_str}</span>')
    except Exception:  # noqa: BLE001
        # Broad catch: __sizeof__ recursively calls into user data which could raise anything
        pass

    parts.append("</div>")
    return "\n".join(parts)


def _render_index_preview(adata: AnnData) -> str:
    """Render preview of obs_names and var_names."""
    parts = ['<div class="anndata-header__index">']

    # obs_names preview
    obs_preview = format_index_preview(adata.obs_names, DEFAULT_PREVIEW_ITEMS)
    parts.append(f"<div><strong>obs_names:</strong> {obs_preview}</div>")

    # var_names preview
    var_preview = format_index_preview(adata.var_names, DEFAULT_PREVIEW_ITEMS)
    parts.append(f"<div><strong>var_names:</strong> {var_preview}</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_max_depth_indicator(adata: AnnData) -> str:
    """Render indicator when max depth is reached."""
    n_obs = getattr(adata, "n_obs", "?")
    n_vars = getattr(adata, "n_vars", "?")
    return f'<div class="anndata-depth-limit">AnnData ({format_number(n_obs)} × {format_number(n_vars)}) - max depth reached</div>'
