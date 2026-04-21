"""
Core rendering primitives for AnnData HTML representation.

This module contains shared rendering functions used by both:
- html.py (main orchestration)
- sections.py (section-specific renderers)

By extracting these to a separate module, we avoid circular imports
between html.py and sections.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from markupsafe import Markup

from .._repr_constants import (
    CSS_DTYPE_CATEGORY,
    CSS_DTYPE_DATAFRAME,
    CSS_TEXT_ERROR,
    CSS_TEXT_MUTED,
)
from .environment import get_env
from .registry import formatter_registry
from .utils import escape_html, format_number

if TYPE_CHECKING:
    from .registry import FormattedEntry, FormatterContext


def render_section(  # noqa: PLR0913
    name: str,
    entries: Markup,
    *,
    n_items: int,
    doc_url: str | None = None,
    tooltip: str = "",
    should_collapse: bool = False,
    section_id: str | None = None,
    count_str: str | None = None,
) -> Markup:
    """
    Render a complete section with header and content.

    This is a public API for packages building their own _repr_html_.
    It is also used internally for consistency.

    Parameters
    ----------
    name
        Display name for the section header (e.g., 'images', 'tables')
    entries
        Trusted HTML (``markupsafe.Markup``) for the section body.
        Typically produced by joining per-entry ``Markup`` values,
        e.g. ``Markup("\\n").join(render_formatted_entry(e) for e in entries)``.
    n_items
        Number of items (used for empty check and default count string)
    doc_url
        URL for the help link (? icon)
    tooltip
        Tooltip text for the help link
    should_collapse
        Whether this section should start collapsed
    section_id
        ID for the section in data-section attribute (defaults to name)
    count_str
        Custom count string for header (defaults to "(N items)")

    Returns
    -------
    ``Markup`` HTML for the complete section.

    Examples
    --------
    ::

        from markupsafe import Markup

        from anndata._repr import (
            CSS_DTYPE_NDARRAY,
            FormattedEntry,
            FormattedOutput,
            render_formatted_entry,
            render_section,
        )

        rows = [
            render_formatted_entry(
                FormattedEntry(
                    key=key,
                    output=FormattedOutput(
                        type_name=info["type"], css_class=CSS_DTYPE_NDARRAY
                    ),
                )
            )
            for key, info in items.items()
        ]

        html = render_section(
            "images",
            Markup("\\n").join(rows),
            n_items=len(items),
            doc_url="https://docs.example.com/images",
            tooltip="Image data",
        )
    """
    if section_id is None:
        section_id = name
    if count_str is None:
        count_str = f"({n_items} items)"

    return Markup(
        get_env()
        .get_template("section.j2")
        .render(
            name=name,
            count_str=count_str,
            doc_url=doc_url,
            tooltip=tooltip,
            should_collapse=should_collapse,
            section_id=section_id,
            n_items=n_items,
            entries=entries,
        )
    )


def render_empty_section(
    name: str,
    doc_url: str | None = None,
    tooltip: str = "",
) -> Markup:
    """Render an empty section indicator."""
    return render_section(name, Markup(""), n_items=0, doc_url=doc_url, tooltip=tooltip)


def render_truncation_indicator(remaining: int) -> Markup:
    """Render a truncation indicator."""
    return Markup(
        f'<div class="anndata-section__truncated">... and {format_number(remaining)} more</div>'
    )


def get_section_tooltip(section: str) -> str:
    """Get tooltip text for a section."""
    tooltips = {
        "obs": "Observation (cell) annotations",
        "var": "Variable (gene) annotations",
        "uns": "Unstructured annotation",
        "obsm": "Multi-dimensional observation annotations",
        "varm": "Multi-dimensional variable annotations",
        "layers": "Additional data layers (same shape as X)",
        "obsp": "Pairwise observation annotations",
        "varp": "Pairwise variable annotations",
        "raw": "Raw data (original unprocessed)",
    }
    return tooltips.get(section, "")


def render_x_entry(obj: object, context: FormatterContext) -> Markup:
    """Render X as a single compact entry row.

    Works with AnnData, Raw, and any object with an X attribute.
    Handles missing or broken X attributes gracefully.
    """
    parts: list[Markup] = [
        Markup('<div class="anndata-x__entry">'),
        Markup("<span>X</span>"),
    ]

    try:
        X = obj.X
    except Exception as e:  # noqa: BLE001
        error_msg = f"error: {type(e).__name__}"
        parts.append(
            Markup(
                f'<span class="{CSS_TEXT_MUTED}"><em>({escape_html(error_msg)})</em></span>'
            )
        )
        parts.append(Markup("</div>"))
        return Markup("\n").join(parts)

    if X is None:
        parts.append(Markup("<span><em>None</em></span>"))
    else:
        try:
            output = formatter_registry.format_value(X, context)
            parts.append(
                Markup(
                    f'<span class="{output.css_class}">{escape_html(output.type_name)}</span>'
                )
            )
        except Exception as e:  # noqa: BLE001
            error_msg = f"error formatting: {type(e).__name__}"
            parts.append(
                Markup(
                    f'<span class="{CSS_TEXT_MUTED}"><em>({escape_html(error_msg)})</em></span>'
                )
            )

    parts.append(Markup("</div>"))
    return Markup("\n").join(parts)


def render_formatted_entry(
    entry: FormattedEntry,
    section: str = "",
    *,
    extra_warnings: list[str] | None = None,
    append_type_markup: bool = False,
    preview_note: str | None = None,
) -> Markup:
    """
    Render a FormattedEntry as a table row.

    This is the unified entry renderer used both internally and as a public API
    for packages building their own _repr_html_.

    Parameters
    ----------
    entry
        A FormattedEntry containing the key and FormattedOutput
    section
        Optional section name (used for meta column rendering)
    extra_warnings
        Additional warnings to display (e.g., key validation warnings)
    append_type_markup
        If True, append type_markup below type_name instead of replacing it.
        Used for mapping entries (obsm, varm, etc.) to show extra content.
    preview_note
        Optional note to prepend to preview text (for type hints in uns)

    Returns
    -------
    ``Markup`` HTML for the table row(s)

    Examples
    --------
    ::

        from anndata._repr import (
            CSS_DTYPE_ANNDATA,
            CSS_DTYPE_NDARRAY,
            FormattedEntry,
            FormattedOutput,
            render_formatted_entry,
        )

        entry = FormattedEntry(
            key="my_array",
            output=FormattedOutput(
                type_name="ndarray (100, 50) float32",
                css_class=CSS_DTYPE_NDARRAY,
                tooltip="My custom array",
                warnings=["Some warning"],
            ),
        )
        html = render_formatted_entry(entry)

    With expandable nested content::

        from markupsafe import Markup

        nested_html = generate_repr_html(adata, depth=1)
        entry = FormattedEntry(
            key="cell_table",
            output=FormattedOutput(
                type_name="AnnData (150 × 30)",
                css_class=CSS_DTYPE_ANNDATA,
                expanded_markup=Markup(nested_html),
            ),
        )
        html = render_formatted_entry(entry)

    With key validation warnings::

        entry = FormattedEntry(
            key="bad/key",
            output=FormattedOutput(...),
        )
        html = render_formatted_entry(
            entry, extra_warnings=["Contains '/' (deprecated)"]
        )

    With explicit error::

        entry = FormattedEntry(
            key="broken_data",
            output=FormattedOutput(
                type_name="MyType",
                error="Failed to load: file not found",
            ),
        )
        html = render_formatted_entry(entry)
    """
    output = entry.output
    all_warnings = (extra_warnings or []) + list(output.warnings)
    has_error = output.error is not None or not output.is_serializable
    has_expandable_content = output.expanded_markup is not None
    has_categories = output.css_class == CSS_DTYPE_CATEGORY and bool(
        output.preview_markup
    )
    has_columns_list = output.css_class == CSS_DTYPE_DATAFRAME and bool(
        output.preview_markup
    )

    preview_markup = output.preview_markup
    preview_text = output.preview
    if output.error and not preview_markup:
        preview_markup = Markup(
            f'<span class="{CSS_TEXT_ERROR}">{escape_html(output.error)}</span>'
        )

    if preview_note and preview_text:
        preview_text = f"{preview_note} {preview_text}"
    elif preview_note:
        preview_text = preview_note

    rendered = (
        get_env()
        .get_template("entry.j2")
        .render(
            entry_key=entry.key,
            type_name=output.type_name,
            css_class=output.css_class,
            type_markup=output.type_markup,
            tooltip=output.tooltip,
            all_warnings=all_warnings,
            is_not_serializable=not output.is_serializable,
            has_error=has_error,
            has_expandable_content=has_expandable_content,
            has_columns_list=has_columns_list,
            has_categories_list=has_categories,
            append_type_markup=append_type_markup,
            preview_markup=preview_markup,
            preview_text=preview_text,
            expanded_markup=output.expanded_markup,
        )
    )
    return Markup(rendered)
