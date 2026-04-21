"""
Reusable UI components for HTML representation.

This module provides building blocks for creating consistent HTML representations:
- Warning/error icons with tooltips
- Search box with filter toggles
- Fold/expand icons for collapsible sections
- Copy-to-clipboard buttons
- Status badges (view, backed, sparse, etc.)

These components are designed to be used by both anndata's internal repr
and by external packages (MuData, SpatialData, TreeData) that want to
build compatible representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

from markupsafe import Markup

from .._repr_constants import CSS_ENTRY, STYLE_HIDDEN
from .environment import get_env
from .utils import sanitize_css_color


@cache
def _macros():
    return get_env().get_template("_macros.j2").module


def render_entry_row_open(
    key: str,
    dtype: str,
    *,
    has_warnings: bool = False,
    is_error: bool = False,
    has_expandable_content: bool = False,
    extra_classes: str = "",
) -> Markup:
    """Render the opening tag for an entry row.

    For regular entries, returns ``<div class="anndata-entry ...">``.
    For expandable entries, returns ``<details class="anndata-entry ..."><summary ...>``
    so the whole row acts as the disclosure toggle.

    Parameters
    ----------
    key
        The entry key (column name, field name, etc.)
    dtype
        The data type string (for data-dtype attribute)
    has_warnings
        Whether the entry has warnings
    is_error
        Whether the entry has errors (not serializable, invalid key)
    has_expandable_content
        Whether this entry has nested content (uses ``<details>``/``<summary>``)
    extra_classes
        Additional CSS classes to include

    Returns
    -------
    ``Markup`` HTML for the opening tag(s) with class and data attributes.
    """
    classes = [CSS_ENTRY]
    if extra_classes:
        classes.append(extra_classes)
    if has_warnings:
        classes.append("warning")
    if is_error:
        classes.append("error")
    css_class = " ".join(classes)
    return Markup(
        _macros().row_open(key, dtype, css_class, has_expandable_content)
    )


def render_warning_icon(
    warnings: list[str], *, is_not_serializable: bool = False
) -> Markup:
    """Render warning icon with tooltip if there are warnings or serialization issues.

    Parameters
    ----------
    warnings
        List of warning messages to show in tooltip.
    is_not_serializable
        If True, prepends "Not serializable to H5AD/Zarr" to warnings.

    Returns
    -------
    ``Markup`` HTML for warning icon, or empty ``Markup`` if no warnings.
    """
    return Markup(_macros().warning_icon(warnings or [], is_not_serializable))


def render_search_box(container_id: str = "") -> Markup:
    """
    Render a search box with filter indicator and search mode toggles.

    The search box is hidden by default and shown when JavaScript is enabled.
    It filters entries across all sections by key, type, or content.
    Includes toggle buttons for case-sensitive search and regex mode.

    Parameters
    ----------
    container_id
        Unique ID for the container (used for label association)

    Returns
    -------
    ``Markup`` HTML for the search box.
    """
    search_id = f"{container_id}-search" if container_id else "anndata-search"
    return Markup(
        '<span class="anndata-search__box" style="{style}">'
        '<input type="text" id="{sid}" name="{sid}" '
        'class="anndata-search__input" '
        'placeholder="Search..." aria-label="Search fields">'
        '<span class="anndata-search__toggles">'
        '<button type="button" class="anndata-search__toggle anndata-search__toggle--case" '
        'title="Match case" aria-label="Match case" aria-pressed="false">Aa</button>'
        '<button type="button" class="anndata-search__toggle anndata-search__toggle--regex" '
        'title="Use regular expression" aria-label="Use regular expression" aria-pressed="false">.*</button>'
        "</span>"
        "</span>"
        '<span class="anndata-search__indicator"></span>'
    ).format(style=STYLE_HIDDEN, sid=search_id)


def render_copy_button(text: str, tooltip: str = "Copy") -> Markup:
    """
    Render a copy-to-clipboard button.

    The button is hidden by default and shown when JavaScript is enabled.
    When clicked, it copies the specified text to the clipboard.

    Parameters
    ----------
    text
        The text to copy when clicked
    tooltip
        Tooltip text (default: "Copy")

    Returns
    -------
    ``Markup`` HTML for the copy button

    Example
    -------
    >>> name = "gene_expression"
    >>> html = f"<span>{name}</span>{render_copy_button(name, 'Copy name')}"
    """
    return Markup(_macros().copy_button(text, tooltip))


def _render_wrap_button(css_class: str) -> Markup:
    """Render a wrap toggle button with the specified CSS class.

    Internal helper used by render_categories_wrap_button and render_columns_wrap_button.
    """
    return Markup(_macros().wrap_button(css_class))


def render_categories_wrap_button() -> Markup:
    """Render a button to toggle category list between single-line and multi-line.

    Returns
    -------
    ``Markup`` HTML for the wrap button (▼ expands, ▲ collapses)
    """
    return _render_wrap_button("anndata-categories__wrap")


def render_columns_wrap_button() -> Markup:
    """Render a button to toggle column list between single-line and multi-line.

    Returns
    -------
    ``Markup`` HTML for the wrap button (▼ expands, ▲ collapses)
    """
    return _render_wrap_button("anndata-columns__wrap")


def render_muted_span(text: str) -> Markup:
    """Render text in a muted span (gray color).

    Parameters
    ----------
    text
        Text to render (will be HTML-escaped)

    Returns
    -------
    ``Markup`` HTML with muted styling
    """
    return Markup(_macros().muted_span(text))


def render_nested_content(html_content: str | Markup) -> Markup:
    """Render nested/expanded content inside an expandable entry.

    The entry must have been opened with ``has_expandable_content=True``
    (which makes it a ``<details>`` with ``<summary>``).  This function
    closes the ``<summary>`` and adds the nested content.  The caller
    must close the entry with ``</details>``.

    Parameters
    ----------
    html_content
        The trusted HTML (``Markup`` preferred; ``str`` is wrapped) to
        display when expanded.

    Returns
    -------
    ``Markup`` HTML closing the summary and wrapping nested content.
    """
    body = html_content if isinstance(html_content, Markup) else Markup(html_content)
    return Markup(_macros().nested_content(body))


def render_badge(
    text: str,
    variant: str = "",
    tooltip: str = "",
) -> Markup:
    """
    Render a badge (pill-shaped label).

    Parameters
    ----------
    text
        Badge text
    variant
        Variant class for styling. Built-in variants:
        - "" (default gray)
        - "anndata-badge--view" (blue, for views)
        - "anndata-badge--backed" (orange, for backed mode)
        - "anndata-badge--sparse" (green, for sparse matrices)
        - "anndata-badge--dask" (purple, for Dask arrays)
        - "anndata-badge--extension" (for extension types)
    tooltip
        Tooltip text on hover

    Returns
    -------
    ``Markup`` HTML for the badge

    Example
    -------
    >>> badge = render_badge("Zarr", "anndata-badge--backed", "Backed by Zarr store")
    """
    return Markup(_macros().badge(text, variant, tooltip))


def render_header_badges(
    *,
    is_view: bool = False,
    is_backed: bool = False,
    is_lazy: bool = False,
    backing_path: str | None = None,
    backing_format: str | None = None,
) -> Markup:
    """
    Render standard header badges for view/backed/lazy status.

    Parameters
    ----------
    is_view
        Whether this is a view
    is_backed
        Whether this is backed by a file
    is_lazy
        Whether this uses lazy loading (experimental read_lazy)
    backing_path
        Path to the backing file (for tooltip)
    backing_format
        Format of the backing file ("H5AD", "Zarr", etc.)

    Returns
    -------
    ``Markup`` HTML with badges

    Example
    -------
    >>> badges = render_header_badges(
    ...     is_backed=True,
    ...     backing_path="/data/sample.zarr",
    ...     backing_format="Zarr",
    ... )
    """
    parts: list[Markup] = []
    if is_view:
        parts.append(
            render_badge(
                "View", "anndata-badge--view", "This is a view of another object"
            )
        )
    if is_backed:
        tooltip = f"Backed by {backing_path}" if backing_path else "Backed mode"
        label = backing_format or "Backed"
        parts.append(render_badge(label, "anndata-badge--backed", tooltip))
    if is_lazy:
        parts.append(
            render_badge(
                "Lazy", "anndata-badge--lazy", "Lazy loading (experimental read_lazy)"
            )
        )
    return Markup("").join(parts)


def render_name_cell(name: str) -> Markup:
    """Render a name cell with copy button and tooltip for truncated names.

    The structure uses flexbox so the copy button stays visible even when
    the name text overflows and shows ellipsis.

    Parameters
    ----------
    name
        The field name to display

    Returns
    -------
    ``Markup`` HTML for the cell span.
    """
    return Markup(_macros().name_cell(name))


def render_category_list(
    categories: list,
    colors: list[str] | None,
    max_cats: int,
    *,
    n_hidden: int = 0,
) -> Markup:
    """Render a list of category values with optional color dots.

    Parameters
    ----------
    categories
        List of category values to display
    colors
        Optional list of colors matching categories
    max_cats
        Maximum number of categories to show
    n_hidden
        Number of additional hidden categories (for lazy truncation).
        These are added to any truncation from max_cats.

    Returns
    -------
    HTML string for the category list
    """
    visible = categories[:max_cats]
    items: list[tuple[str, str | None]] = []
    for i, cat in enumerate(visible):
        raw_color = colors[i] if colors and i < len(colors) else None
        safe_color = sanitize_css_color(str(raw_color)) if raw_color else None
        items.append((str(cat), safe_color))

    hidden_from_max_cats = max(0, len(categories) - max_cats)
    total_hidden = hidden_from_max_cats + n_hidden
    return Markup(_macros().category_list(items, total_hidden))


@dataclass
class TypeCellConfig:
    """Configuration for rendering a type cell.

    Groups the many parameters of render_entry_type_cell into a single object,
    making call sites cleaner and easier to understand.

    Attributes
    ----------
    type_name
        The type name to display (e.g., "ndarray (100, 50) float32")
    css_class
        CSS class for the type span (e.g., "anndata-dtype--ndarray")
    type_markup
        Optional custom HTML content for the type cell
    tooltip
        Optional tooltip for the type label
    warnings
        List of warning messages
    is_not_serializable
        Whether the data cannot be serialized to H5AD/Zarr
    has_columns_list
        Whether to show columns wrap button
    has_categories_list
        Whether to show categories wrap button
    append_type_markup
        If True, type_markup is appended below type_name instead of replacing it

    Examples
    --------
    >>> config = TypeCellConfig(
    ...     type_name="ndarray (100, 50) float32",
    ...     css_class="anndata-dtype--ndarray",
    ...     tooltip="Dense array",
    ... )
    >>> html = render_entry_type_cell(config)

    With warnings::

        >>> config = TypeCellConfig(
        ...     type_name="object",
        ...     css_class="anndata-dtype--object",
        ...     warnings=["Custom warning"],
        ...     is_not_serializable=True,
        ... )
    """

    type_name: str
    css_class: str
    type_markup: Markup | None = None
    tooltip: str = ""
    warnings: list[str] = field(default_factory=list)
    is_not_serializable: bool = False
    has_columns_list: bool = False
    has_categories_list: bool = False
    append_type_markup: bool = False


def render_entry_type_cell(config: TypeCellConfig) -> Markup:
    """Render the type cell for an entry row.

    This is a unified helper that handles all type cell variations:
    - Type label with optional tooltip
    - Custom type_markup (as replacement or appended content)
    - Warning icon
    - Expand/wrap buttons

    The type_markup and append_type_markup config fields control content rendering:

    1. No type_markup: Shows type_name in a styled span
       ``<span class="anndata-dtype--X">type_name</span>``

    2. type_markup with append_type_markup=False (default): type_markup REPLACES type_name
       Used for fully custom type content (e.g., category swatches instead of text)

    3. type_markup with append_type_markup=True: type_markup is shown BELOW type_name
       Used to add extra content while keeping the type label
       (e.g., showing category list below "categorical" label)

    Parameters
    ----------
    config
        TypeCellConfig object with all rendering options

    Returns
    -------
    ``Markup`` HTML for the complete type cell.
    """
    return Markup(
        _macros().type_cell(
            type_name=config.type_name,
            css_class=config.css_class,
            type_markup=config.type_markup,
            tooltip=config.tooltip,
            all_warnings=config.warnings,
            is_not_serializable=config.is_not_serializable,
            has_columns_list=config.has_columns_list,
            has_categories_list=config.has_categories_list,
            append_type_markup=config.append_type_markup,
        )
    )


def render_entry_preview_cell(
    preview_markup: Markup | None = None,
    preview_text: str | None = None,
) -> Markup:
    """Render the preview cell (third column) for an entry row.

    Formatters are responsible for producing complete preview content.
    This function just wraps it in the appropriate cell element.

    Parameters
    ----------
    preview_markup
        Trusted HTML (``Markup``) for preview (highest priority).
    preview_text
        Plain text preview (autoescaped and muted).

    Returns
    -------
    ``Markup`` HTML for the preview cell.
    """
    return Markup(
        _macros().preview_cell(
            preview_markup=preview_markup,
            preview_text=preview_text,
        )
    )
