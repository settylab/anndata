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

from markupsafe import Markup

from .._repr_constants import (
    CSS_BADGE_BACKED,
    CSS_BADGE_LAZY,
    CSS_BADGE_VIEW,
    CSS_ENTRY,
    CSS_TEXT_MUTED,
    NOT_SERIALIZABLE_MSG,
    STYLE_HIDDEN,
)
from .utils import escape_html, join_markup, sanitize_css_color, trusted_html

_EMPTY = Markup()
_COPY_OPEN = Markup('<button class="anndata-entry__copy" style="')
_COPY_DATA = Markup('" data-copy="')
_COPY_TITLE = Markup('" title="')
_COPY_LABEL = Markup('" aria-label="')
_COPY_CLOSE = Markup('"></button>')
_NAME_OPEN = Markup(
    '<span class="anndata-entry__name" style="display:inline-block;min-width:var(--anndata-name-col-width,100px);vertical-align:top">'
    '<span class="anndata-entry__name-inner">'
    '<span class="anndata-entry__name-text" title="'
)
_NAME_MID = Markup('">')
_TYPE_OPEN = Markup('<span class="')
_TYPE_TITLE = Markup('" title="')
_CAT_ITEM_OPEN = Markup('<span class="anndata-categories__item">')
_CAT_NAME_OPEN = Markup("<span>")
_CAT_ITEM_CLOSE = Markup("</span></span>")
_DOT_OPEN = Markup('<span class="anndata-categories__dot" style="background:')
_DOT_CLOSE = Markup(';"></span>')
_SPAN_CLOSE = Markup("</span>")
_NAME_CLOSE = Markup("</span></span>")
_DIV_OPEN = Markup('<div class="')
_DETAILS_OPEN = Markup('<details class="')
_ATTR_KEY = Markup('" data-key="')
_ATTR_DTYPE = Markup('" data-dtype="')
_TAG_END = Markup('">')
_ROW_OPEN_SUMMARY = Markup('"><summary class="anndata-entry__summary">')
_CATEGORY_SEP = Markup('<span class="anndata-categories__sep">, </span>')


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
    Opening tag(s) with class and data attributes
    """
    # Build CSS class string
    classes = [CSS_ENTRY]
    if extra_classes:
        classes.append(extra_classes)
    if has_warnings:
        classes.append("warning")
    if is_error:
        classes.append("error")
    css_class = " ".join(classes)

    return join_markup([
        _DETAILS_OPEN if has_expandable_content else _DIV_OPEN,
        escape_html(css_class),
        _ATTR_KEY,
        escape_html(key),
        _ATTR_DTYPE,
        escape_html(dtype),
        _ROW_OPEN_SUMMARY if has_expandable_content else _TAG_END,
    ])


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
    HTML string for warning icon, or empty string if no warnings.
    """
    if not warnings and not is_not_serializable:
        return Markup()

    # Build the tooltip message
    if is_not_serializable:
        if warnings:
            # "Not serializable: reason1; reason2"
            reasons = "; ".join(warnings)
            title = f"{NOT_SERIALIZABLE_MSG}: {reasons}"
        else:
            # Just "Not serializable to H5AD/Zarr"
            title = NOT_SERIALIZABLE_MSG
    else:
        # Independent warnings joined with ";"
        title = "; ".join(warnings)

    return Markup('<span class="anndata-entry__warning" title="{}">(!)</span>').format(
        escape_html(title)
    )


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
    HTML string for the search box

    Example
    -------
    >>> from markupsafe import Markup
    >>> container_id = "spatialdata-123"
    >>> parts = [Markup('<div class="anndata-header">')]
    >>> parts.append(Markup('<span class="anndata-header__type">SpatialData</span>'))
    >>> parts.append(Markup('<span class="anndata-spacer"></span>'))  # Spacer
    >>> parts.append(render_search_box(container_id))
    >>> parts.append(Markup("</div>"))
    >>> html = Markup("").join(parts)
    """
    search_id = f"{container_id}-search" if container_id else "anndata-search"
    return Markup(
        '<span class="anndata-search__box" style="{style}">'
        '<input type="text" id="{id}" name="{id}" '
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
    ).format(style=STYLE_HIDDEN, id=escape_html(search_id))


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
    HTML string for the copy button

    Example
    -------
    >>> name = "gene_expression"
    >>> from markupsafe import Markup
    >>> html = Markup("<span>{}</span>{}").format(
    ...     name, render_copy_button(name, "Copy name")
    ... )
    """
    tooltip_html = escape_html(tooltip)
    return join_markup([
        _COPY_OPEN,
        escape_html(STYLE_HIDDEN),
        _COPY_DATA,
        escape_html(text),
        _COPY_TITLE,
        tooltip_html,
        _COPY_LABEL,
        tooltip_html,
        _COPY_CLOSE,
    ])


def _render_wrap_button(css_class: str) -> Markup:
    """Render a wrap toggle button with the specified CSS class.

    Internal helper used by render_categories_wrap_button and render_columns_wrap_button.
    """
    return Markup(
        '<button class="{}" style="display:none" title="Expand to multi-line view">▼</button>'
    ).format(css_class)


def render_categories_wrap_button() -> Markup:
    """Render a button to toggle category list between single-line and multi-line.

    Returns
    -------
    HTML string for the wrap button (▼ expands, ▲ collapses)
    """
    return _render_wrap_button("anndata-categories__wrap")


def render_columns_wrap_button() -> Markup:
    """Render a button to toggle column list between single-line and multi-line.

    Returns
    -------
    HTML string for the wrap button (▼ expands, ▲ collapses)
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
    HTML string with muted styling
    """
    return Markup('<span class="%s">%s</span>') % (CSS_TEXT_MUTED, escape_html(text))


def render_nested_content(html_content: str) -> Markup:
    """Render nested/expanded content inside an expandable entry.

    The entry must have been opened with ``has_expandable_content=True``
    (which makes it a ``<details>`` with ``<summary>``).  This function
    closes the ``<summary>`` and adds the nested content.  The caller
    must close the entry with ``</details>``.

    Parameters
    ----------
    html_content
        The HTML content to display when expanded

    Returns
    -------
    HTML closing the summary and wrapping nested content
    """
    return Markup(
        "</summary>"
        '<div class="anndata-entry__nested-content" style="margin-left:1.5em">'
        '<div class="anndata-entry__expanded">{}</div>'
        "</div>"
    ).format(trusted_html(html_content))


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
    HTML string for the badge

    Example
    -------
    >>> badge = render_badge("Zarr", "anndata-badge--backed", "Backed by Zarr store")
    """
    title_attr = (
        Markup(' title="{}"').format(escape_html(tooltip)) if tooltip else Markup()
    )
    # Always include base class, optionally add variant
    css_class = f"anndata-badge {variant}".strip() if variant else "anndata-badge"
    return Markup('<span class="{}"{}>{}</span>').format(
        css_class, title_attr, escape_html(text)
    )


def render_header_badges(
    *,
    is_view: bool = False,
    is_backed: bool = False,
    is_lazy: bool = False,
    backing_path: str | None = None,
    backing_format: str | None = None,
    is_open: bool | None = None,
) -> Markup:
    """
    Render standard header badges for view/backed/lazy status.

    When the object is backed or lazy and ``backing_path`` is given, the path
    is shown next to the badges.

    Parameters
    ----------
    is_view
        Whether this is a view
    is_backed
        Whether this is backed by a file
    is_lazy
        Whether this uses lazy loading (experimental read_lazy)
    backing_path
        Path to the backing file
    backing_format
        Format of the backing file ("H5AD", "Zarr", etc.)
    is_open
        For backed objects: whether the backing file is open (``None``: unknown)

    Returns
    -------
    HTML string with badges

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
            render_badge("View", CSS_BADGE_VIEW, "This is a view of another object")
        )
    if is_backed:
        label = backing_format or "Backed"
        if is_open is not None:
            label += " (Open)" if is_open else " (Closed)"
        tooltip = f"Backed by {backing_path}" if backing_path else "Backed mode"
        parts.append(render_badge(label, CSS_BADGE_BACKED, tooltip))
    if is_lazy:
        label = f"Lazy ({backing_format})" if backing_format else "Lazy"
        parts.append(
            render_badge(label, CSS_BADGE_LAZY, "Lazy loading (experimental read_lazy)")
        )
    if (is_backed or is_lazy) and backing_path:
        parts.append(
            Markup('<span class="anndata-header__filepath">{}</span>').format(
                escape_html(backing_path)
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
    HTML string for the cell div
    """
    escaped_name = escape_html(name)
    return join_markup([
        _NAME_OPEN,
        escaped_name,
        _NAME_MID,
        escaped_name,
        _SPAN_CLOSE,
        render_copy_button(name, "Copy name"),
        _NAME_CLOSE,
    ])


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
    parts = [Markup('<span class="anndata-categories">')]
    for i, cat in enumerate(categories[:max_cats]):
        if i > 0:
            parts.append(_CATEGORY_SEP)
        color = colors[i] if colors and i < len(colors) else None
        dot = _EMPTY
        if color:
            # Sanitize color to prevent CSS injection
            safe_color = sanitize_css_color(str(color))
            if safe_color:
                dot = join_markup([_DOT_OPEN, escape_html(safe_color), _DOT_CLOSE])
            # Skip color dot if color is invalid/unsafe
        parts.extend([
            _CAT_ITEM_OPEN,
            dot,
            _CAT_NAME_OPEN,
            escape_html(str(cat)),
            _CAT_ITEM_CLOSE,
        ])

    # Calculate total hidden: from max_cats truncation + lazy truncation
    hidden_from_max_cats = max(0, len(categories) - max_cats)
    total_hidden = hidden_from_max_cats + n_hidden

    if total_hidden > 0:
        parts.append(
            Markup('<span class="{}">...+{}</span>').format(
                CSS_TEXT_MUTED, total_hidden
            )
        )
    parts.append(_SPAN_CLOSE)
    return join_markup(parts)


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
    type_html
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
    append_type_html
        If True, type_html is appended below type_name instead of replacing it

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
    type_html: str | Markup | None = None
    tooltip: str = ""
    warnings: list[str] = field(default_factory=list)
    is_not_serializable: bool = False
    has_columns_list: bool = False
    has_categories_list: bool = False
    append_type_html: bool = False


def render_entry_type_cell(config: TypeCellConfig) -> Markup:
    """Render the type cell for an entry row.

    This is a unified helper that handles all type cell variations:
    - Type label with optional tooltip
    - Custom type_html (as replacement or appended content)
    - Warning icon
    - Expand/wrap buttons

    The type_html and append_type_html config fields control content rendering:

    1. No type_html: Shows type_name in a styled span
       ``<span class="anndata-dtype--X">type_name</span>``

    2. type_html with append_type_html=False (default): type_html REPLACES type_name
       Used for fully custom type content (e.g., category swatches instead of text)

    3. type_html with append_type_html=True: type_html is shown BELOW type_name
       Used to add extra content while keeping the type label
       (e.g., showing category list below "categorical" label)

    Parameters
    ----------
    config
        TypeCellConfig object with all rendering options

    Returns
    -------
    HTML string for the complete type cell

    Examples
    --------
    >>> config = TypeCellConfig(
    ...     type_name="ndarray (100, 50) float32",
    ...     css_class="anndata-dtype--ndarray",
    ...     tooltip="Dense array",
    ... )
    >>> html = render_entry_type_cell(config)
    """
    type_name = config.type_name
    css_class = config.css_class
    type_html = config.type_html
    tooltip = config.tooltip
    warnings = config.warnings
    is_not_serializable = config.is_not_serializable
    has_columns_list = config.has_columns_list
    has_categories_list = config.has_categories_list
    append_type_html = config.append_type_html

    parts = [
        Markup(
            '<span class="anndata-entry__type" style="display:inline-block;min-width:var(--anndata-type-col-width,180px);vertical-align:top">'
        )
    ]

    # Type content: handle different cases
    if type_html and not append_type_html:
        # type_html replaces the type label entirely
        parts.append(trusted_html(type_html))
    elif tooltip:
        parts.extend([
            _TYPE_OPEN,
            escape_html(css_class),
            _TYPE_TITLE,
            escape_html(tooltip),
            _NAME_MID,
            escape_html(type_name),
            _SPAN_CLOSE,
        ])
    else:
        parts.extend([
            _TYPE_OPEN,
            escape_html(css_class),
            _NAME_MID,
            escape_html(type_name),
            _SPAN_CLOSE,
        ])

    # Warning icon
    parts.append(
        render_warning_icon(warnings or [], is_not_serializable=is_not_serializable)
    )

    # Wrap buttons
    if has_columns_list:
        parts.append(render_columns_wrap_button())
    if has_categories_list:
        parts.append(render_categories_wrap_button())

    # Appended type_html (for custom inline rendering below the type)
    if type_html and append_type_html:
        parts.append(
            Markup('<span class="anndata-entry__custom">{}</span>').format(
                trusted_html(type_html)
            )
        )

    parts.append(_SPAN_CLOSE)
    return join_markup(parts)


def render_entry_preview_cell(
    preview_html: str | None = None,
    preview_text: str | None = None,
) -> Markup:
    """Render the preview cell (third column) for an entry row.

    Formatters are responsible for producing complete preview content.
    This function just wraps it in the appropriate cell element.

    Parameters
    ----------
    preview_html
        Raw HTML content for preview (highest priority)
    preview_text
        Plain text preview (will be escaped and muted)

    Returns
    -------
    HTML string for the preview cell
    """
    parts = [
        Markup(
            '<span class="anndata-entry__preview" style="display:inline-block;vertical-align:top">'
        )
    ]

    if preview_html:
        parts.append(trusted_html(preview_html))
    elif preview_text:
        parts.append(render_muted_span(preview_text))

    parts.append(_SPAN_CLOSE)
    return join_markup(parts)
