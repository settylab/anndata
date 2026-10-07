"""
Section-specific renderers for AnnData HTML representation.

This module contains renderers for each section type:
- DataFrame sections (obs, var)
- Mapping sections (obsm, varm, layers, obsp, varp)
- Uns section (unstructured annotations)
- Raw section (unprocessed data)
- Unknown sections (extension attributes)

Error Handling Policy
---------------------
This module uses broad exception handling (``except Exception``) in several places.
This is intentional - user data may contain arbitrary objects that raise unexpected
exceptions when accessed. The repr should never crash; instead it should:

1. Use ``# noqa: BLE001`` to acknowledge the broad catch
2. Provide a fallback (e.g., show "?" or skip the problematic item)
3. Continue rendering the rest of the representation

This ensures a partially-rendered repr is always better than a crashed cell.
"""

from __future__ import annotations

import inspect
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import TYPE_CHECKING

from markupsafe import Markup

from .._repr_constants import (
    CSS_DTYPE_ANNDATA,
    CSS_DTYPE_UNKNOWN,
    INTERNAL_ANNDATA_ATTRS,
    UNIQUE_COUNT_BUDGET,
)
from .._types import AnnDataElem
from ..utils import get_literal_members
from . import (
    get_section_doc_url,
)
from .components import (
    TypeCellConfig,
    render_entry_preview_cell,
    render_entry_row_open,
    render_entry_type_cell,
    render_name_cell,
    render_nested_content,
)
from .core import (
    get_section_tooltip,
    pluralize,
    render_details_section,
    render_empty_section,
    render_error_section,
    render_formatted_entry,
    render_index_preview,
    render_section,
    render_truncation_indicator,
    render_x_entry,
)
from .environment import get_env, get_macros
from .registry import (
    FormattedEntry,
    FormattedOutput,
    extract_uns_type_hint,
    formatter_registry,
)
from .utils import (
    format_number,
)

if TYPE_CHECKING:
    import pandas as pd

    from anndata import AnnData

    from .registry import FormatterContext


def _render_entry_row(
    key: str,
    output: FormattedOutput,
    *,
    append_type_markup: bool = False,
    preview_note: str | None = None,
) -> Markup:
    """Render an entry row for DataFrame, mapping, or uns sections.

    Key validation is handled by FormatterRegistry.format_value() via context.key,
    so output already contains any key-related warnings and serialization flags.

    Parameters
    ----------
    key
        Entry key/name to display
    output
        FormattedOutput from a TypeFormatter (already includes key validation)
    append_type_markup
        If True, append type_markup below type_name (for mapping entries)
    preview_note
        Optional note to prepend to preview (for type hints in uns)

    Returns
    -------
    HTML string for the entry row (and optional expandable content row)
    """
    entry = FormattedEntry(key=key, output=output)
    return render_formatted_entry(
        entry,
        append_type_markup=append_type_markup,
        preview_note=preview_note,
    )


# -----------------------------------------------------------------------------
# DataFrame Section (obs, var)
# -----------------------------------------------------------------------------


def _render_dataframe_section(
    section: str,
    df: pd.DataFrame,
    context: FormatterContext,
) -> Markup:
    """Render obs or var section."""
    n_cols = len(df.columns)

    # Doc URL and tooltip for this section
    doc_url = get_section_doc_url(section)
    tooltip = get_section_tooltip(section)

    if n_cols == 0:
        return render_empty_section(section, doc_url, tooltip)

    # Set section for section-specific formatters (e.g., LazyColumnFormatter)
    section_context = replace(context, section=section)

    # Unique counts cost ~10ms per column per million rows: only count the
    # first columns until the total scanned values reach the budget.
    n_rows = len(df)
    n_unique_cols = UNIQUE_COUNT_BUDGET // n_rows if n_rows else n_cols

    # Render entries (with truncation)
    rows = []
    for i, col_name in enumerate(df.columns):
        if i >= context.max_items:
            rows.append(render_truncation_indicator(n_cols - context.max_items))
            break
        col = df[col_name]
        col_context = replace(section_context, key=col_name)
        if i >= n_unique_cols:
            col_context = replace(col_context, unique_limit=0)
        output = formatter_registry.format_value(col, col_context)
        rows.append(_render_entry_row(col_name, output))

    return render_section(
        section,
        Markup("\n").join(rows),
        n_items=n_cols,
        doc_url=doc_url,
        tooltip=tooltip,
        should_collapse=n_cols > context.fold_threshold,
        count_str=f"({pluralize(n_cols, 'column')})",
    )


# -----------------------------------------------------------------------------
# Mapping Section (obsm, varm, layers, obsp, varp)
# -----------------------------------------------------------------------------


def _render_mapping_section(
    section: str,
    mapping: Mapping[str | None, object],
    context: FormatterContext,
) -> Markup:
    """Render obsm, varm, layers, obsp, varp sections."""
    if mapping is None:
        return Markup("")

    # `.X` is stored as `layers[None]`; it gets its own row, so hide it here
    hide_x = section == "layers" and None in mapping

    # Get count without creating full list (O(1) for most mappings)
    n_items = len(mapping) - hide_x

    # Doc URL and tooltip for this section
    doc_url = get_section_doc_url(section)
    tooltip = get_section_tooltip(section)

    if n_items == 0:
        return render_empty_section(section, doc_url, tooltip)

    # Set section for section-specific formatters (e.g., DaskArrayFormatter)
    section_context = replace(context, section=section)

    # Render entries (with truncation) - iterate lazily, stop at max_items
    rows = []
    keys = (k for k in mapping if k is not None)
    for i, key in enumerate(keys):
        if i >= context.max_items:
            rows.append(render_truncation_indicator(n_items - context.max_items))
            break
        value = mapping[key]
        key_context = replace(section_context, key=key)
        output = formatter_registry.format_value(value, key_context)
        rows.append(_render_entry_row(key, output, append_type_markup=True))

    return render_section(
        section,
        Markup("\n").join(rows),
        n_items=n_items,
        doc_url=doc_url,
        tooltip=tooltip,
        should_collapse=n_items > context.fold_threshold,
    )


# -----------------------------------------------------------------------------
# Uns Section (unstructured annotations)
# -----------------------------------------------------------------------------


def _render_uns_section(
    uns: Mapping[str, object],
    context: FormatterContext,
) -> Markup:
    """Render the uns section with special handling."""
    # Get count without creating full list (O(1) for dict)
    n_items = len(uns)

    # Doc URL and tooltip
    doc_url = get_section_doc_url("uns")
    tooltip = get_section_tooltip("uns")

    if n_items == 0:
        return render_empty_section("uns", doc_url, tooltip)

    # Render entries (with truncation) - iterate lazily, stop at max_items
    rows = []
    for i, key in enumerate(uns.keys()):
        if i >= context.max_items:
            rows.append(render_truncation_indicator(n_items - context.max_items))
            break
        value = uns[key]
        rows.append(_render_uns_entry(key, value, context))

    return render_section(
        "uns",
        Markup("\n").join(rows),
        n_items=n_items,
        doc_url=doc_url,
        tooltip=tooltip,
        should_collapse=n_items > context.fold_threshold,
    )


def _render_uns_entry(
    key: str,
    value: object,
    context: FormatterContext,
) -> Markup:
    """Render a single uns entry with special type handling.

    Rendering priority:
    1. Custom TypeFormatter (may handle type hints, color lists, AnnData)
    2. Unhandled type hint (show import suggestion)
    3. Default formatter
    """
    # Section and key for section-restricted formatters and key-based
    # detection (e.g., color lists)
    key_context = replace(context, section="uns", key=key)

    # 1. Try formatter first - handles type hints, color lists, AnnData
    output = formatter_registry.format_value(value, key_context)

    # If a custom formatter produced preview_markup, use it directly
    if output.preview_markup:
        return _render_entry_row(key, output)

    # 2. Check for unhandled type hint (basic formatter matched, not custom)
    type_hint, cleaned_value = extract_uns_type_hint(value)
    if type_hint is not None:
        # Type hint present but no custom formatter - show import suggestion
        package_name = type_hint.split(".")[0] if "." in type_hint else type_hint
        cleaned_output = formatter_registry.format_value(cleaned_value, key_context)
        return _render_entry_row(
            key,
            cleaned_output,
            preview_note=f"[{type_hint}] (import {package_name} to enable)",
        )

    # 3. Use formatter output
    return _render_entry_row(key, output)


# -----------------------------------------------------------------------------
# Unknown Sections (extension attributes)
# -----------------------------------------------------------------------------


def _detect_unknown_sections(adata: AnnData) -> list[tuple[str, str]]:
    """Detect mapping-like attributes not surfaced by the standard section list.

    Methods are recognized statically and skipped, and a ``cached_property``
    that has not been computed yet is skipped too, so detection does not
    trigger expensive computations. Plain properties are evaluated (they are
    expected to be cheap, like any attribute access).

    Returns list of (attr_name, type_description) tuples for unknown sections.
    """
    # See INTERNAL_ANNDATA_ATTRS docstring for why the internal list is explicit.
    known = set(get_literal_members(AnnDataElem)) | INTERNAL_ANNDATA_ATTRS

    # Also exclude sections with registered custom formatters
    # (including should_show=False ones that suppress display).
    known |= set(formatter_registry.get_registered_sections())

    unknown = []
    for attr in dir(adata):
        if attr.startswith("_") or attr in known:
            continue

        try:
            static = inspect.getattr_static(adata, attr)
        except AttributeError:
            continue  # only reachable via a dynamic __getattr__: not a data slot
        if isinstance(static, _NON_DATA_TYPES):
            continue

        try:
            # Probing must not surface warnings to the user, e.g. from
            # deprecated properties (MuData's `mod_names`, AnnData's `isview`)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = getattr(adata, attr)
            # Check if it's a data container (mapping-like or has keys())
            if isinstance(val, Mapping) or (
                hasattr(val, "keys")
                and hasattr(val, "__getitem__")
                and not callable(val)
            ):
                # Get type description
                type_name = type(val).__name__
                try:
                    type_desc = f"{type_name} ({pluralize(len(val), 'item')})"
                except Exception:  # noqa: BLE001
                    type_desc = type_name
                unknown.append((attr, type_desc))
        except Exception:  # noqa: BLE001
            # If we can't even access the attribute, note it as inaccessible
            unknown.append((attr, "inaccessible"))

    return unknown


# Attributes that are statically known not to be data sections: methods, and
# not-yet-computed cached properties (computing them could be expensive).
# A computed ``cached_property`` lives in the instance ``__dict__``, so
# ``inspect.getattr_static`` returns its value rather than the descriptor.
_NON_DATA_TYPES = (
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    staticmethod,
    classmethod,
    cached_property,
)


def _render_unknown_sections(unknown_sections: list[tuple[str, str]]) -> Markup:
    """Render a section showing unknown/unrecognized attributes."""
    rows: list[Markup] = [
        render_formatted_entry(
            FormattedEntry(
                key=attr_name,
                output=FormattedOutput(
                    type_name=type_desc,
                    css_class=CSS_DTYPE_UNKNOWN,
                    tooltip="Unrecognized attribute",
                ),
            )
        )
        for attr_name, type_desc in unknown_sections
    ]

    return render_details_section(
        "unknown",
        "other",
        f"({len(unknown_sections)})",
        Markup('<div class="anndata-section__entries">{}</div>').format(
            Markup("\n").join(rows)
        ),
        is_open=False,
        extra_classes="anndata-sec-unknown",
    )


# -----------------------------------------------------------------------------
# Raw Section
# -----------------------------------------------------------------------------


def _safe_get_attr(obj: object, attr: str, default: object = "?") -> object:
    """Safely get an attribute with fallback.

    Parameters
    ----------
    obj
        Object to get attribute from
    attr
        Attribute name
    default
        Default value if attribute is missing or access raises exception

    Returns
    -------
    Attribute value or default
    """
    try:
        val = getattr(obj, attr, None)
        return val if val is not None else default
    except Exception:  # noqa: BLE001
        return default


def _get_raw_meta_parts(raw: object) -> list[str]:
    """Build meta info parts for raw section.

    Parameters
    ----------
    raw
        Raw object to extract metadata from

    Returns
    -------
    List of metadata strings like ["var: 5 cols", "varm: 2"]
    """
    meta_parts = []
    try:
        if hasattr(raw, "var") and raw.var is not None and len(raw.var.columns) > 0:
            meta_parts.append(f"var: {pluralize(len(raw.var.columns), 'column')}")
    except Exception:  # noqa: BLE001
        pass
    try:
        if hasattr(raw, "varm") and raw.varm is not None and len(raw.varm) > 0:
            meta_parts.append(f"varm: {pluralize(len(raw.varm), 'item')}")
    except Exception:  # noqa: BLE001
        pass
    return meta_parts


def _render_raw_section(
    raw: object,
    context: FormatterContext,
) -> Markup:
    """Render the raw section as a single expandable row.

    The raw section shows unprocessed data that was saved before filtering/normalization.
    It contains raw.X (the matrix), raw.var (variable annotations), and raw.varm
    (multi-dimensional variable annotations).

    Unlike the main AnnData, raw shares obs with the parent but has its own var
    (which may have more variables than the filtered main data).

    Rendered as a single row with an expand button (no section header).
    When expanded, shows a full AnnData-like repr for Raw contents (X, var, varm).
    The depth parameter prevents infinite recursion.
    """
    if raw is None:
        return Markup("")

    # Safely get dimensions with fallbacks
    n_obs = _safe_get_attr(raw, "n_obs", "?")
    n_vars = _safe_get_attr(raw, "n_vars", "?")

    # Check if we can expand (same logic as nested AnnData)
    can_expand = context.depth < context.max_depth - 1

    # Build meta info string safely
    meta_parts = _get_raw_meta_parts(raw)
    meta_text = ", ".join(meta_parts) if meta_parts else ""

    # Single row with raw info
    type_str = f"{format_number(n_obs)} obs × {format_number(n_vars)} vars"
    row_parts: list[Markup] = [
        render_entry_row_open("raw", "Raw", has_expandable_content=can_expand),
        render_name_cell("raw"),
        render_entry_type_cell(
            TypeCellConfig(type_name=type_str, css_class=CSS_DTYPE_ANNDATA)
        ),
        render_entry_preview_cell(preview_text=meta_text),
    ]
    if can_expand:
        nested_html = _generate_raw_repr_html(raw, context.child("raw"))
        wrapped_html = Markup(get_macros().nested_anndata_wrapper(nested_html))
        row_parts.append(render_nested_content(wrapped_html))
        row_parts.append(Markup("</details>"))
    else:
        row_parts.append(Markup("</div>"))

    entry_markup = Markup("\n").join(row_parts)
    return Markup(
        get_env().get_template("raw_section.j2").render(entry_markup=entry_markup)
    )


def _generate_raw_repr_html(
    raw,
    context: FormatterContext,
) -> Markup:
    """Generate HTML repr for a Raw object.

    This renders X, var, and varm sections similar to AnnData,
    but without obs, obsm, layers, obsp, varp, uns, or raw sections.

    Parameters
    ----------
    raw
        Raw object to render
    context
        FormatterContext with depth, max_depth, fold_threshold, max_items
    """
    n_obs = _safe_get_attr(raw, "n_obs", "?")
    n_vars = _safe_get_attr(raw, "n_vars", "?")
    shape_str = f"{format_number(n_obs)} obs × {format_number(n_vars)} vars"

    sections: list[Markup] = []

    # X section - show matrix info (with error handling)
    try:
        if hasattr(raw, "X") and raw.X is not None:
            sections.append(render_x_entry(raw, context))
    except Exception as e:  # noqa: BLE001
        sections.append(render_error_section("X", str(e)))

    try:
        if hasattr(raw, "var") and raw.var is not None and len(raw.var.columns) > 0:
            var_context = replace(context, adata_ref=None, section="var")
            sections.append(_render_dataframe_section("var", raw.var, var_context))
    except Exception as e:  # noqa: BLE001
        sections.append(render_error_section("var", str(e)))

    try:
        if hasattr(raw, "varm") and raw.varm is not None and len(raw.varm) > 0:
            varm_context = replace(context, adata_ref=None, section="varm")
            sections.append(_render_mapping_section("varm", raw.varm, varm_context))
    except Exception as e:  # noqa: BLE001
        sections.append(render_error_section("varm", str(e)))

    return Markup(
        get_env()
        .get_template("raw_repr.j2")
        .render(
            container_id=f"anndata-raw-{uuid.uuid4().hex[:8]}",
            shape_str=shape_str,
            index_preview=render_index_preview(raw),
            sections=sections,
        )
    )
