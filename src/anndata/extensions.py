"""
Public API for extending AnnData functionality.

This module provides registration mechanisms for:

1. **Accessors** - Add custom namespaces to AnnData objects (e.g., `adata.myns.method()`)
2. **HTML Formatters** - Customize how types are displayed in Jupyter notebooks

Examples
--------
Register a custom accessor namespace::

    import anndata as ad
    from anndata.extensions import register_anndata_namespace


    @register_anndata_namespace("transform")
    class TransformAccessor:
        def __init__(self, adata: ad.AnnData):
            self._adata = adata

        def log1p(self):
            import numpy as np

            self._adata.X = np.log1p(self._adata.X)
            return self._adata


    # Usage: adata.transform.log1p()

Register a custom HTML formatter for a type::

    from anndata.extensions import register_formatter, TypeFormatter, FormattedOutput


    @register_formatter
    class MyArrayFormatter(TypeFormatter):
        priority = 100  # Higher = checked first

        def can_format(self, obj):
            return isinstance(obj, MyArrayType)

        def format(self, obj, context):
            return FormattedOutput(
                type_name=f"MyArray {obj.shape}",
                css_class="dtype-custom",
            )

Register a custom section formatter (for packages like TreeData, SpatialData)::

    from anndata.extensions import register_formatter, SectionFormatter
    from anndata.extensions import FormattedEntry, FormattedOutput


    @register_formatter
    class ObstSectionFormatter(SectionFormatter):
        section_name = "obst"
        after_section = "obsm"  # Position in display order

        def should_show(self, obj):
            return hasattr(obj, "obst") and len(obj.obst) > 0

        def get_entries(self, obj, context):
            return [
                FormattedEntry(
                    key=k,
                    output=FormattedOutput(type_name=f"Tree ({v.n_nodes} nodes)"),
                )
                for k, v in obj.obst.items()
            ]

See Also
--------
anndata._repr : Full documentation of the HTML representation system
"""

from __future__ import annotations

# Accessor registration (from PR #1870)
from anndata._core.extensions import register_anndata_namespace

# HTML representation formatters
from anndata._repr import (
    # Type hint utilities for tagged data
    UNS_TYPE_HINT_KEY,
    # Core formatter classes
    FormattedEntry,
    FormattedOutput,
    FormatterContext,
    FormatterRegistry,
    SectionFormatter,
    TypeFormatter,
    extract_uns_type_hint,
    # Global registry instance
    formatter_registry,
    # Registration function
    register_formatter,
)

__all__ = [  # noqa: RUF022  # organized by category, not alphabetically
    # Accessor registration
    "register_anndata_namespace",
    # HTML formatter registration
    "register_formatter",
    "TypeFormatter",
    "SectionFormatter",
    "FormattedOutput",
    "FormattedEntry",
    "FormatterContext",
    "FormatterRegistry",
    "formatter_registry",
    # Type hint utilities
    "extract_uns_type_hint",
    "UNS_TYPE_HINT_KEY",
]
