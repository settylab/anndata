"""Pluggable section registry for AnnData.

Provides the infrastructure for :func:`~anndata.extensions.register_section`:
container classes, view handling, and property descriptors that let external
packages add new sections to AnnData without subclassing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .views import view_update

if TYPE_CHECKING:
    from anndata import AnnData

    from .._repr.registry import FormattedOutput, FormatterContext


def _axis_len(value: Any, dim: int) -> int | None:
    """Get length of value along a dimension, or None if not applicable."""
    if hasattr(value, "shape"):
        shape = value.shape
        if dim < len(shape):
            return shape[dim]
    return None


@dataclass(frozen=True)
class SectionSpec:
    """Complete specification for a registered section.

    Created by :func:`register_section` from the decorated class.
    """

    name: str
    """Attribute name on AnnData (e.g., ``"obst"``)."""
    alignment: tuple[Literal["obs", "var"], ...]
    """Axes each dimension is aligned to. Empty tuple for unaligned."""
    io_key: str
    """Key used in h5ad/zarr files."""

    # Optional callbacks extracted from the section class
    value_type: type | None = None
    validate_fn: Callable[[str, Any], None] | None = None
    subset_fn: Callable[[Any, Any], Any] | None = None
    serialize_fn: Callable[[Any], Any] | None = None
    deserialize_fn: Callable[[Any], Any] | None = None
    repr_entry_fn: Callable[[str, Any, FormatterContext], FormattedOutput] | None = None

    # Repr metadata
    section_after: str | None = None
    section_tooltip: str = ""
    section_doc_url: str | None = None


class SectionMapping(MutableMapping):
    """Container for a registered section's data.

    Validates values on assignment using the section's spec (type check,
    alignment validation, custom validator).
    """

    def __init__(
        self, parent: AnnData, spec: SectionSpec, data: dict | None = None
    ) -> None:
        self._parent = parent
        self._spec = spec
        self._data: dict[str, Any] = data if data is not None else {}

    def __repr__(self) -> str:
        return f"{self._spec.name}: {', '.join(map(repr, self._data.keys()))}"

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        # Type check
        if self._spec.value_type is not None and not isinstance(
            value, self._spec.value_type
        ):
            msg = (
                f"Values in {self._spec.name!r} must be {self._spec.value_type.__name__}, "
                f"got {type(value).__name__}"
            )
            raise TypeError(msg)
        # Alignment validation
        self._validate_alignment(key, value)
        # Custom validation
        if self._spec.validate_fn is not None:
            self._spec.validate_fn(key, value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def _validate_alignment(self, key: str, value: Any) -> None:
        """Check that value dimensions match the expected axes."""
        for i, axis in enumerate(self._spec.alignment):
            expected = self._parent.n_obs if axis == "obs" else self._parent.n_vars
            actual = _axis_len(value, i)
            if actual is not None and actual != expected:
                n_name = "n_obs" if axis == "obs" else "n_vars"
                msg = (
                    f"Value for {self._spec.name}[{key!r}] has shape[{i}]={actual}, "
                    f"expected {expected} ({n_name})"
                )
                raise ValueError(msg)

    def copy(self) -> dict[str, Any]:
        """Return a deep copy of the underlying data."""
        return {
            k: copy(v) if not hasattr(v, "copy") else v.copy()
            for k, v in self._data.items()
        }


class SectionMappingView(Mapping):
    """Read-only view of a registered section that subsets on access.

    Writing triggers copy-on-write via anndata's view_update mechanism.
    """

    def __init__(
        self,
        parent_mapping: SectionMapping,
        parent_view: AnnData,
        obs_idx: Any,
        var_idx: Any,
    ) -> None:
        self._parent_mapping = parent_mapping
        self._parent = parent_view
        self._spec = parent_mapping._spec
        self._obs_idx = obs_idx
        self._var_idx = var_idx

    def __repr__(self) -> str:
        return f"{self._spec.name} (view): {', '.join(map(repr, self._parent_mapping._data.keys()))}"

    def __getitem__(self, key: str) -> Any:
        value = self._parent_mapping[key]
        if not self._spec.alignment:
            return value  # unaligned, no subsetting
        return self._subset_value(value)

    def __setitem__(self, key: str, value: Any) -> None:
        from .._warnings import ImplicitModificationWarning
        from ..utils import warn

        warn(
            f"Setting element `.{self._spec.name}[{key!r}]` of view, "
            "initializing view as actual.",
            ImplicitModificationWarning,
        )
        with view_update(self._parent, self._spec.name, ()) as new_mapping:
            new_mapping[key] = value

    def __delitem__(self, key: str) -> None:
        from .._warnings import ImplicitModificationWarning
        from ..utils import warn

        if key not in self:
            msg = f"{key!r} not found in view of {self._spec.name}"
            raise KeyError(msg)
        warn(
            f"Removing element `.{self._spec.name}[{key!r}]` of view, "
            "initializing view as actual.",
            ImplicitModificationWarning,
        )
        with view_update(self._parent, self._spec.name, ()) as new_mapping:
            del new_mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parent_mapping)

    def __len__(self) -> int:
        return len(self._parent_mapping)

    def __contains__(self, key: object) -> bool:
        return key in self._parent_mapping

    def _subset_value(self, value: Any) -> Any:
        """Subset a value according to the alignment tuple."""
        idx = self._build_index()
        if self._spec.subset_fn is not None:
            return self._spec.subset_fn(value, idx)
        # Default subsetting: handle N-dimensional alignment
        # anndata's _subset is designed for ≤2D, so for higher dims
        # we do the indexing directly.
        import numpy as np

        from anndata.compat import IndexManager

        if isinstance(idx, tuple) and len(idx) > 2:
            # Convert IndexManagers to numpy arrays
            resolved = []
            for ix in idx:
                if isinstance(ix, IndexManager):
                    resolved.append(np.asarray(ix))
                else:
                    resolved.append(ix)
            # Use np.ix_ for fancy indexing on non-slice dims
            fancy_dims = [
                i for i, ix in enumerate(resolved) if not isinstance(ix, slice)
            ]
            if fancy_dims:
                # Build an open mesh for fancy-indexed dims
                fancy_arrs = [resolved[i] for i in fancy_dims]
                mesh = np.ix_(*fancy_arrs)
                # Build the full index tuple
                full_idx = list(resolved)
                for mi, di in enumerate(fancy_dims):
                    full_idx[di] = mesh[mi]
                return value[tuple(full_idx)]
            return value[tuple(resolved)]
        # ≤2D: use anndata's _subset
        from .index import _subset

        return _subset(value, idx)

    def _build_index(self) -> tuple:
        """Build the index tuple from alignment and view indices."""
        indices = []
        for axis in self._spec.alignment:
            if axis == "obs":
                indices.append(self._obs_idx)
            elif axis == "var":
                indices.append(self._var_idx)
        return tuple(indices)

    def copy(self) -> dict[str, Any]:
        """Copy with subsetting applied."""
        return {
            k: self[k].copy() if hasattr(self[k], "copy") else self[k] for k in self
        }


class SectionProperty:
    """Descriptor for registered sections on AnnData.

    Creates ephemeral SectionMapping / SectionMappingView on access,
    similar to AlignedMappingProperty for built-in sections.
    """

    def __init__(self, spec: SectionSpec) -> None:
        self.spec = spec

    def __get__(self, obj: AnnData | None, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        if not obj.is_view:
            data = getattr(obj, f"_{self.spec.name}", None)
            if data is None:
                data = {}
                setattr(obj, f"_{self.spec.name}", data)
            return SectionMapping(obj, self.spec, data)
        # View: create subsetting view
        parent = obj._adata_ref
        parent_mapping = getattr(parent, self.spec.name)
        return SectionMappingView(parent_mapping, obj, obj._oidx, obj._vidx)

    def __set__(self, obj: AnnData, value: Mapping[str, Any] | None) -> None:
        if value is None:
            value = {}
        if isinstance(value, (SectionMapping, SectionMappingView)) or isinstance(
            value, Mapping
        ):
            value = dict(value)
        # Validate all values via SectionMapping
        mapping = SectionMapping(obj, self.spec, {})
        for k, v in value.items():
            mapping[k] = v  # validates each
        if obj.is_view:
            obj._init_as_actual(obj.copy())
        setattr(obj, f"_{self.spec.name}", mapping._data)

    def __delete__(self, obj: AnnData) -> None:
        setattr(obj, f"_{self.spec.name}", {})
