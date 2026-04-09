from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, get_type_hints, overload

from ..types import ExtensionNamespace
from ..utils import warn
from .anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from anndata._repr.registry import FormattedEntry, FormatterContext


# Based off of the extension framework in Polars
# https://github.com/pola-rs/polars/blob/main/py-polars/polars/api.py

__all__ = ["SectionSpec", "register_anndata_namespace", "register_section"]

# Protocol for accessors that provide section visualization
REPR_SECTION_METHOD = "_repr_section_"


# Reserved namespaces include accessors built into AnnData (currently there are none)
# and all current attributes of AnnData
_reserved_namespaces: set[str] = set(dir(AnnData))


class AccessorNameSpace[NameSpT: ExtensionNamespace](ExtensionNamespace):
    """Establish property-like namespace object for user-defined functionality."""

    def __init__(self, name: str, namespace: type[NameSpT]) -> None:
        self._accessor = name
        self._ns = namespace

    @overload
    def __get__[T](self, instance: None, cls: type[T]) -> type[NameSpT]: ...

    @overload
    def __get__[T](self, instance: T, cls: type[T]) -> NameSpT: ...

    def __get__[T](self, instance: T | None, cls: type[T]) -> NameSpT | type[NameSpT]:
        if instance is None:
            return self._ns

        ns_instance = self._ns(instance)  # type: ignore[call-arg]
        setattr(instance, self._accessor, ns_instance)
        return ns_instance


def _check_namespace_signature(ns_class: type) -> None:
    """Validate the signature of a namespace class for AnnData extensions.

    This function ensures that any class intended to be used as an extension namespace
    has a properly formatted `__init__` method such that:

    1. Accepts at least two parameters (self and adata)
    2. Has 'adata' as the name of the second parameter
    3. Has the second parameter properly type-annotated as 'AnnData' or any equivalent import alias

    The function performs runtime validation of these requirements before a namespace
    can be registered through the `register_anndata_namespace` decorator.

    Parameters
    ----------
    ns_class
        The namespace class to validate.

    Raises
    ------
    TypeError
        If the `__init__` method has fewer than 2 parameters (missing the AnnData parameter).
    AttributeError
        If the second parameter of `__init__` lacks a type annotation.
    TypeError
        If the second parameter of `__init__` is not named 'adata'.
    TypeError
        If the second parameter of `__init__` is not annotated as the 'AnnData' class.
    TypeError
        If both the name and type annotation of the second parameter are incorrect.

    """
    sig = inspect.signature(ns_class.__init__)
    params = list(sig.parameters.values())

    # Ensure there are at least two parameters (self and adata)
    if len(params) < 2:
        error_msg = "Namespace initializer must accept an AnnData instance as the second parameter."
        raise TypeError(error_msg)

    # Get the second parameter (expected to be 'adata')
    param = params[1]
    if param.annotation is inspect._empty:
        err_msg = "Namespace initializer's second parameter must be annotated as the 'AnnData' class, got empty annotation."
        raise AttributeError(err_msg)

    name_ok = param.name == "adata"

    # Resolve the annotation using get_type_hints to handle forward references and aliases.
    try:
        type_hints = get_type_hints(ns_class.__init__)
        resolved_type = type_hints.get(param.name, param.annotation)
    except NameError as e:
        err_msg = f"Namespace initializer's second parameter must be named 'adata', got '{param.name}'."
        raise NameError(err_msg) from e

    type_ok = resolved_type is AnnData

    match (name_ok, type_ok):
        case (True, True):
            return  # Signature is correct.
        case (False, True):
            msg = f"Namespace initializer's second parameter must be named 'adata', got {param.name!r}."
            raise TypeError(msg)
        case (True, False):
            type_repr = getattr(resolved_type, "__name__", str(resolved_type))
            msg = f"Namespace initializer's second parameter must be annotated as the 'AnnData' class, got '{type_repr}'."
            raise TypeError(msg)
        case _:
            type_repr = getattr(resolved_type, "__name__", str(resolved_type))
            msg = (
                f"Namespace initializer's second parameter must be named 'adata', got {param.name!r}. "
                f"And must be annotated as 'AnnData', got {type_repr!r}."
            )
            raise TypeError(msg)


def _create_accessor_section_formatter(
    name: str, ns_class: type[ExtensionNamespace]
) -> None:
    """Create and register a SectionFormatter for an accessor with _repr_section_ method.

    This enables unified accessor + visualization registration. When an accessor
    class defines a `_repr_section_` method, a SectionFormatter is automatically
    registered that delegates to the accessor instance.

    Parameters
    ----------
    name
        The accessor name (used as section name)
    ns_class
        The accessor class that has a _repr_section_ method
    """
    from anndata._repr.registry import (
        FormatterContext,
        SectionFormatter,
        register_formatter,
    )

    # Get optional section configuration from class attributes
    after_section = getattr(ns_class, "section_after", None)
    display_name = getattr(ns_class, "section_display_name", name)
    tooltip = getattr(ns_class, "section_tooltip", "")
    doc_url = getattr(ns_class, "section_doc_url", None)

    class AccessorSectionFormatter(SectionFormatter):
        """Auto-generated SectionFormatter that delegates to accessor._repr_section_."""

        @property
        def section_name(self) -> str:
            return name

        @property
        def display_name(self) -> str:
            return display_name

        @property
        def after_section(self) -> str | None:
            return after_section

        @property
        def tooltip(self) -> str:
            return tooltip

        @property
        def doc_url(self) -> str | None:
            return doc_url

        def should_show(self, obj: AnnData) -> bool:
            if not hasattr(obj, name):
                return False
            accessor = getattr(obj, name)
            if not hasattr(accessor, REPR_SECTION_METHOD):
                return False
            # Call _repr_section_ to check if it returns entries
            result = getattr(accessor, REPR_SECTION_METHOD)(FormatterContext())
            return result is not None and len(result) > 0

        def get_entries(
            self, obj: AnnData, context: FormatterContext
        ) -> list[FormattedEntry]:
            accessor = getattr(obj, name)
            result = getattr(accessor, REPR_SECTION_METHOD)(context)
            return result if result is not None else []

    # Give it a meaningful name for debugging
    AccessorSectionFormatter.__name__ = f"{ns_class.__name__}SectionFormatter"
    AccessorSectionFormatter.__qualname__ = f"{ns_class.__name__}SectionFormatter"

    register_formatter(AccessorSectionFormatter())


def _create_namespace[NameSpT: ExtensionNamespace](
    name: str, cls: type[AnnData]
) -> Callable[[type[NameSpT]], type[NameSpT]]:
    """Register custom namespace against the underlying AnnData class."""

    def namespace(ns_class: type[NameSpT]) -> type[NameSpT]:
        _check_namespace_signature(ns_class)  # Perform the runtime signature check
        if name in _reserved_namespaces:
            msg = f"cannot override reserved attribute {name!r}"
            raise AttributeError(msg)
        elif name in cls._accessors:
            warn(
                f"Overriding existing custom namespace {name!r} (on {cls.__name__!r})",
                UserWarning,
            )
        setattr(cls, name, AccessorNameSpace(name, ns_class))
        cls._accessors.add(name)

        # Auto-register SectionFormatter if accessor has _repr_section_ method
        if hasattr(ns_class, REPR_SECTION_METHOD):
            _create_accessor_section_formatter(name, ns_class)

        return ns_class

    return namespace


def register_anndata_namespace[NameSpT: ExtensionNamespace](
    name: str,
) -> Callable[[type[NameSpT]], type[NameSpT]]:
    """Decorator for registering custom functionality with an :class:`~anndata.AnnData` object.

    This decorator allows you to extend AnnData objects with custom methods and properties
    organized under a namespace. The namespace becomes accessible as an attribute on AnnData
    instances, providing a clean way to you to add domain-specific functionality without modifying
    the AnnData class itself, or extending the class with additional methods as you see fit in your workflow.

    Parameters
    ----------
    name
        Name under which the accessor should be registered. This will be the attribute name
        used to access your namespace's functionality on AnnData objects (e.g., `adata.{name}`).
        Cannot conflict with existing AnnData attributes like `obs`, `var`, `X`, etc. The list of reserved
        attributes includes everything outputted by `dir(AnnData)`.

    Returns
    -------
    A decorator that registers the decorated class as a custom namespace.

    Notes
    -----
    Implementation requirements:

    1. The decorated class must have an `__init__`` method that accepts exactly one parameter
       (besides `self`) named `adata` and annotated with type :class:`~anndata.AnnData`.
    2. The namespace will be initialized with the AnnData object on first access and then
       cached on the instance.
    3. If the namespace name conflicts with an existing namespace, a warning is issued.
    4. If the namespace name conflicts with a built-in AnnData attribute, an AttributeError is raised.

    HTML Representation
    ~~~~~~~~~~~~~~~~~~~
    If the accessor class defines a ``_repr_section_`` method, a section will automatically
    be added to the HTML representation. This enables unified accessor + visualization
    registration with a single decorator.

    The ``_repr_section_`` method should have the signature::

        def _repr_section_(self, context: FormatterContext) -> list[FormattedEntry] | None:
            '''Return entries for HTML repr, or None to hide section.'''
            ...

    Optional class attributes for section configuration:

    - ``section_after``: Section name after which this section appears (e.g., "obsm")
    - ``section_display_name``: Display name for the section header (defaults to accessor name)
    - ``section_tooltip``: Tooltip text for the section header
    - ``section_doc_url``: URL to documentation (shown as link icon in header)

    Examples
    --------
    Simple transformation namespace with two methods:

    >>> import anndata as ad
    >>> import numpy as np
    >>>
    >>> @ad.register_anndata_namespace("transform")
    ... class TransformX:
    ...     def __init__(self, adata: ad.AnnData):
    ...         self._adata = adata
    ...
    ...     def log1p(
    ...         self, layer: str = None, inplace: bool = False
    ...     ) -> ad.AnnData | None:
    ...         '''Log1p transform the data.'''
    ...         data = self._adata.layers[layer] if layer else self._adata.X
    ...         log1p_data = np.log1p(data)
    ...
    ...         if layer:
    ...             layer_name = f"{layer}_log1p" if not inplace else layer
    ...         else:
    ...             layer_name = "log1p"
    ...
    ...         self._adata.layers[layer_name] = log1p_data
    ...
    ...         if not inplace:
    ...             return self._adata
    ...
    ...     def arcsinh(
    ...         self, layer: str = None, scale: float = 1.0, inplace: bool = False
    ...     ) -> ad.AnnData | None:
    ...         '''Arcsinh transform the data with optional scaling.'''
    ...         data = self._adata.layers[layer] if layer else self._adata.X
    ...         asinh_data = np.arcsinh(data / scale)
    ...
    ...         if layer:
    ...             layer_name = f"{layer}_arcsinh" if not inplace else layer
    ...         else:
    ...             layer_name = "arcsinh"
    ...
    ...         self._adata.layers[layer_name] = asinh_data
    ...
    ...         if not inplace:
    ...             return self._adata
    >>>
    >>> # Create an AnnData object
    >>> rng = np.random.default_rng(42)
    >>> adata = ad.AnnData(X=rng.poisson(1, size=(100, 2000)))
    >>>
    >>> # Use the registered namespace
    >>> adata.transform.log1p()  # Transforms X and returns the AnnData object
    AnnData object with n_obs × n_vars = 100 × 2000
        layers: 'log1p'
    >>> adata.transform.arcsinh()  # Transforms X and returns the AnnData object
    AnnData object with n_obs × n_vars = 100 × 2000
        layers: 'log1p', 'arcsinh'

    Accessor with HTML section visualization:

    .. code-block:: python

        from anndata.extensions import (
            register_anndata_namespace,
            FormattedEntry,
            FormattedOutput,
        )


        @register_anndata_namespace("spatial")
        class SpatialAccessor:
            # Optional: configure section positioning and display
            section_after = "obsm"
            section_display_name = "spatial"
            section_tooltip = "Spatial data (images, coordinates)"
            section_doc_url = "https://spatialdata.readthedocs.io/"

            def __init__(self, adata: ad.AnnData):
                self._adata = adata

            @property
            def images(self):
                return self._adata.uns.get("spatial_images", {})

            def add_image(self, key, image):
                if "spatial_images" not in self._adata.uns:
                    self._adata.uns["spatial_images"] = {}
                self._adata.uns["spatial_images"][key] = image

            def _repr_section_(self, context) -> list[FormattedEntry] | None:
                '''Return entries for HTML repr, or None to hide section.'''
                if not self.images:
                    return None
                return [
                    FormattedEntry(
                        key=k,
                        output=FormattedOutput(
                            type_name=f"Image {v.shape}",
                            css_class="dtype-array",
                        ),
                    )
                    for k, v in self.images.items()
                ]


        # Usage:
        adata.spatial.add_image("hires", np.zeros((100, 100, 3)))
        adata._repr_html_()  # Shows "spatial" section with "hires" entry
    """
    return _create_namespace(name, AnnData)


# ---------------------------------------------------------------------------
# Section registration
# ---------------------------------------------------------------------------

from .section_registry import SectionProperty, SectionSpec  # noqa: E402


def register_section(
    name: str,
    *,
    alignment: Literal["obs", "var"] | tuple[Literal["obs", "var"], ...] = (),
    io_key: str | None = None,
) -> Callable[[type], type]:
    """Register a new section on :class:`~anndata.AnnData`.

    Decorator that creates a section from a class definition. The class
    can optionally define methods and attributes to customize behavior.

    Parameters
    ----------
    name
        Attribute name on AnnData (e.g., ``"obst"``). Becomes ``adata.obst``.
    alignment
        Axes each dimension is aligned to. A string for single-axis
        alignment, or a tuple for multi-axis. Examples:
        ``"obs"`` for obs-aligned (like obsm),
        ``("obs", "var")`` for both axes (like layers),
        ``("obs", "obs")`` for pairwise (like obsp),
        ``()`` for unaligned.
    io_key
        Key used in h5ad/zarr files. Defaults to *name*.

    Class Attributes (all optional)
    --------------------------------
    value_type : type
        Type check on assignment (e.g., ``nx.DiGraph``).
    section_after : str
        Position in repr (e.g., ``"obsm"``).
    section_tooltip : str
        Hover text in HTML repr.
    section_doc_url : str
        Documentation link in HTML repr.

    Class Methods (all optional, must be static)
    ---------------------------------------------
    validate(key, value)
        Custom validation on assignment. Raise on invalid.
    subset(value, idx)
        Custom subsetting for ``adata[idx]``. Default uses anndata's
        ``_subset`` dispatch (works for arrays, sparse, DataFrames).
    serialize(value)
        Custom serialization for IO. Return a serializable object.
    deserialize(data)
        Custom deserialization for IO.
    repr_entry(key, value, context)
        Custom HTML repr formatting. Return ``FormattedOutput``.

    Examples
    --------
    Simple axis-aligned section (arrays, no custom behavior):

    .. code-block:: python

        @register_section("obst", alignment="obs")
        class ObstSection:
            pass

    Full-featured section (TreeData-like):

    .. code-block:: python

        @register_section("obst", alignment="obs")
        class ObstSection:
            value_type = nx.DiGraph
            section_after = "obsm"
            section_tooltip = "Observation trees"

            @staticmethod
            def validate(key, value):
                if not nx.is_tree(value):
                    raise ValueError(f"{key} must be a tree")

            @staticmethod
            def subset(value, idx):
                return subset_tree(value, idx)

            @staticmethod
            def serialize(value):
                return digraph_to_json(value)

            @staticmethod
            def deserialize(data):
                return json_to_digraph(data)

    Unaligned section (SpatialData-like):

    .. code-block:: python

        @register_section("images", alignment=())
        class ImagesSection:
            value_type = MultiscaleImage
    """

    # Normalize alignment: string → 1-tuple
    if isinstance(alignment, str):
        alignment = (alignment,)

    def decorator(cls: type) -> type:
        if name in AnnData._registered_sections:
            msg = f"Section {name!r} is already registered"
            raise ValueError(msg)
        if name in _reserved_namespaces:
            msg = f"Cannot register section {name!r}: conflicts with existing AnnData attribute"
            raise AttributeError(msg)

        # Extract optional methods and attributes from the class
        spec = SectionSpec(
            name=name,
            alignment=alignment,
            io_key=io_key or name,
            value_type=getattr(cls, "value_type", None),
            validate_fn=getattr(cls, "validate", None),
            subset_fn=getattr(cls, "subset", None),
            serialize_fn=getattr(cls, "serialize", None),
            deserialize_fn=getattr(cls, "deserialize", None),
            repr_entry_fn=getattr(cls, "repr_entry", None),
            section_after=getattr(cls, "section_after", None),
            section_tooltip=getattr(cls, "section_tooltip", ""),
            section_doc_url=getattr(cls, "section_doc_url", None),
        )

        # Create and attach the property descriptor
        prop = SectionProperty(spec)
        setattr(AnnData, name, prop)

        # Register
        AnnData._registered_sections[name] = spec
        _reserved_namespaces.add(name)

        # Auto-register SectionFormatter for HTML repr if repr metadata is present
        if spec.section_after or spec.repr_entry_fn:
            _create_section_repr_formatter(spec)

        return cls

    return decorator


def _create_section_repr_formatter(spec: SectionSpec) -> None:
    """Auto-register a SectionFormatter for a registered section."""
    from anndata._repr.registry import (
        FormattedEntry,
        FormattedOutput,
        SectionFormatter,
        register_formatter,
    )

    class RegisteredSectionFormatter(SectionFormatter):
        @property
        def section_name(self) -> str:
            return spec.name

        @property
        def after_section(self) -> str | None:
            return spec.section_after

        @property
        def tooltip(self) -> str:
            return spec.section_tooltip

        @property
        def doc_url(self) -> str | None:
            return spec.section_doc_url

        def should_show(self, obj: AnnData) -> bool:
            mapping = getattr(obj, spec.name, None)
            return mapping is not None and len(mapping) > 0

        def get_entries(
            self, obj: AnnData, context: FormatterContext
        ) -> list[FormattedEntry]:
            mapping = getattr(obj, spec.name)
            entries = []
            for k in mapping:
                if spec.repr_entry_fn is not None:
                    output = spec.repr_entry_fn(k, mapping[k], context)
                else:
                    output = FormattedOutput(
                        type_name=type(mapping[k]).__name__,
                    )
                entries.append(FormattedEntry(key=k, output=output))
            return entries

    RegisteredSectionFormatter.__name__ = f"{spec.name}SectionFormatter"
    register_formatter(RegisteredSectionFormatter())
