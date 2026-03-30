## `@register_section`: pluggable AnnData sections

This PR lets external packages add new sections to AnnData — with storage, validation, subsetting, IO, and repr — using a single decorator. No subclassing needed.

### Quick example

```python
from anndata.extensions import register_section

@register_section("obst", alignment="obs")
class ObstSection:
    """Observation trees (like obsm, but for tree data)."""
    pass
```

That's it. Now every AnnData object has an `obst` section:

```python
>>> adata = ad.AnnData(
...     X=np.random.rand(4, 3),
...     obs=pd.DataFrame({"cell_type": ["T", "T", "B", "B"]}, index=["c1", "c2", "c3", "c4"]),
...     var=pd.DataFrame(index=["CD8A", "LCK", "MS4A1"]),
...     obst={"lineage": np.random.rand(4, 2)},    # init kwarg works
... )

>>> adata.obst["lineage"].shape
(4, 2)

>>> repr(adata)
AnnData object with n_obs × n_vars = 4 × 3
    obs: 'cell_type'
    obst: 'lineage'

>>> t_cells = adata[adata.obs["cell_type"] == "T"]   # subsetting works
>>> t_cells.obst["lineage"].shape
(2, 2)

>>> adata.write("test.h5ad")                          # IO works
>>> adata2 = ad.read_h5ad("test.h5ad")
>>> adata2.obst["lineage"].shape
(4, 2)
```

### Why

Packages that extend AnnData (TreeData, SpatialData) currently have to subclass `AnnData` or reimplement its internals. TreeData [reimplements the entire write traversal](https://github.com/YosefLab/treedata/blob/main/src/treedata/_core/write.py#L73-L90) with its own hardcoded section list. The same list is hardcoded in at least four places across anndata (`write_h5ad`, `write_anndata`, `read_anndata`, `_gen_repr`). `@register_section` makes all four discoverable.

### Alignment

The `alignment` parameter declares which AnnData axes each dimension of the stored data is aligned to. This controls both validation (shape must match) and subsetting (which dims get sliced when you do `adata[obs_idx, var_idx]`).

| alignment | subsetting behavior | like | use case |
|-----------|-------------------|------|----------|
| `"obs"` | dim 0 follows obs | obsm | per-cell embeddings |
| `"var"` | dim 0 follows var | varm | per-gene annotations |
| `("obs", "var")` | dim 0 = obs, dim 1 = var | layers | alternative matrices |
| `("obs", "obs")` | both dims follow obs | obsp | cell-cell distances |
| `("var", "var")` | both dims follow var | varp | gene-gene correlations |
| `()` | no subsetting | — | images, configs |
| `("obs", "obs", "var")` | 3D tensor | — | cell-cell communication per gene |
| `("obs", "var", "var")` | 3D tensor | — | cell-specific gene regulation |

**3D tensors for cell-cell communication** (CellChat, LIANA, CellPhoneDB):

```python
@register_section("cellcomm", alignment=("obs", "obs", "var"))
class CellCommSection:
    """Ligand-receptor scores: sender_cell × receiver_cell × gene."""
    pass
```

```python
>>> adata.cellcomm["lr_scores"] = np.random.rand(4, 4, 3)  # (n_obs, n_obs, n_vars)
>>> adata.cellcomm["lr_scores"].shape
(4, 4, 3)

>>> t_cells = adata[adata.obs["cell_type"] == "T"]
>>> t_cells.cellcomm["lr_scores"].shape      # both cell dims subset
(2, 2, 3)

>>> sub = adata[:, ["CD8A", "LCK"]]
>>> sub.cellcomm["lr_scores"].shape           # gene dim subsets
(4, 4, 2)
```

**Cell-specific gene regulatory networks** (SCENIC, CellOracle, Dictys):

```python
@register_section("genereg", alignment=("obs", "var", "var"))
class GeneRegSection:
    """Per-cell GRN: cell × source_gene × target_gene."""
    pass
```

```python
>>> adata.genereg["scenic"] = np.random.rand(4, 3, 3)  # (n_obs, n_vars, n_vars)

>>> t_cells = adata[adata.obs["cell_type"] == "T"]
>>> t_cells.genereg["scenic"].shape          # cell dim subsets
(2, 3, 3)

>>> sub = adata[:, ["CD8A", "LCK"]]
>>> sub.genereg["scenic"].shape               # both gene dims subset
(4, 2, 2)
```

### Custom behavior

All methods are optional. Omit any you don't need.

```python
@register_section("obst", alignment="obs")
class ObstSection:
    value_type = nx.DiGraph                      # type enforcement
    section_after = "obsm"                       # position in repr
    section_tooltip = "Observation trees"        # hover text

    @staticmethod
    def validate(key, value):                    # custom validation
        if not nx.is_tree(value):
            raise ValueError(f"{key} must be a tree")

    @staticmethod
    def subset(value, idx):                      # custom subsetting
        return subset_tree(value, idx)

    @staticmethod
    def serialize(value):                        # custom write
        return digraph_to_json(value)

    @staticmethod
    def deserialize(data):                       # custom read
        return json_to_digraph(data)

    @staticmethod
    def repr_entry(key, value, context):         # custom HTML repr
        return FormattedOutput(type_name=f"Tree ({value.number_of_nodes()} nodes)")
```

Validation in action:

```python
>>> adata.obst["bad"] = [[1, 2], [3, 4]]
TypeError: Values in 'obst' must be ndarray, got list

>>> adata.obst["bad"] = np.ones(3)              # custom validate
ValueError: bad must be 2D, got 1D

>>> adata.obst["bad"] = np.ones((10, 2))        # alignment check
ValueError: Value for obst['bad'] has shape[0]=10, expected 4 (n_obs)
```

### xarray DataArray example

Custom types that anndata can't natively serialize work end-to-end via `serialize`/`deserialize`:

```python
import xarray as xr

@register_section("xr_layers", alignment=("obs", "var"))
class XarrayLayers:
    value_type = xr.DataArray

    @staticmethod
    def serialize(value):
        return value.values          # xarray → numpy for h5ad

    @staticmethod
    def deserialize(data):
        return xr.DataArray(data)    # numpy → xarray on read
```

```python
>>> adata.xr_layers["scaled"] = xr.DataArray(np.random.rand(4, 3), dims=["obs", "var"])
>>> adata.write("test.h5ad")
>>> adata2 = ad.read_h5ad("test.h5ad")
>>> isinstance(adata2.xr_layers["scaled"], xr.DataArray)
True
```

### What you get for free

| Feature | Works automatically |
|---------|-------------------|
| `adata.obst["x"] = array` | Property accessor + validation |
| `adata[:10].obst` | Subsetting via declared alignment |
| `adata.copy()` | Deep copy of registered sections |
| `adata.write("f.h5ad")` | IO via serialize (or standard write_elem) |
| `ad.read_h5ad("f.h5ad")` | IO via deserialize (or standard read_elem) |
| `AnnData(obst={...})` | Init kwargs |
| `repr(adata)` | Shows when non-empty |
| View copy-on-write | Writing to a view triggers copy |

### Also in this PR

- **`@register_anndata_namespace`** — custom accessor APIs (`adata.spatial.images`)
- **`@register_formatter`** — custom HTML type/section formatters
- **`anndata.extensions`** module consolidating all extension APIs

### Test coverage

67 tests covering all alignment patterns, custom validation, custom IO (JSON, xarray), 3D tensor subsetting, copy-on-write, and end-to-end workflows for TreeData-like, SpatialData-like, CellChat-like, and SCENIC-like scenarios.

### Future direction

The `alignment` tuple naturally extends to custom axes beyond obs/var. A future `register_axis` could let packages define new named dimensions with their own indices, enabling N-dimensional indexing like `adata[obs_idx, var_idx, spatial_idx]`. This is the conceptual step from DataFrame (2D) to xarray Dataset (N-D) — with `@register_section` as the foundation.
