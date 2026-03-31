"""Tests for register_section decorator.

Validates that registered sections behave correctly for all alignment
combinations, custom validation, custom subsetting, custom IO, and
HTML repr integration. Uses TreeData-like, SpatialData-like, and
xarray scenarios.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

import anndata as ad
from anndata.extensions import register_section

# ---------------------------------------------------------------------------
# Fixtures: register sections once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _register_test_sections():  # noqa: PLR0912
    """Register test sections for all alignment combinations."""
    # obs-aligned (like obsm)
    if "sec_obs" not in ad.AnnData._registered_sections:

        @register_section("sec_obs", alignment="obs")
        class SecObs:
            pass

    # var-aligned (like varm)
    if "sec_var" not in ad.AnnData._registered_sections:

        @register_section("sec_var", alignment="var")
        class SecVar:
            pass

    # Both axes (like layers)
    if "sec_both" not in ad.AnnData._registered_sections:

        @register_section("sec_both", alignment=("obs", "var"))
        class SecBoth:
            pass

    # Pairwise obs (like obsp)
    if "sec_pair_obs" not in ad.AnnData._registered_sections:

        @register_section("sec_pair_obs", alignment=("obs", "obs"))
        class SecPairObs:
            pass

    # Pairwise var (like varp)
    if "sec_pair_var" not in ad.AnnData._registered_sections:

        @register_section("sec_pair_var", alignment=("var", "var"))
        class SecPairVar:
            pass

    # Unaligned (like SpatialData images)
    if "sec_unaligned" not in ad.AnnData._registered_sections:

        @register_section("sec_unaligned", alignment=())
        class SecUnaligned:
            pass

    # Custom type validation (TreeData-like)
    if "sec_typed" not in ad.AnnData._registered_sections:

        @register_section("sec_typed", alignment="obs")
        class SecTyped:
            value_type = np.ndarray

            @staticmethod
            def validate(key, value):
                if value.ndim != 2:
                    msg = f"{key} must be 2D"
                    raise ValueError(msg)

    # Custom serialize/deserialize
    if "sec_custom_io" not in ad.AnnData._registered_sections:

        @register_section("sec_custom_io", alignment="obs")
        class SecCustomIO:
            @staticmethod
            def serialize(value):
                # Convert dict to JSON string for storage
                return json.dumps(value)

            @staticmethod
            def deserialize(data):
                return json.loads(data)

    # Cell-cell communication tensor: (sender, receiver, gene)
    if "cellcomm" not in ad.AnnData._registered_sections:

        @register_section("cellcomm", alignment=("obs", "obs", "var"))
        class CellCommSection:
            """Ligand-receptor communication scores (sender × receiver × gene)."""

            section_after = "obsp"
            section_tooltip = "Cell-cell communication"

    # Cell-specific gene-gene interactions: (obs, var, var)
    if "genereg" not in ad.AnnData._registered_sections:

        @register_section("genereg", alignment=("obs", "var", "var"))
        class GeneRegSection:
            """Cell-specific gene regulatory networks (cell × gene × gene)."""

            section_after = "varp"
            section_tooltip = "Gene regulation per cell"

    # Custom subset
    if "sec_custom_subset" not in ad.AnnData._registered_sections:

        @register_section("sec_custom_subset", alignment="obs")
        class SecCustomSubset:
            @staticmethod
            def subset(value, idx):
                # Custom: return a dict describing the subset
                return {"original": value, "subset_idx": idx}

    # Factored tensor: store rank-R factors, reconstruct on demand
    if "comm_obs" not in ad.AnnData._registered_sections:

        @register_section("comm_obs", alignment="obs")
        class CommObs:
            """Cell factor matrix (n_obs × rank) for communication tensor."""

    if "comm_var" not in ad.AnnData._registered_sections:

        @register_section("comm_var", alignment="var")
        class CommVar:
            """Gene factor matrix (n_vars × rank) for communication tensor."""

    # xarray layers (custom type with serialize/deserialize)
    if "xr_layers" not in ad.AnnData._registered_sections:

        @register_section("xr_layers", alignment=("obs", "var"))
        class XarrayLayers:
            value_type = xr.DataArray

            @staticmethod
            def serialize(value):
                return value.values  # xarray → numpy for h5ad

            @staticmethod
            def deserialize(data):
                return xr.DataArray(data)  # numpy → xarray on read


@pytest.fixture
def adata():
    """Basic AnnData for testing."""
    return ad.AnnData(
        X=np.ones((5, 3)),
        obs=pd.DataFrame({"group": list("aabbc")}, index=[f"c{i}" for i in range(5)]),
        var=pd.DataFrame(index=[f"v{i}" for i in range(3)]),
    )


# ---------------------------------------------------------------------------
# Registration API
# ---------------------------------------------------------------------------


class TestRegistrationAPI:
    def test_register_creates_property(self):
        assert hasattr(ad.AnnData, "sec_obs")
        adata = ad.AnnData(np.ones((3, 4)))
        assert len(adata.sec_obs) == 0

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_section("sec_obs", alignment="obs")(type("Dup", (), {}))

    def test_register_reserved_name_raises(self):
        # "obs" is a built-in registered section, so it's already registered
        with pytest.raises(ValueError, match="already registered"):
            register_section("obs", alignment="obs")(type("Bad", (), {}))

    def test_all_sections_in_registry(self):
        for name in [
            "sec_obs",
            "sec_var",
            "sec_both",
            "sec_pair_obs",
            "sec_pair_var",
            "sec_unaligned",
            "sec_typed",
        ]:
            assert name in ad.AnnData._registered_sections


# ---------------------------------------------------------------------------
# Obs-aligned: alignment=("obs",)
# ---------------------------------------------------------------------------


class TestObsAligned:
    def test_store_and_retrieve(self, adata):
        arr = np.random.rand(5, 3)
        adata.sec_obs["x"] = arr
        np.testing.assert_array_equal(adata.sec_obs["x"], arr)

    def test_wrong_shape_raises(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.sec_obs["bad"] = np.ones((10, 3))

    def test_sparse(self, adata):
        adata.sec_obs["sp"] = csr_matrix(np.eye(5))
        assert adata.sec_obs["sp"].shape == (5, 5)

    def test_subset_obs(self, adata):
        adata.sec_obs["x"] = np.arange(15).reshape(5, 3)
        sub = adata[:3]
        assert sub.sec_obs["x"].shape == (3, 3)

    def test_subset_var_unchanged(self, adata):
        adata.sec_obs["x"] = np.arange(15).reshape(5, 3)
        sub = adata[:, :2]
        # obs-aligned section not affected by var subsetting
        assert sub.sec_obs["x"].shape == (5, 3)


# ---------------------------------------------------------------------------
# Var-aligned: alignment=("var",)
# ---------------------------------------------------------------------------


class TestVarAligned:
    def test_store_and_retrieve(self, adata):
        arr = np.random.rand(3, 2)
        adata.sec_var["x"] = arr
        np.testing.assert_array_equal(adata.sec_var["x"], arr)

    def test_wrong_shape_raises(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.sec_var["bad"] = np.ones((10, 2))

    def test_subset_var(self, adata):
        adata.sec_var["x"] = np.arange(6).reshape(3, 2)
        sub = adata[:, :2]
        assert sub.sec_var["x"].shape == (2, 2)

    def test_subset_obs_unchanged(self, adata):
        adata.sec_var["x"] = np.arange(6).reshape(3, 2)
        sub = adata[:3]
        assert sub.sec_var["x"].shape == (3, 2)


# ---------------------------------------------------------------------------
# Both axes: alignment=("obs", "var")
# ---------------------------------------------------------------------------


class TestBothAxes:
    def test_store_and_retrieve(self, adata):
        arr = np.random.rand(5, 3)
        adata.sec_both["x"] = arr
        np.testing.assert_array_equal(adata.sec_both["x"], arr)

    def test_wrong_obs_shape_raises(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.sec_both["bad"] = np.ones((10, 3))

    def test_wrong_var_shape_raises(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.sec_both["bad"] = np.ones((5, 10))

    def test_subset_both(self, adata):
        adata.sec_both["x"] = np.arange(15).reshape(5, 3)
        sub = adata[:3, :2]
        assert sub.sec_both["x"].shape == (3, 2)


# ---------------------------------------------------------------------------
# Pairwise obs: alignment=("obs", "obs")
# ---------------------------------------------------------------------------


class TestPairwiseObs:
    def test_store_and_retrieve(self, adata):
        arr = np.random.rand(5, 5)
        adata.sec_pair_obs["dist"] = arr
        np.testing.assert_array_equal(adata.sec_pair_obs["dist"], arr)

    def test_non_square_raises(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.sec_pair_obs["bad"] = np.ones((5, 3))

    def test_subset(self, adata):
        adata.sec_pair_obs["dist"] = np.eye(5)
        sub = adata[:3]
        assert sub.sec_pair_obs["dist"].shape == (3, 3)


# ---------------------------------------------------------------------------
# Pairwise var: alignment=("var", "var")
# ---------------------------------------------------------------------------


class TestPairwiseVar:
    def test_store_and_retrieve(self, adata):
        arr = np.random.rand(3, 3)
        adata.sec_pair_var["corr"] = arr
        np.testing.assert_array_equal(adata.sec_pair_var["corr"], arr)

    def test_subset(self, adata):
        adata.sec_pair_var["corr"] = np.eye(3)
        sub = adata[:, :2]
        assert sub.sec_pair_var["corr"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Unaligned: alignment=()
# ---------------------------------------------------------------------------


class TestUnaligned:
    def test_store_anything(self, adata):
        adata.sec_unaligned["img"] = np.random.rand(100, 100, 3)
        assert adata.sec_unaligned["img"].shape == (100, 100, 3)

    def test_no_shape_validation(self, adata):
        # Any shape is fine for unaligned
        adata.sec_unaligned["a"] = np.ones((1,))
        adata.sec_unaligned["b"] = np.ones((999, 888))
        assert len(adata.sec_unaligned) == 2

    def test_subset_unchanged(self, adata):
        adata.sec_unaligned["img"] = np.random.rand(100, 100, 3)
        sub = adata[:3]
        # Unaligned data is not subsetted
        assert sub.sec_unaligned["img"].shape == (100, 100, 3)

    def test_non_array_values(self, adata):
        adata.sec_unaligned["config"] = {"key": "value"}
        assert adata.sec_unaligned["config"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Custom type validation
# ---------------------------------------------------------------------------


class TestCustomValidation:
    def test_type_check(self, adata):
        adata.sec_typed["x"] = np.eye(5)
        assert adata.sec_typed["x"].shape == (5, 5)

    def test_wrong_type_raises(self, adata):
        with pytest.raises(TypeError, match="must be ndarray"):
            adata.sec_typed["bad"] = [[1, 2], [3, 4]]

    def test_custom_validate(self, adata):
        with pytest.raises(ValueError, match="must be 2D"):
            adata.sec_typed["bad"] = np.ones(5)  # 1D, not 2D


# ---------------------------------------------------------------------------
# Custom subset
# ---------------------------------------------------------------------------


class TestCustomSubset:
    def test_custom_subset_fn(self, adata):
        adata.sec_custom_subset["x"] = np.eye(5)
        sub = adata[:3]
        result = sub.sec_custom_subset["x"]
        assert isinstance(result, dict)
        assert "original" in result
        assert "subset_idx" in result


# ---------------------------------------------------------------------------
# Init kwargs
# ---------------------------------------------------------------------------


class TestInitKwargs:
    def test_init_with_section(self):
        adata = ad.AnnData(
            np.ones((3, 4)),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            sec_obs={"x": np.eye(3)},
        )
        assert "x" in adata.sec_obs

    def test_init_with_multiple_sections(self):
        adata = ad.AnnData(
            np.ones((3, 4)),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            var=pd.DataFrame(index=["v1", "v2", "v3", "v4"]),
            sec_obs={"x": np.eye(3)},
            sec_var={"y": np.eye(4)},
        )
        assert "x" in adata.sec_obs
        assert "y" in adata.sec_var

    def test_init_without_section(self):
        adata = ad.AnnData(np.ones((3, 4)))
        assert len(adata.sec_obs) == 0


# ---------------------------------------------------------------------------
# IO roundtrip (h5ad)
# ---------------------------------------------------------------------------


class TestH5adRoundtrip:
    def test_write_read_obs_aligned(self, adata, tmp_path):
        adata.sec_obs["x"] = np.eye(5)
        path = tmp_path / "test.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert "x" in adata2.sec_obs
        np.testing.assert_array_equal(adata2.sec_obs["x"], np.eye(5))

    def test_write_read_both_axes(self, adata, tmp_path):
        adata.sec_both["x"] = np.arange(15).reshape(5, 3).astype(float)
        path = tmp_path / "test.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        np.testing.assert_array_equal(adata2.sec_both["x"], np.arange(15).reshape(5, 3))

    def test_empty_section_not_written(self, adata, tmp_path):
        import h5py

        path = tmp_path / "test.h5ad"
        adata.write(path)
        with h5py.File(path, "r") as f:
            assert "sec_obs" not in f

    def test_custom_serialize_deserialize(self, adata, tmp_path):
        adata.sec_custom_io["config"] = {"lr": 0.001, "epochs": 100}
        path = tmp_path / "test.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert adata2.sec_custom_io["config"] == {"lr": 0.001, "epochs": 100}

    def test_subset_then_write(self, adata, tmp_path):
        adata.sec_obs["x"] = np.arange(15).reshape(5, 3).astype(float)
        sub = adata[:3].copy()
        path = tmp_path / "test.h5ad"
        sub.write(path)
        sub2 = ad.read_h5ad(path)
        assert sub2.sec_obs["x"].shape == (3, 3)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_shows_section(self, adata):
        adata.sec_obs["x"] = np.eye(5)
        assert "sec_obs" in repr(adata)

    def test_repr_hides_empty(self, adata):
        assert "sec_obs" not in repr(adata)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_preserves(self, adata):
        adata.sec_obs["x"] = np.eye(5)
        adata2 = adata.copy()
        assert "x" in adata2.sec_obs
        np.testing.assert_array_equal(adata2.sec_obs["x"], np.eye(5))

    def test_copy_is_independent(self, adata):
        adata.sec_obs["x"] = np.eye(5)
        adata2 = adata.copy()
        adata2.sec_obs["new"] = np.ones((5, 2))
        assert "new" not in adata.sec_obs

    def test_view_copy_on_write(self, adata):
        adata.sec_obs["x"] = np.eye(5)
        sub = adata[:3]
        sub.sec_obs["new"] = np.ones((3, 2))
        assert not sub.is_view
        assert "new" in sub.sec_obs
        assert "new" not in adata.sec_obs


# ---------------------------------------------------------------------------
# TreeData-like end-to-end scenario
# ---------------------------------------------------------------------------


class TestTreeDataScenario:
    def test_full_workflow(self, tmp_path):
        n_obs, n_vars = 10, 5
        adata = ad.AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame(
                {"cell_type": pd.Categorical(["A"] * 5 + ["B"] * 5)},
                index=[f"cell_{i}" for i in range(n_obs)],
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
        )

        # Store tree embeddings
        adata.sec_obs["lineage"] = np.random.rand(n_obs, 4)
        adata.sec_var["phylogeny"] = np.random.rand(n_vars, 3)

        # Subset
        mask = adata.obs["cell_type"] == "A"
        sub = adata[mask]
        assert sub.sec_obs["lineage"].shape == (5, 4)
        assert sub.sec_var["phylogeny"].shape == (n_vars, 3)

        # IO roundtrip
        path = tmp_path / "treedata.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert set(adata2.sec_obs.keys()) == {"lineage"}
        assert set(adata2.sec_var.keys()) == {"phylogeny"}

        # Repr
        assert "sec_obs" in repr(adata2)
        assert "sec_var" in repr(adata2)


# ---------------------------------------------------------------------------
# SpatialData-like end-to-end scenario
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# xarray DataArray layers (custom type + serialize/deserialize)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cell-cell communication tensor: alignment=("obs", "obs", "var")
# ---------------------------------------------------------------------------


class TestCellCommunication:
    """3D tensor for ligand-receptor communication scores.

    Tools like CellChat, LIANA, and CellPhoneDB compute communication
    strengths between cell pairs mediated by specific genes. The natural
    shape is (sender_cell, receiver_cell, gene). With alignment=("obs",
    "obs", "var"), the tensor subsets correctly when filtering cells or genes.
    """

    def test_store_tensor(self, adata):
        comm = np.random.rand(5, 5, 3)
        adata.cellcomm["lr_scores"] = comm
        assert adata.cellcomm["lr_scores"].shape == (5, 5, 3)

    def test_validates_obs_dim(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.cellcomm["bad"] = np.ones((10, 10, 3))

    def test_validates_var_dim(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.cellcomm["bad"] = np.ones((5, 5, 10))

    def test_validates_square_obs(self, adata):
        """Sender and receiver must both be n_obs."""
        with pytest.raises(ValueError, match="shape"):
            adata.cellcomm["bad"] = np.ones((5, 3, 3))

    def test_subset_cells(self, adata):
        """Filtering cells subsets both sender and receiver dims."""
        adata.cellcomm["lr"] = np.random.rand(5, 5, 3)
        sub = adata[:3]
        assert sub.cellcomm["lr"].shape == (3, 3, 3)

    def test_subset_genes(self, adata):
        """Filtering genes subsets the third dim."""
        adata.cellcomm["lr"] = np.random.rand(5, 5, 3)
        sub = adata[:, :2]
        assert sub.cellcomm["lr"].shape == (5, 5, 2)

    def test_subset_both(self, adata):
        adata.cellcomm["lr"] = np.random.rand(5, 5, 3)
        sub = adata[:3, :2]
        assert sub.cellcomm["lr"].shape == (3, 3, 2)

    def test_io_roundtrip(self, adata, tmp_path):
        comm = np.random.rand(5, 5, 3)
        adata.cellcomm["lr_scores"] = comm
        path = tmp_path / "comm.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        np.testing.assert_array_almost_equal(adata2.cellcomm["lr_scores"], comm)

    def test_workflow(self, tmp_path):
        """End-to-end: simulate CellChat-like analysis."""
        n_obs, n_vars = 20, 50
        adata = ad.AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame(
                {"cell_type": pd.Categorical(["T"] * 10 + ["B"] * 10)},
                index=[f"cell_{i}" for i in range(n_obs)],
            ),
            var=pd.DataFrame(
                {"is_ligand": [True] * 25 + [False] * 25},
                index=[f"gene_{i}" for i in range(n_vars)],
            ),
        )

        # Compute communication scores (simulated)
        adata.cellcomm["cellchat"] = np.random.rand(n_obs, n_obs, n_vars)

        # Filter to T cells only
        t_cells = adata.obs["cell_type"] == "T"
        sub = adata[t_cells]
        assert sub.cellcomm["cellchat"].shape == (10, 10, n_vars)

        # Filter to ligand genes only
        ligands = adata.var["is_ligand"]
        sub2 = adata[:, ligands]
        assert sub2.cellcomm["cellchat"].shape == (n_obs, n_obs, 25)

        # IO roundtrip
        path = tmp_path / "cellchat.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert adata2.cellcomm["cellchat"].shape == (n_obs, n_obs, n_vars)


# ---------------------------------------------------------------------------
# xarray DataArray layers (custom type + serialize/deserialize)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cell-specific gene regulation: alignment=("obs", "var", "var")
# ---------------------------------------------------------------------------


class TestGeneRegulation:
    """3D tensor for cell-specific gene regulatory networks.

    Each cell has its own gene-gene interaction matrix (e.g., inferred
    from single-cell GRN methods like SCENIC, CellOracle, or Dictys).
    Shape is (cell, source_gene, target_gene). Subsetting cells reduces
    the first dim, subsetting genes reduces both gene dims.
    """

    def test_store_tensor(self, adata):
        grn = np.random.rand(5, 3, 3)
        adata.genereg["scenic"] = grn
        assert adata.genereg["scenic"].shape == (5, 3, 3)

    def test_validates_obs_dim(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.genereg["bad"] = np.ones((10, 3, 3))

    def test_validates_var_dims(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.genereg["bad"] = np.ones((5, 10, 3))  # source wrong
        with pytest.raises(ValueError, match="shape"):
            adata.genereg["bad"] = np.ones((5, 3, 10))  # target wrong

    def test_subset_cells(self, adata):
        """Filtering cells subsets the first dim only."""
        adata.genereg["grn"] = np.random.rand(5, 3, 3)
        sub = adata[:3]
        assert sub.genereg["grn"].shape == (3, 3, 3)

    def test_subset_genes(self, adata):
        """Filtering genes subsets both gene dims (source and target)."""
        adata.genereg["grn"] = np.random.rand(5, 3, 3)
        sub = adata[:, :2]
        assert sub.genereg["grn"].shape == (5, 2, 2)

    def test_subset_both(self, adata):
        adata.genereg["grn"] = np.random.rand(5, 3, 3)
        sub = adata[:3, :2]
        assert sub.genereg["grn"].shape == (3, 2, 2)

    def test_io_roundtrip(self, adata, tmp_path):
        grn = np.random.rand(5, 3, 3)
        adata.genereg["scenic"] = grn
        path = tmp_path / "grn.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        np.testing.assert_array_almost_equal(adata2.genereg["scenic"], grn)


# ---------------------------------------------------------------------------
# Factored tensor: sections + accessor (scalable communication analysis)
# ---------------------------------------------------------------------------


class TestFactoredTensor:
    """Store rank-R factors in sections, reconstruct tensor via accessor.

    For million-cell datasets, a dense (n_obs × n_obs × n_vars) tensor
    is infeasible. Instead, store compact factors (n_obs × rank) and
    (n_vars × rank), and reconstruct on demand. The factors subset
    correctly, serialize to h5ad, and the accessor provides the tensor API.
    """

    @pytest.fixture(autouse=True)
    def _ensure_accessor(self):
        """Create and register the accessor (idempotent)."""
        if hasattr(ad.AnnData, "comm"):
            return
        from anndata.extensions import (
            FormattedEntry,
            FormattedOutput,
            register_anndata_namespace,
        )

        @register_anndata_namespace("comm")
        class CellCommAccessor:
            section_after = "obsp"
            section_tooltip = "Cell-cell communication (factored)"

            def __init__(self, adata: ad.AnnData):
                self._adata = adata

            def tensor(self, key="default"):
                """Reconstruct (obs × obs × var) tensor from factors."""
                U = self._adata.comm_obs[key]
                V = self._adata.comm_var[key]
                return np.einsum("ir,jr,kr->ijk", U, U, V)

            def query(self, sender, receiver, gene, key="default"):
                """O(rank) point query without materializing tensor."""
                U = self._adata.comm_obs[key]
                V = self._adata.comm_var[key]
                i = self._adata.obs_names.get_loc(sender)
                j = self._adata.obs_names.get_loc(receiver)
                k = self._adata.var_names.get_loc(gene)
                return float(U[i] @ (U[j] * V[k]))

            def _repr_section_(self, context):
                keys = list(self._adata.comm_obs.keys())
                if not keys:
                    return None
                return [
                    FormattedEntry(
                        key=k,
                        output=FormattedOutput(
                            type_name=f"rank-{self._adata.comm_obs[k].shape[1]} factors",
                            preview=(
                                f"({self._adata.comm_obs[k].shape[0]} cells "
                                f"× {self._adata.comm_var[k].shape[0]} genes)"
                            ),
                        ),
                    )
                    for k in keys
                ]

    def test_store_factors(self, adata):
        n_obs, n_vars, rank = 5, 3, 2
        adata.comm_obs["lr"] = np.random.rand(n_obs, rank)
        adata.comm_var["lr"] = np.random.rand(n_vars, rank)
        assert adata.comm_obs["lr"].shape == (n_obs, rank)
        assert adata.comm_var["lr"].shape == (n_vars, rank)

    def test_reconstruct_tensor(self, adata):
        rank = 3
        adata.comm_obs["lr"] = np.random.rand(5, rank)
        adata.comm_var["lr"] = np.random.rand(3, rank)
        tensor = adata.comm.tensor("lr")
        assert tensor.shape == (5, 5, 3)

    def test_point_query(self, adata):
        rank = 3
        U = np.random.rand(5, rank)
        V = np.random.rand(3, rank)
        adata.comm_obs["lr"] = U
        adata.comm_var["lr"] = V
        score = adata.comm.query("c0", "c1", "v0", "lr")
        expected = float(U[0] @ (U[1] * V[0]))
        assert abs(score - expected) < 1e-10

    def test_subset_preserves_reconstruction(self, adata):
        rank = 3
        adata.comm_obs["lr"] = np.random.rand(5, rank)
        adata.comm_var["lr"] = np.random.rand(3, rank)
        full_tensor = adata.comm.tensor("lr")

        sub = adata[:3, :2]
        sub_tensor = sub.comm.tensor("lr")
        assert sub_tensor.shape == (3, 3, 2)
        np.testing.assert_array_almost_equal(sub_tensor, full_tensor[:3, :3, :2])

    def test_io_roundtrip(self, adata, tmp_path):
        rank = 3
        adata.comm_obs["lr"] = np.random.rand(5, rank)
        adata.comm_var["lr"] = np.random.rand(3, rank)
        tensor_before = adata.comm.tensor("lr")

        path = tmp_path / "factored.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        tensor_after = adata2.comm.tensor("lr")
        np.testing.assert_array_almost_equal(tensor_before, tensor_after)

    def test_compression_ratio(self):
        """Factors are orders of magnitude smaller than dense tensor."""
        n_obs, n_vars, rank = 1000, 500, 10
        factor_bytes = (n_obs * rank + n_vars * rank) * 8  # float64
        tensor_bytes = n_obs * n_obs * n_vars * 8
        ratio = tensor_bytes / factor_bytes
        assert ratio > 100  # ~33,000× for these sizes


# ---------------------------------------------------------------------------
# xarray DataArray layers (custom type + serialize/deserialize)
# ---------------------------------------------------------------------------


class TestXarrayScenario:
    """xarray DataArrays as layer values with custom serialization."""

    def test_store_xarray(self, adata):
        da = xr.DataArray(np.random.rand(5, 3), dims=["obs", "var"])
        adata.xr_layers["normalized"] = da
        assert isinstance(adata.xr_layers["normalized"], xr.DataArray)
        assert adata.xr_layers["normalized"].shape == (5, 3)

    def test_type_enforcement(self, adata):
        with pytest.raises(TypeError, match="must be DataArray"):
            adata.xr_layers["bad"] = np.ones((5, 3))

    def test_alignment_validation(self, adata):
        with pytest.raises(ValueError, match="shape"):
            adata.xr_layers["bad"] = xr.DataArray(np.ones((10, 3)))

    def test_subset(self, adata):
        da = xr.DataArray(np.arange(15.0).reshape(5, 3), dims=["obs", "var"])
        adata.xr_layers["data"] = da
        sub = adata[:3, :2]
        result = sub.xr_layers["data"]
        assert result.shape == (3, 2)

    def test_io_roundtrip(self, adata, tmp_path):
        da = xr.DataArray(np.arange(15.0).reshape(5, 3), dims=["obs", "var"])
        adata.xr_layers["data"] = da
        path = tmp_path / "xr.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        # Deserialized back to xarray
        assert isinstance(adata2.xr_layers["data"], xr.DataArray)
        np.testing.assert_array_equal(
            adata2.xr_layers["data"].values, np.arange(15.0).reshape(5, 3)
        )

    def test_full_workflow(self, tmp_path):
        """End-to-end: store, subset, copy, IO with xarray layers."""
        adata = ad.AnnData(
            X=np.ones((10, 5)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
            xr_layers={
                "scaled": xr.DataArray(np.random.rand(10, 5), dims=["obs", "var"]),
            },
        )

        # Subset preserves type
        sub = adata[:5]
        assert isinstance(sub.xr_layers["scaled"], xr.DataArray)
        assert sub.xr_layers["scaled"].shape == (5, 5)

        # Copy preserves type
        copy = adata.copy()
        assert isinstance(copy.xr_layers["scaled"], xr.DataArray)

        # IO roundtrip preserves type
        path = tmp_path / "xr_workflow.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert isinstance(adata2.xr_layers["scaled"], xr.DataArray)
        assert adata2.xr_layers["scaled"].shape == (10, 5)

        # Repr shows section
        assert "xr_layers" in repr(adata)


# ---------------------------------------------------------------------------
# SpatialData-like end-to-end scenario
# ---------------------------------------------------------------------------


class TestSpatialDataScenario:
    def test_unaligned_images(self, adata, tmp_path):
        # Store images of arbitrary size
        adata.sec_unaligned["hires"] = np.random.rand(200, 200, 3)
        adata.sec_unaligned["lowres"] = np.random.rand(50, 50, 3)

        # Subsetting obs doesn't affect images
        sub = adata[:3]
        assert sub.sec_unaligned["hires"].shape == (200, 200, 3)

        # IO roundtrip
        path = tmp_path / "spatial.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)
        assert set(adata2.sec_unaligned.keys()) == {"hires", "lowres"}
        assert adata2.sec_unaligned["hires"].shape == (200, 200, 3)
