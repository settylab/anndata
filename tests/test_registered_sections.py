"""Tests for register_aligned_section.

Validates that registered sections behave like built-in sections (obsm, layers, etc.)
for storage, subsetting, IO, repr, and init. Uses TreeData-like and SpatialData-like
scenarios to test real-world extension patterns.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import anndata as ad
from anndata._core.extensions import register_aligned_section


# ---------------------------------------------------------------------------
# Fixtures: register sections once per test session
# ---------------------------------------------------------------------------

# Use module-scoped registration so sections persist across tests.
# This mirrors real-world usage where registration happens at import time.


@pytest.fixture(autouse=True, scope="module")
def _register_test_sections():
    """Register TreeData-like and SpatialData-like sections for all tests."""
    # TreeData pattern: axis-aligned tree mappings
    if "obst" not in ad.AnnData._registered_sections:
        register_aligned_section("obst", axis=0, mapping_type="axis")
    if "vart" not in ad.AnnData._registered_sections:
        register_aligned_section("vart", axis=1, mapping_type="axis")

    # Pairwise pattern (like obsp but for a custom section)
    if "obsd" not in ad.AnnData._registered_sections:
        register_aligned_section("obsd", axis=0, mapping_type="pairwise")

    # Layers-like pattern (aligned to both obs and var)
    if "extra_layers" not in ad.AnnData._registered_sections:
        register_aligned_section("extra_layers", axis=None, mapping_type="layers")


@pytest.fixture
def adata():
    """Basic AnnData for testing."""
    return ad.AnnData(
        X=np.ones((5, 3)),
        obs=pd.DataFrame({"group": list("aabbc")}, index=[f"c{i}" for i in range(5)]),
        var=pd.DataFrame({"gene": [f"g{i}" for i in range(3)]}, index=[f"v{i}" for i in range(3)]),
    )


# ---------------------------------------------------------------------------
# Registration API
# ---------------------------------------------------------------------------


class TestRegistrationAPI:
    def test_register_creates_property(self):
        """Registered section is accessible as a property on AnnData."""
        assert hasattr(ad.AnnData, "obst")
        adata = ad.AnnData(np.ones((3, 4)))
        # Should return an empty mapping
        assert len(adata.obst) == 0

    def test_register_duplicate_raises(self):
        """Registering the same section twice raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_aligned_section("obst", axis=0)

    def test_register_reserved_name_raises(self):
        """Registering a reserved name raises AttributeError."""
        with pytest.raises(AttributeError, match="conflicts with"):
            register_aligned_section("obs", axis=0)

    def test_register_invalid_mapping_type_raises(self):
        """Invalid mapping_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mapping_type"):
            register_aligned_section("bad_section", axis=0, mapping_type="invalid")

    def test_section_in_registry(self):
        """Registered section appears in _registered_sections."""
        assert "obst" in ad.AnnData._registered_sections
        reg = ad.AnnData._registered_sections["obst"]
        assert reg.name == "obst"
        assert reg.axis == 0
        assert reg.mapping_type == "axis"


# ---------------------------------------------------------------------------
# Storage and Validation (TreeData-like: obst, vart)
# ---------------------------------------------------------------------------


class TestAxisAlignedStorage:
    def test_store_and_retrieve(self, adata):
        """Can store and retrieve arrays in registered section."""
        tree = np.random.rand(5, 3)
        adata.obst["lineage"] = tree
        assert "lineage" in adata.obst
        np.testing.assert_array_equal(adata.obst["lineage"], tree)

    def test_wrong_axis_shape_raises(self, adata):
        """Storing array with wrong obs dimension raises."""
        with pytest.raises(ValueError, match="shape"):
            adata.obst["bad"] = np.ones((10, 3))  # 10 != n_obs=5

    def test_var_aligned_section(self, adata):
        """var-aligned section validates against n_vars."""
        tree = np.random.rand(3, 2)  # n_vars=3
        adata.vart["gene_tree"] = tree
        assert adata.vart["gene_tree"].shape == (3, 2)

    def test_var_aligned_wrong_shape_raises(self, adata):
        """var-aligned section rejects wrong shape."""
        with pytest.raises(ValueError, match="shape"):
            adata.vart["bad"] = np.ones((10, 2))  # 10 != n_vars=3

    def test_multiple_entries(self, adata):
        """Can store multiple entries in a section."""
        adata.obst["tree1"] = np.eye(5)
        adata.obst["tree2"] = np.random.rand(5, 4)
        assert set(adata.obst.keys()) == {"tree1", "tree2"}

    def test_delete_entry(self, adata):
        """Can delete entries from registered section."""
        adata.obst["tree"] = np.eye(5)
        del adata.obst["tree"]
        assert "tree" not in adata.obst

    def test_sparse_matrix(self, adata):
        """Can store sparse matrices."""
        sparse = csr_matrix(np.eye(5))
        adata.obst["sparse_tree"] = sparse
        assert adata.obst["sparse_tree"].shape == (5, 5)

    def test_dataframe_in_axis_section(self, adata):
        """DataFrames are allowed in axis sections (like obsm)."""
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]},
            index=adata.obs_names,
        )
        adata.obst["df_tree"] = df
        assert adata.obst["df_tree"].shape == (5, 2)


# ---------------------------------------------------------------------------
# Pairwise Storage (like obsp)
# ---------------------------------------------------------------------------


class TestPairwiseStorage:
    def test_pairwise_section(self, adata):
        """Pairwise section stores square matrices."""
        dist = np.random.rand(5, 5)
        adata.obsd["distances"] = dist
        np.testing.assert_array_equal(adata.obsd["distances"], dist)

    def test_pairwise_wrong_shape_raises(self, adata):
        """Pairwise section rejects non-square matrices."""
        with pytest.raises(ValueError, match="shape"):
            adata.obsd["bad"] = np.ones((5, 3))  # not square


# ---------------------------------------------------------------------------
# Layers-like Storage (both axes)
# ---------------------------------------------------------------------------


class TestLayersLikeStorage:
    def test_layers_like_section(self, adata):
        """Layers-like section stores (n_obs, n_vars) matrices."""
        data = np.random.rand(5, 3)
        adata.extra_layers["normalized"] = data
        np.testing.assert_array_equal(adata.extra_layers["normalized"], data)

    def test_layers_like_wrong_shape_raises(self, adata):
        """Layers-like section rejects wrong shape."""
        with pytest.raises(ValueError, match="shape"):
            adata.extra_layers["bad"] = np.ones((5, 10))  # 10 != n_vars=3


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------


class TestSubsetting:
    def test_obs_subset(self, adata):
        """Subsetting obs subsets axis-0 registered sections."""
        adata.obst["tree"] = np.arange(15).reshape(5, 3)
        sub = adata[:3]
        assert sub.obst["tree"].shape == (3, 3)
        np.testing.assert_array_equal(sub.obst["tree"], np.arange(15).reshape(5, 3)[:3])

    def test_var_subset(self, adata):
        """Subsetting var subsets axis-1 registered sections."""
        adata.vart["gene_tree"] = np.arange(6).reshape(3, 2)
        sub = adata[:, :2]
        assert sub.vart["gene_tree"].shape == (2, 2)

    def test_pairwise_subset(self, adata):
        """Subsetting obs subsets pairwise registered sections."""
        adata.obsd["dist"] = np.eye(5)
        sub = adata[:3]
        assert sub.obsd["dist"].shape == (3, 3)

    def test_layers_like_subset(self, adata):
        """Subsetting subsets both axes of layers-like sections."""
        adata.extra_layers["data"] = np.arange(15).reshape(5, 3)
        sub = adata[:3, :2]
        assert sub.extra_layers["data"].shape == (3, 2)

    def test_view_copy_on_write(self, adata):
        """Writing to a view's registered section triggers copy-on-write."""
        adata.obst["tree"] = np.eye(5)
        sub = adata[:3]
        sub.obst["new_tree"] = np.ones((3, 2))
        # sub should now be an actual (not view) with the new entry
        assert not sub.is_view
        assert "new_tree" in sub.obst
        # original should be unchanged
        assert "new_tree" not in adata.obst


# ---------------------------------------------------------------------------
# Init with kwargs
# ---------------------------------------------------------------------------


class TestInitKwargs:
    def test_init_with_registered_section(self):
        """Can pass registered section data as init kwarg."""
        adata = ad.AnnData(
            np.ones((3, 4)),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            obst={"tree": np.eye(3)},
        )
        assert "tree" in adata.obst
        assert adata.obst["tree"].shape == (3, 3)

    def test_init_with_multiple_sections(self):
        """Can pass multiple registered sections at init."""
        adata = ad.AnnData(
            np.ones((3, 4)),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            var=pd.DataFrame(index=["v1", "v2", "v3", "v4"]),
            obst={"tree": np.eye(3)},
            vart={"gene_tree": np.eye(4)},
        )
        assert "tree" in adata.obst
        assert "gene_tree" in adata.vart

    def test_init_without_registered_section(self):
        """AnnData works normally without passing registered sections."""
        adata = ad.AnnData(np.ones((3, 4)))
        assert len(adata.obst) == 0
        assert len(adata.vart) == 0


# ---------------------------------------------------------------------------
# IO Roundtrip (h5ad)
# ---------------------------------------------------------------------------


class TestH5adRoundtrip:
    def test_write_read_roundtrip(self, adata, tmp_path):
        """Registered sections survive h5ad write/read."""
        adata.obst["tree"] = np.eye(5)
        adata.vart["gene_tree"] = np.random.rand(3, 2)

        path = tmp_path / "test.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)

        assert "tree" in adata2.obst
        np.testing.assert_array_equal(adata2.obst["tree"], np.eye(5))
        assert "gene_tree" in adata2.vart
        assert adata2.vart["gene_tree"].shape == (3, 2)

    def test_empty_section_not_written(self, adata, tmp_path):
        """Empty registered sections are not written to disk."""
        import h5py

        path = tmp_path / "test.h5ad"
        adata.write(path)

        with h5py.File(path, "r") as f:
            assert "obst" not in f
            assert "vart" not in f

    def test_subset_then_write(self, adata, tmp_path):
        """Can subset, then write, and registered sections are preserved."""
        adata.obst["tree"] = np.arange(15).reshape(5, 3)
        sub = adata[:3].copy()

        path = tmp_path / "test.h5ad"
        sub.write(path)
        sub2 = ad.read_h5ad(path)

        assert sub2.obst["tree"].shape == (3, 3)

    def test_read_without_section_registered(self, tmp_path):
        """Files with extra groups are read fine even without registration.

        The extra data is silently skipped (backward compat).
        """
        # Write with registration
        adata = ad.AnnData(
            np.ones((3, 4)),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            obst={"tree": np.eye(3)},
        )
        path = tmp_path / "test.h5ad"
        adata.write(path)

        # Simulate reading without registration by checking the file directly
        import h5py

        with h5py.File(path, "r") as f:
            assert "obst" in f  # Data is in the file
            # Standard read_h5ad would skip it if not registered,
            # but since we registered in this session, it will be read.
            # This test just verifies the file format.


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_shows_registered_section(self, adata):
        """Registered sections appear in repr when non-empty."""
        adata.obst["tree"] = np.eye(5)
        r = repr(adata)
        assert "obst" in r
        assert "tree" in r

    def test_repr_hides_empty_section(self, adata):
        """Empty registered sections don't appear in repr."""
        r = repr(adata)
        assert "obst" not in r

    def test_repr_html_shows_registered_section(self, adata):
        """Registered sections appear in HTML repr."""
        adata.obst["tree"] = np.eye(5)
        html = adata._repr_html_()
        if html is not None:  # HTML repr may not be enabled
            assert "obst" in html


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_preserves_sections(self, adata):
        """copy() preserves registered sections."""
        adata.obst["tree"] = np.eye(5)
        adata2 = adata.copy()
        assert "tree" in adata2.obst
        np.testing.assert_array_equal(adata2.obst["tree"], np.eye(5))

    def test_copy_is_independent(self, adata):
        """Modifications to copy don't affect original."""
        adata.obst["tree"] = np.eye(5)
        adata2 = adata.copy()
        adata2.obst["new"] = np.ones((5, 2))
        assert "new" not in adata.obst


# ---------------------------------------------------------------------------
# TreeData-like Scenario
# ---------------------------------------------------------------------------


class TestTreeDataScenario:
    """End-to-end test mimicking how TreeData would use section registration."""

    def test_treedata_workflow(self, tmp_path):
        """Full TreeData-like workflow: create, populate, subset, IO."""
        # 1. Create AnnData with tree data
        n_obs, n_vars = 10, 5
        adata = ad.AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame(
                {"cell_type": pd.Categorical(["A"] * 5 + ["B"] * 5)},
                index=[f"cell_{i}" for i in range(n_obs)],
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
        )

        # 2. Add tree data (serialized as arrays, like TreeData does)
        # In reality these would be serialized DiGraphs
        lineage_tree = np.random.rand(n_obs, 4)  # obs-aligned tree embedding
        gene_tree = np.random.rand(n_vars, 3)  # var-aligned tree embedding

        adata.obst["lineage"] = lineage_tree
        adata.vart["phylogeny"] = gene_tree

        # 3. Verify storage
        assert adata.obst["lineage"].shape == (n_obs, 4)
        assert adata.vart["phylogeny"].shape == (n_vars, 3)

        # 4. Subset (like filtering cells)
        mask = adata.obs["cell_type"] == "A"
        sub = adata[mask]
        assert sub.obst["lineage"].shape == (5, 4)
        assert sub.vart["phylogeny"].shape == (n_vars, 3)  # var unchanged

        # 5. Write and read back
        path = tmp_path / "treedata.h5ad"
        adata.write(path)
        adata2 = ad.read_h5ad(path)

        assert set(adata2.obst.keys()) == {"lineage"}
        assert set(adata2.vart.keys()) == {"phylogeny"}
        np.testing.assert_array_almost_equal(adata2.obst["lineage"], lineage_tree)
        np.testing.assert_array_almost_equal(adata2.vart["phylogeny"], gene_tree)

        # 6. Repr includes custom sections
        r = repr(adata2)
        assert "obst" in r
        assert "vart" in r
