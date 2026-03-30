"""Tests for register_section decorator.

Validates that registered sections behave correctly for all alignment
combinations, custom validation, custom subsetting, custom IO, and
HTML repr integration. Uses TreeData-like and SpatialData-like scenarios.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import anndata as ad
from anndata.extensions import register_section


# ---------------------------------------------------------------------------
# Fixtures: register sections once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _register_test_sections():
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

    # Custom subset
    if "sec_custom_subset" not in ad.AnnData._registered_sections:

        @register_section("sec_custom_subset", alignment="obs")
        class SecCustomSubset:
            @staticmethod
            def subset(value, idx):
                # Custom: return a dict describing the subset
                return {"original": value, "subset_idx": idx}


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
        with pytest.raises(AttributeError, match="conflicts with"):
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
        np.testing.assert_array_equal(
            adata2.sec_both["x"], np.arange(15).reshape(5, 3)
        )

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
