#!/usr/bin/env python3
"""
Unit tests for SLAF append functionality.

This module tests the append functionality for existing SLAF datasets,
including compatibility validation, fragment management, and data integrity.
"""

import glob
import json
import os

import lance
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from slaf.data.converter import SLAFConverter
from slaf.integrations import ensure_h5ad_writable


def _write_pairwise_h5ad(path, *, prefix: str, values: np.ndarray) -> sc.AnnData:
    """Write a small H5AD with stable obsm and obsp schemas."""
    n_cells, n_genes = values.shape
    adata = sc.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(
            {"cell_type": ["type_a"] * n_cells},
            index=[f"{prefix}_{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    adata.obsm["embedding"] = np.column_stack(
        [np.arange(n_cells), np.arange(n_cells) + 0.5]
    ).astype(np.float32)
    graph = sparse.lil_matrix((n_cells, n_cells), dtype=np.float32)
    for index in range(n_cells - 1):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 1.0
    adata.obsp["connectivities"] = graph.tocsr()
    ensure_h5ad_writable(adata)
    adata.write_h5ad(path)
    return adata


class TestAppendFunctionality:
    """Test the append functionality for existing SLAF datasets."""

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path):
        """Create synthetic test data for append testing."""
        # Create compatible files
        compatible_dir = os.path.join(tmp_path, "compatible")
        os.makedirs(compatible_dir, exist_ok=True)

        # Create 3 compatible h5ad files
        for i in range(3):
            np.random.seed(42 + i)
            n_cells, n_genes = 50, 30
            X = np.random.negative_binomial(5, 0.1, (n_cells, n_genes))
            X_sparse = sparse.csr_matrix(X)

            # Create cell metadata with consistent schema
            cell_ids = [f"cell_{i}_{j:03d}" for j in range(n_cells)]
            cell_types = np.random.choice(["T_cell", "B_cell"], n_cells)
            batches = [f"batch_{i}" for _ in range(n_cells)]

            obs_df = pd.DataFrame(
                {
                    "cell_type": cell_types,
                    "batch": batches,
                    "n_genes": [20] * n_cells,
                    "total_counts": [1000] * n_cells,
                },
                index=cell_ids,
            )

            # Create gene metadata
            gene_ids = [f"GENE_{j:03d}" for j in range(n_genes)]
            var_df = pd.DataFrame(
                {
                    "gene_type": ["protein_coding"] * n_genes,
                    "gene_name": [f"Gene_{j:03d}" for j in range(n_genes)],
                },
                index=gene_ids,
            )

            # Create AnnData object
            adata = sc.AnnData(X=X_sparse, obs=obs_df, var=var_df)
            ensure_h5ad_writable(adata)

            # Save as h5ad
            output_file = os.path.join(compatible_dir, f"synthetic_data_{i:02d}.h5ad")
            adata.write_h5ad(output_file)

        # Create incompatible files
        incompatible_dir = os.path.join(tmp_path, "incompatible")
        os.makedirs(incompatible_dir, exist_ok=True)

        # File with different genes
        np.random.seed(999)
        X_incompatible = np.random.negative_binomial(
            5, 0.1, (n_cells, 20)
        )  # Different gene count
        X_incompatible_sparse = sparse.csr_matrix(X_incompatible)

        incompatible_gene_ids = [f"DIFFERENT_GENE_{j:03d}" for j in range(20)]
        incompatible_cell_ids = [f"incompatible_cell_{j:03d}" for j in range(n_cells)]

        obs_incompatible = pd.DataFrame(
            {
                "cell_type": ["T_cell"] * n_cells,
                "batch": ["incompatible_batch"] * n_cells,
                "n_genes": [20] * n_cells,
                "total_counts": [1000] * n_cells,
            },
            index=incompatible_cell_ids,
        )

        var_incompatible = pd.DataFrame(
            {
                "gene_type": ["protein_coding"] * 20,
                "gene_name": [f"DifferentGene_{j:03d}" for j in range(20)],
            },
            index=incompatible_gene_ids,
        )

        adata_incompatible = sc.AnnData(
            X=X_incompatible_sparse, obs=obs_incompatible, var=var_incompatible
        )
        ensure_h5ad_writable(adata_incompatible)
        incompatible_file = os.path.join(incompatible_dir, "incompatible_genes.h5ad")
        adata_incompatible.write_h5ad(incompatible_file)

        return {
            "compatible_dir": compatible_dir,
            "incompatible_dir": incompatible_dir,
            "compatible_files": sorted(
                glob.glob(os.path.join(compatible_dir, "*.h5ad"))
            ),
            "incompatible_files": sorted(
                glob.glob(os.path.join(incompatible_dir, "*.h5ad"))
            ),
        }

    def test_append_single_file(self, synthetic_data_dir, tmp_path):
        """Test appending a single file to existing SLAF dataset."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Verify initial dataset
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cell_count = len(initial_cells_dataset.to_table())
        assert initial_cell_count == 50

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

        # Check auto-incrementing IDs
        final_cells_table = final_cells_dataset.to_table()
        cell_integer_ids = final_cells_table.column("cell_integer_id").to_numpy()
        expected_ids = set(range(100))
        actual_ids = set(cell_integer_ids)
        assert actual_ids == expected_ids

        # Check source file tracking
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {"original_data", "synthetic_data_01.h5ad"}
        assert unique_sources == expected_sources

    def test_append_multiple_files(self, synthetic_data_dir, tmp_path):
        """Test appending multiple files to existing SLAF dataset."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        # Copy remaining files to a directory for appending
        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        # Append remaining files
        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 150  # 3 files × 50 cells each

        # Check auto-incrementing IDs
        final_cells_table = final_cells_dataset.to_table()
        cell_integer_ids = final_cells_table.column("cell_integer_id").to_numpy()
        expected_ids = set(range(150))
        actual_ids = set(cell_integer_ids)
        assert actual_ids == expected_ids

        # Check source file tracking
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {
            "original_data",
            "remaining_00.h5ad",
            "remaining_01.h5ad",
        }
        assert unique_sources == expected_sources

    def test_append_fragment_structure(self, synthetic_data_dir, tmp_path):
        """Test that append creates correct fragment structure."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Check fragment structure
        expression_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "expression.lance")
        )
        cells_dataset = lance.dataset(os.path.join(initial_slaf_path, "cells.lance"))
        genes_dataset = lance.dataset(os.path.join(initial_slaf_path, "genes.lance"))

        expression_fragments = expression_dataset.get_fragments()
        cells_fragments = cells_dataset.get_fragments()
        genes_fragments = genes_dataset.get_fragments()

        # Expected: 3 files × (50 cells / 25 chunk_size) = 6 fragments for expression
        assert len(expression_fragments) == 6
        # Expected: 3 fragments for cells (1 per file)
        assert len(cells_fragments) == 3
        # Expected: 1 fragment for genes (from initial file)
        assert len(genes_fragments) == 1

    def test_append_config_metadata(self, synthetic_data_dir, tmp_path):
        """Test that append updates config metadata correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Check config file
        config_path = os.path.join(initial_slaf_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Should have multi_file section
        assert "multi_file" in config
        assert "source_files" in config["multi_file"]
        assert len(config["multi_file"]["source_files"]) == 2  # 2 appended files
        assert config["n_cells"] == 150  # 3 files × 50 cells each

        # Check source file information
        source_files = config["multi_file"]["source_files"]
        assert len(source_files) == 2
        assert source_files[0]["file_name"] == "remaining_00.h5ad"
        assert source_files[1]["file_name"] == "remaining_01.h5ad"

    def test_append_compatibility_validation(self, synthetic_data_dir, tmp_path):
        """Test that append validates compatibility correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Try to append incompatible file
        incompatible_file = synthetic_data_dir["incompatible_files"][0]

        with pytest.raises(ValueError, match="Validation failed"):
            converter.append(str(incompatible_file), str(initial_slaf_path))

    def test_append_creates_nonexistent_dataset(self, synthetic_data_dir, tmp_path):
        """The first append initializes the destination and records its source."""
        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        first_file = synthetic_data_dir["compatible_files"][0]
        nonexistent_slaf = os.path.join(tmp_path, "nonexistent.slaf")

        converter.append(str(first_file), str(nonexistent_slaf))

        cells = lance.dataset(os.path.join(nonexistent_slaf, "cells.lance"))
        assert cells.count_rows() == 50
        assert set(
            cells.to_table(columns=["source_file"])["source_file"].to_pylist()
        ) == {os.path.basename(first_file)}
        with open(os.path.join(nonexistent_slaf, "config.json")) as config_file:
            config = json.load(config_file)
        assert config["multi_file"] == {
            "source_files": [
                {
                    "file_path": first_file,
                    "file_name": os.path.basename(first_file),
                    "n_cells": 50,
                    "cell_offset": 0,
                }
            ],
            "total_files": 1,
            "total_cells_from_files": 50,
        }

    def test_append_create_then_append_uses_uniform_workflow(
        self, synthetic_data_dir, tmp_path
    ):
        """Repeated append calls initialize once and then append normally."""
        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )
        output_path = tmp_path / "streamed.slaf"
        first_file, second_file = synthetic_data_dir["compatible_files"][:2]

        converter.append(first_file, str(output_path))
        converter.append(second_file, str(output_path))

        cells = lance.dataset(output_path / "cells.lance")
        assert cells.count_rows() == 100
        assert set(
            cells.to_table(columns=["source_file"])["source_file"].to_pylist()
        ) == {
            os.path.basename(first_file),
            os.path.basename(second_file),
        }
        with open(output_path / "config.json") as config_file:
            config = json.load(config_file)
        assert [
            source["file_name"] for source in config["multi_file"]["source_files"]
        ] == [os.path.basename(first_file), os.path.basename(second_file)]
        manifest = converter.get_source_manifest(str(output_path))
        assert [source["file_name"] for source in manifest] == [
            os.path.basename(first_file),
            os.path.basename(second_file),
        ]
        assert (
            converter.get_gene_order(str(output_path))
            == sc.read_h5ad(first_file).var_names.tolist()
        )

    def test_append_sanitizes_metadata_columns_before_schema_validation(self, tmp_path):
        """Append compares incoming metadata using Lance-compatible names."""
        input_paths = []
        for file_index in range(2):
            input_path = tmp_path / f"input_{file_index}.h5ad"
            observations = pd.DataFrame(
                {
                    "orig.ident": [f"sample_{file_index}"] * 2,
                    "nCount_Spatial.1": [file_index + 1, file_index + 2],
                },
                index=[f"cell_{file_index}_{index}" for index in range(2)],
            )
            adata = sc.AnnData(
                X=sparse.csr_matrix(np.eye(2, dtype=np.float32)),
                obs=observations,
                var=pd.DataFrame(index=["gene_0", "gene_1"]),
            )
            adata.write_h5ad(input_path)
            input_paths.append(input_path)

        output_path = tmp_path / "dotted_columns.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )

        converter.append(str(input_paths[0]), str(output_path))
        converter.append(str(input_paths[1]), str(output_path))

        cells = lance.dataset(output_path / "cells.lance").to_table()
        assert cells.num_rows == 4
        assert "orig_ident" in cells.column_names
        assert "nCount_Spatial_1" in cells.column_names
        assert cells["orig_ident"].to_pylist() == [
            "sample_0",
            "sample_0",
            "sample_1",
            "sample_1",
        ]
        assert cells["nCount_Spatial_1"].to_pylist() == [1, 2, 2, 3]

    def test_conversion_rejects_metadata_sanitization_collisions(self, tmp_path):
        """Distinct source columns cannot map to one Lance field name."""
        input_path = tmp_path / "colliding_columns.h5ad"
        adata = sc.AnnData(
            X=sparse.csr_matrix([[1.0]]),
            obs=pd.DataFrame(
                {"source.name": ["first"], "source_name": ["second"]},
                index=["cell_0"],
            ),
            var=pd.DataFrame(index=["gene_0"]),
        )
        adata.write_h5ad(input_path)

        with pytest.raises(ValueError, match="collide after replacing"):
            SLAFConverter().append(
                str(input_path),
                str(tmp_path / "colliding_columns.slaf"),
            )

    def test_append_rejects_directory_initialization(
        self, synthetic_data_dir, tmp_path
    ):
        """A missing destination cannot be initialized from a directory."""
        converter = SLAFConverter()
        output_path = tmp_path / "directory.slaf"

        with pytest.raises(ValueError, match="Cannot initialize.*directory"):
            converter.append(synthetic_data_dir["compatible_dir"], str(output_path))

        assert not output_path.exists()

    def test_append_rejects_incomplete_existing_destination(
        self, synthetic_data_dir, tmp_path
    ):
        """An existing partial directory is not treated as a new SLAF."""
        converter = SLAFConverter()
        output_path = tmp_path / "incomplete.slaf"
        output_path.mkdir()

        with pytest.raises(ValueError, match="Existing SLAF dataset is incomplete"):
            converter.append(
                synthetic_data_dir["compatible_files"][0],
                str(output_path),
            )

        assert list(output_path.iterdir()) == []

    def test_append_empty_directory(self, synthetic_data_dir, tmp_path):
        """Test that append fails gracefully with empty directory."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Create empty directory
        empty_dir = os.path.join(tmp_path, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        with pytest.raises(ValueError, match="No supported files found in directory"):
            converter.append(str(empty_dir), str(initial_slaf_path))

    def test_append_source_file_column_handling(self, synthetic_data_dir, tmp_path):
        """Test that append handles source_file column correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Check that initial dataset doesn't have source_file column
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cells_table = initial_cells_dataset.to_table()
        initial_columns = set(initial_cells_table.column_names)
        assert "source_file" not in initial_columns

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Check that final dataset has source_file column
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cells_table = final_cells_dataset.to_table()
        final_columns = set(final_cells_table.column_names)
        assert "source_file" in final_columns

        # Check source file distribution
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {"original_data", "synthetic_data_01.h5ad"}
        assert unique_sources == expected_sources

    def test_append_chunk_size_affects_fragments(self, synthetic_data_dir, tmp_path):
        """Test that different chunk sizes result in different fragment counts."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        # Test with small chunk size
        converter_small = SLAFConverter(
            chunked=True,
            chunk_size=10,  # Small chunks
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        converter_small.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter_small.append(str(remaining_dir), str(initial_slaf_path))

        # Check fragment count with small chunks
        expression_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "expression.lance")
        )
        expression_fragments_small = expression_dataset.get_fragments()

        # Expected: 3 files × (50 cells / 10 chunk_size) = 15 fragments
        assert len(expression_fragments_small) == 15

    def test_append_memory_efficiency(self, synthetic_data_dir, tmp_path):
        """Test that append is memory efficient with small chunks."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=10,  # Very small chunks for memory testing
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        # This should not cause memory issues
        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Verify output is correct
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 150

    def test_append_preserves_existing_data(self, synthetic_data_dir, tmp_path):
        """Test that append preserves existing data integrity."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Get initial data for comparison
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cells_table = initial_cells_dataset.to_table()
        initial_cell_ids = set(initial_cells_table.column("cell_id").to_numpy())

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Check that initial data is preserved
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cells_table = final_cells_dataset.to_table()
        final_cell_ids = set(final_cells_table.column("cell_id").to_numpy())

        # Initial cell IDs should still be present
        assert initial_cell_ids.issubset(final_cell_ids)

        # Check that new data is added
        assert len(final_cell_ids) > len(initial_cell_ids)

    def test_append_error_handling(self, synthetic_data_dir, tmp_path):
        """Test that append handles errors gracefully."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Try to append to a file that doesn't exist
        nonexistent_file = os.path.join(tmp_path, "nonexistent.h5ad")
        with open(nonexistent_file, "w") as f:
            f.write("not a valid h5ad file")

        with pytest.raises(ValueError, match="Validation failed for nonexistent.h5ad"):
            converter.append(str(nonexistent_file), str(initial_slaf_path))

    def test_append_with_skip_validation(self, synthetic_data_dir, tmp_path):
        """Test that append works with skip_validation flag."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file (should work without validation)
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

    def test_append_auto_detection(self, synthetic_data_dir, tmp_path):
        """Test that append auto-detects file format correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file with auto-detection
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path), input_format="auto")

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

    def test_append_explicit_format(self, synthetic_data_dir, tmp_path):
        """Test that append works with explicit format specification."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file with explicit format
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path), input_format="h5ad")

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

    def test_append_preserves_obsm_obsp_and_statistics(self, tmp_path):
        """Chunked initialization and append preserve obsm, obsp, and metadata."""
        first_path = tmp_path / "first.h5ad"
        second_path = tmp_path / "second.h5ad"
        first = _write_pairwise_h5ad(
            first_path,
            prefix="first",
            values=np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32),
        )
        second = _write_pairwise_h5ad(
            second_path,
            prefix="second",
            values=np.array([[4, 0, 5], [0, 6, 0], [7, 0, 8]], dtype=np.float32),
        )
        output_path = tmp_path / "pairwise.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )

        converter.append(str(first_path), str(output_path))
        converter.append(str(second_path), str(output_path))

        from slaf.core.slaf import SLAFArray
        from slaf.integrations.anndata import LazyAnnData

        lazy = LazyAnnData(SLAFArray(output_path, load_metadata=False))
        np.testing.assert_allclose(
            lazy.obsm["embedding"],
            np.vstack([first.obsm["embedding"], second.obsm["embedding"]]),
        )
        expected_graph = sparse.block_diag(
            [
                first.obsp["connectivities"],
                second.obsp["connectivities"],
            ],
            format="csr",
        )
        np.testing.assert_allclose(
            lazy.obsp["connectivities"].toarray(),
            expected_graph.toarray(),
        )

        with open(output_path / "config.json") as config_file:
            config = json.load(config_file)
        all_nonzero = np.concatenate([first.X.data, second.X.data]).astype(np.float64)
        assert config["n_cells"] == 5
        assert config["obsm"]["dimensions"]["embedding"] == 2
        assert config["obsp"]["dimensions"]["connectivities"] == 5
        assert config["metadata"]["expression_count"] == len(all_nonzero)
        assert config["metadata"]["expression_stats"]["mean_value"] == pytest.approx(
            all_nonzero.mean()
        )
        assert config["metadata"]["expression_stats"]["std_value"] == pytest.approx(
            all_nonzero.std(ddof=1)
        )

    def test_append_rejects_reordered_genes_before_writing(self, tmp_path):
        """Matching gene sets with a different token order are incompatible."""
        first_path = tmp_path / "first.h5ad"
        reordered_path = tmp_path / "reordered.h5ad"
        first = _write_pairwise_h5ad(
            first_path,
            prefix="first",
            values=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        reordered = first[:, ::-1].copy()
        reordered.obs_names = ["reordered_0", "reordered_1"]
        reordered.write_h5ad(reordered_path)
        output_path = tmp_path / "ordered.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(first_path), str(output_path))
        initial_cells = lance.dataset(output_path / "cells.lance").count_rows()
        initial_expression = lance.dataset(
            output_path / "expression.lance"
        ).count_rows()

        with pytest.raises(ValueError, match="Gene order differs"):
            converter.append(str(reordered_path), str(output_path))

        assert lance.dataset(output_path / "cells.lance").count_rows() == initial_cells
        assert (
            lance.dataset(output_path / "expression.lance").count_rows()
            == initial_expression
        )

    @pytest.mark.parametrize("mismatch", ["missing_obsm", "obsm_width", "missing_obsp"])
    def test_append_rejects_obsm_obsp_schema_mismatch(self, tmp_path, mismatch):
        """Append requires the established obsm and obsp contract."""
        first_path = tmp_path / "first.h5ad"
        mismatch_path = tmp_path / f"{mismatch}.h5ad"
        first = _write_pairwise_h5ad(
            first_path,
            prefix="first",
            values=np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32),
        )
        candidate = first.copy()
        candidate.obs_names = [f"candidate_{index}" for index in range(candidate.n_obs)]
        if mismatch == "missing_obsm":
            del candidate.obsm["embedding"]
        elif mismatch == "obsm_width":
            candidate.obsm["embedding"] = np.ones(
                (candidate.n_obs, 3), dtype=np.float32
            )
        else:
            del candidate.obsp["connectivities"]
        candidate.write_h5ad(mismatch_path)
        output_path = tmp_path / "pairwise.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(first_path), str(output_path))

        with pytest.raises(ValueError, match="obsm|obsp"):
            converter.append(str(mismatch_path), str(output_path))

        assert lance.dataset(output_path / "cells.lance").count_rows() == first.n_obs

    def test_append_stops_after_first_processing_failure(
        self, synthetic_data_dir, tmp_path, monkeypatch
    ):
        """A runtime failure prevents append from attempting later files."""
        output_path = tmp_path / "fail_fast.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            compact_after_write=False,
        )
        converter.convert(synthetic_data_dir["compatible_files"][0], str(output_path))
        attempts = 0

        def fail_processing(*args, **kwargs):
            nonlocal attempts
            del args, kwargs
            attempts += 1
            raise RuntimeError("injected append failure")

        monkeypatch.setattr(
            converter,
            "_process_file_chunks_with_checkpoint",
            fail_processing,
        )

        with pytest.raises(RuntimeError, match="injected append failure"):
            converter.append(
                synthetic_data_dir["compatible_dir"],
                str(output_path),
            )

        assert attempts == 1

    def test_finalize_validates_counts_and_creates_indices(self, tmp_path):
        """Finalization creates indexes once streaming appends are complete."""
        input_path = tmp_path / "input.h5ad"
        _write_pairwise_h5ad(
            input_path,
            prefix="cell",
            values=np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32),
        )
        output_path = tmp_path / "finalized.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(input_path), str(output_path))

        converter.finalize(str(output_path), create_indices=True)

        indexed_fields = {
            field
            for index in lance.dataset(
                output_path / "cellsxcells.lance"
            ).describe_indices()
            for field in index.field_names
        }
        assert "cell_integer_id_i" in indexed_fields

    def test_finalize_can_skip_expression_indices(self, tmp_path):
        """Finalization can omit expression indices without omitting graph indices."""
        input_path = tmp_path / "input.h5ad"
        _write_pairwise_h5ad(
            input_path,
            prefix="cell",
            values=np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32),
        )
        output_path = tmp_path / "finalized.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=2,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(input_path), str(output_path))

        converter.finalize(
            str(output_path),
            create_indices=True,
            create_expression_indices=False,
        )
        converter.finalize(
            str(output_path),
            create_indices=True,
            create_expression_indices=False,
        )

        expression_fields = {
            field
            for index in lance.dataset(
                output_path / "expression.lance"
            ).describe_indices()
            for field in index.field_names
        }
        graph_fields = {
            field
            for index in lance.dataset(
                output_path / "cellsxcells.lance"
            ).describe_indices()
            for field in index.field_names
        }
        assert expression_fields == set()
        assert "cell_integer_id_i" in graph_fields

    def test_finalize_compacts_expression_without_changing_rows(self, tmp_path):
        """Finalization merges small expression fragments and preserves order."""
        input_path = tmp_path / "input.h5ad"
        _write_pairwise_h5ad(
            input_path,
            prefix="cell",
            values=np.array(
                [
                    [1, 0, 2],
                    [0, 3, 0],
                    [4, 0, 5],
                    [0, 6, 0],
                ],
                dtype=np.float32,
            ),
        )
        output_path = tmp_path / "compacted.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=1,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(input_path), str(output_path))
        expression_path = output_path / "expression.lance"
        before = lance.dataset(expression_path)
        before_fragments = len(list(before.get_fragments()))
        before_table = before.to_table()

        converter.finalize(
            str(output_path),
            create_indices=False,
            compact=True,
            expression_target_rows_per_fragment=100,
        )

        after = lance.dataset(expression_path)
        assert len(list(after.get_fragments())) < before_fragments
        assert len(after.versions()) == 1
        assert len(lance.dataset(output_path / "cells.lance").versions()) == 1
        assert len(lance.dataset(output_path / "genes.lance").versions()) == 1
        assert after.to_table().equals(before_table)

    def test_compact_expression_discards_obsolete_versions(self, tmp_path):
        """Incremental expression compaction retains exact rows and one version."""
        input_path = tmp_path / "input.h5ad"
        _write_pairwise_h5ad(
            input_path,
            prefix="cell",
            values=np.array(
                [[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]],
                dtype=np.float32,
            ),
        )
        output_path = tmp_path / "compacted.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=1,
            create_indices=False,
            compact_after_write=False,
            use_optimized_dtypes=False,
        )
        converter.convert(str(input_path), str(output_path))
        expression_path = output_path / "expression.lance"
        before = lance.dataset(expression_path)
        before_table = before.to_table()
        before_fragments = len(list(before.get_fragments()))

        converter.compact_expression(
            str(output_path),
            target_rows_per_fragment=100,
        )

        after = lance.dataset(expression_path)
        assert len(list(after.get_fragments())) < before_fragments
        assert len(after.versions()) == 1
        assert after.to_table().equals(before_table)
