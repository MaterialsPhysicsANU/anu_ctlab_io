import json
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    from dask.array.core import PerformanceWarning
    from PIL import Image

    import anu_ctlab_io.zarr
    from anu_ctlab_io.zarr import OMEZarrVersion
except ImportError:
    pytest.skip("Requires 'zarr' extra", allow_module_level=True)

import anu_ctlab_io


def test_write_single_ome_zarr(_make_dataset):
    """Test writing a single OME-Zarr group."""
    shape = (10, 20, 30)
    dataset, data = _make_dataset(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.zarr"

        # Write the dataset as OME-Zarr
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_dataset",
            ome_zarr_version=OMEZarrVersion.v05,
        )

        # Verify file exists
        assert output_path.exists()
        assert output_path.is_dir()

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.dimension_names == dataset.dimension_names


def test_write_single_zarr_array(_make_dataset):
    """Test writing a simple Zarr V3 array with mango metadata."""
    shape = (10, 20, 30)
    dataset, data = _make_dataset(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.zarr"

        # Write as simple Zarr array
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_dataset",
            ome_zarr_version=None,
        )

        # Verify file exists
        assert output_path.exists()
        assert output_path.is_dir()

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.dimension_names == dataset.dimension_names


@pytest.mark.parametrize(
    ("ome_zarr_version", "num_chunks", "expected_separator"),
    [
        (None, 63, "."),
        (None, 64, "."),
        (None, 65, "/"),
        (OMEZarrVersion.v05, 63, "."),
        (OMEZarrVersion.v05, 64, "."),
        (OMEZarrVersion.v05, 65, "/"),
    ],
)
def test_chunk_key_separator_depends_on_number_of_chunks(
    _make_dataset, ome_zarr_version, num_chunks, expected_separator
):
    """Use flat chunk keys when the number of chunks does not exceed the threshold."""
    dataset, _ = _make_dataset((num_chunks, 1, 1), chunks=(1, 1, 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "separator.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=ome_zarr_version,
            chunks=(1, 1, 1),
            shards=None,
            slice_thumbnails=False,
        )

        metadata_path = output_path / "zarr.json"
        if ome_zarr_version is not None:
            metadata_path = output_path / "0" / "zarr.json"
        metadata = json.loads(metadata_path.read_text())
        assert (
            metadata["chunk_key_encoding"]["configuration"]["separator"]
            == expected_separator
        )


@pytest.mark.parametrize(
    ("ome_zarr_version", "num_shards", "expected_separator"),
    [
        (None, 63, "."),
        (None, 64, "."),
        (None, 65, "/"),
        (OMEZarrVersion.v05, 63, "."),
        (OMEZarrVersion.v05, 64, "."),
        (OMEZarrVersion.v05, 65, "/"),
    ],
)
def test_chunk_key_separator_uses_shards_when_sharded(
    _make_dataset, ome_zarr_version, num_shards, expected_separator
):
    """Use flat chunk keys based on number of shards, not inner chunks, when sharded."""
    # Create an array where inner chunks would be >64 but shards are <64.
    dataset, _ = _make_dataset((num_shards, 1, 1), chunks=(1, 1, 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "separator_sharded.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=ome_zarr_version,
            chunks=(1, 1, 1),
            shards=(1, 1, 1),
            slice_thumbnails=False,
        )

        metadata_path = output_path / "zarr.json"
        if ome_zarr_version is not None:
            metadata_path = output_path / "0" / "zarr.json"
        metadata = json.loads(metadata_path.read_text())
        assert (
            metadata["chunk_key_encoding"]["configuration"]["separator"]
            == expected_separator
        )


def test_dimension_separator_threshold_can_be_overridden(_make_dataset, tmp_path):
    """The dimension separator threshold can be configured by the caller."""
    dataset, _ = _make_dataset((20, 1, 1), chunks=(1, 1, 1))
    output_path = tmp_path / "separator.zarr"

    dataset.to_path(
        output_path,
        ome_zarr_version=None,
        chunks=(1, 1, 1),
        shards=None,
        slice_thumbnails=False,
        dimension_separator_threshold=21,
    )

    metadata = json.loads((output_path / "zarr.json").read_text())
    assert metadata["chunk_key_encoding"]["configuration"]["separator"] == "."


def test_none_dimension_separator_threshold_uses_zarr_default(_make_dataset, tmp_path):
    """A None threshold leaves chunk key encoding selection to Zarr."""
    dataset, _ = _make_dataset((1, 1, 1), chunks=(1, 1, 1))
    output_path = tmp_path / "separator.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        ome_zarr_version=None,
        chunks=(1, 1, 1),
        shards=None,
        slice_thumbnails=False,
        dimension_separator_threshold=None,
    )

    metadata = json.loads((output_path / "zarr.json").read_text())
    assert metadata["chunk_key_encoding"]["configuration"]["separator"] == "/"


def test_write_without_datatype(_make_dataset):
    """Test writing OME-Zarr without mango metadata (no datatype)."""
    shape = (5, 10, 15)
    dataset, data = _make_dataset(
        shape,
        dtype=np.float32,
        voxel_unit=anu_ctlab_io.VoxelUnit.UM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_no_mango.zarr"

        # Write without mango metadata (no datatype)
        anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path)

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify basic properties
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.UM
        assert np.allclose(read_dataset.voxel_size, (1.0, 1.0, 1.0))


def test_write_zarr_element_targets_control_auto_layout(_make_dataset):
    """Integer chunk and shard specs control automatic layouts by element count."""
    import zarr

    shape = (100, 20, 60)
    dataset, data = _make_dataset(shape, chunks=(10, 20, 30))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_sharded.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_sharded_dataset",
            chunks=8**3,
            shards=16**3,
        )

        array = zarr.open_array(output_path / "0", mode="r")
        assert array.chunks == (8, 8, 8)
        assert array.shards == (16, 16, 16)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert loaded_dataset.data.shape == shape
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_roundtrip_against_reference():
    """Test roundtrip by comparing with reference Zarr files."""
    # Read existing test file
    original_dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes.zarr")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "roundtrip.zarr"

        # Write it as simple array (to match source format)
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            output_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test",
            ome_zarr_version=None,
        )

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify data matches
        assert read_dataset.data.shape == original_dataset.data.shape
        assert np.array_equal(
            read_dataset.data.compute(), original_dataset.data.compute()
        )
        assert np.allclose(read_dataset.voxel_size, original_dataset.voxel_size)
        assert read_dataset.voxel_unit == original_dataset.voxel_unit
        assert read_dataset.dimension_names == original_dataset.dimension_names


def test_roundtrip_ome_zarr():
    """Test roundtrip with OME-Zarr format."""
    # Read existing OME-Zarr test file
    original_dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS_AM.zarr")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "ome_roundtrip.zarr"

        # Write it as OME-Zarr
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            output_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="ome_roundtrip_test",
            ome_zarr_version=OMEZarrVersion.v05,
        )

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify data matches
        assert read_dataset.data.shape == original_dataset.data.shape
        assert np.array_equal(
            read_dataset.data.compute(), original_dataset.data.compute()
        )
        assert np.allclose(read_dataset.voxel_size, original_dataset.voxel_size)
        assert read_dataset.voxel_unit == original_dataset.voxel_unit


def test_to_path_auto_detection(_make_dataset):
    """Test that Dataset.to_path() correctly auto-detects .zarr extension."""
    shape = (5, 10, 15)
    dataset, data = _make_dataset(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "auto_detect.zarr"

        # Use to_path with auto-detection
        dataset.to_path(output_path)

        # Verify it was written as Zarr
        assert output_path.exists()
        assert output_path.is_dir()

        # Read back and verify
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


def test_write_with_history(_make_dataset):
    """Test writing with custom history metadata."""
    shape = (5, 10, 15)
    custom_history = {
        "step1": {"operation": "reconstruction", "timestamp": "2024-01-01"},
        "step2": {"operation": "filtering", "timestamp": "2024-01-02"},
    }

    dataset, _ = _make_dataset(shape, history=custom_history)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "history_test.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, ome_zarr_version=None)

        # Read back and verify history is preserved
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert isinstance(read_dataset.history, dict)
        assert "step1" in read_dataset.history
        assert "step2" in read_dataset.history


def test_write_different_dtypes(_make_dataset):
    """Test writing with different numpy dtypes."""
    dtypes = [np.uint8, np.uint16, np.int16, np.float32]

    for dtype in dtypes:
        shape = (5, 10, 15)
        dataset, data = _make_dataset(shape, dtype=dtype, datatype=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"dtype_{dtype.__name__}.zarr"
            anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path)

            # Read back and verify dtype preserved
            read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
            assert read_dataset.data.dtype == dtype
            assert np.array_equal(read_dataset.data.compute(), data.compute())


def test_write_with_explicit_chunks_and_shards(_make_dataset):
    """Test writing with user-provided chunk and shard shapes."""
    shape = (100, 50, 60)
    dataset, data = _make_dataset(shape, chunks=(10, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "explicit_shapes.zarr"

        # Write with explicit chunks and shards
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            chunks=(5, 50, 60),
            shards=(20, 50, 60),
        )

        assert output_path.exists()

        # Read back and verify
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


def test_write_explicit_shapes_roundtrip(_make_dataset):
    """Test roundtrip with explicit chunk and shard shapes."""
    shape = (60, 40, 50)
    dataset, data = _make_dataset(
        shape,
        data=np.random.randint(0, 1000, shape, dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "shapes_roundtrip.zarr"

        # Write with custom chunks and shards
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            chunks=(10, 40, 50),
            shards=(30, 40, 50),
            ome_zarr_version=None,
        )

        # Read back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify complete data integrity
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.voxel_unit == dataset.voxel_unit
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)


@pytest.mark.xfail(reason="This was the behaviour in <=1.2.2, but it is not useful")
def test_error_chunks_without_shards(_make_dataset):
    """Test that providing chunks without shards raises ValueError."""
    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.raises(ValueError, match="chunks and shards must both be provided"):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks=(5, 20, 30),
            )


def test_error_shards_without_chunks(_make_dataset):
    """Test that providing shards without chunks raises ValueError."""
    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.raises(
            ValueError, match="shards cannot be provided without explicit chunks"
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                shards=(10, 20, 30),
            )

        with pytest.raises(
            ValueError, match="shards cannot be provided without explicit chunks"
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks="auto",
                shards=(10, 20, 30),
            )


def test_explicit_shapes_ignore_deprecated_chunk_size(_make_dataset):
    """Deprecated size parameters are ignored when explicit shapes are provided."""
    import zarr

    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.warns(UserWarning, match="chunk_size_mb is ignored"):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks=(5, 20, 30),
                shards=(10, 20, 30),
                chunk_size_mb=5.0,
            )

        array = zarr.open_array(output_path / "0", mode="r")
        assert array.chunks == (5, 20, 30)
        assert array.shards == (10, 20, 30)


def test_auto_shards_ignore_deprecated_max_shard_size(_make_dataset):
    """Deprecated shard size is ignored when selecting an auto shard layout."""
    import zarr

    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.warns(UserWarning, match="max_shard_size_mb is ignored"):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks=(5, 20, 30),
                shards="auto",
                max_shard_size_mb=0.02,
            )

        array = zarr.open_array(output_path / "0", mode="r")
        assert array.chunks == (5, 20, 30)
        assert array.shards == (10, 20, 30)


def test_user_shapes_used_without_validation(_make_dataset):
    """Test that user-provided shapes are used directly without validation."""
    shape = (100, 50, 60)
    dataset, _ = _make_dataset(
        shape,
        data=np.zeros(shape, dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "unvalidated_shapes.zarr"

        # Use shapes that wouldn't normally be chosen by automatic calculation
        # (e.g., single z-slice chunks which violate MIN_Z_SLICES=2 rule)
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            chunks=(1, 50, 60),  # Single slice - would normally be rejected
            shards=(25, 50, 60),  # Not a perfect divisor of 100
        )

        # Should succeed without validation errors
        assert output_path.exists()

        # Data should still be readable
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert read_dataset.data.shape == shape


def test_irregular_dask_chunks_write_correct_data(_make_dataset):
    """Regression test: dask chunks smaller than the shard must not produce zeros (corruption) in output.

    dask's to_zarr, when writing to an existing zarr.Array, calls normalize_chunks("auto")
    internally and rechunks to that size before writing — ignoring the caller's chunk
    structure entirely. For large arrays the auto size is often smaller than the shard
    and not a multiple of it, so each dask chunk writes a partial shard that is then
    overwritten by the next chunk, leaving large regions of zeros in the output.

    The fix replaces to_zarr with da.store, which writes each dask chunk directly into
    its region of the zarr array without any internal rechunking.

    This test uses a conditional normalize_chunks mock that only intercepts the
    chunks="auto" call made by _write_dask_to_existing_zarr, injecting a sub-shard
    size that would corrupt data. All other normalize_chunks calls (e.g. from dask
    internals during rechunk) are forwarded to the real implementation. This lets
    the test confirm:
      1. to_zarr with injected sub-shard chunks corrupts the output (the old bug).
      2. dataset_to_zarr produces correct output under the same mock (the fix).
    """
    import warnings
    from unittest.mock import patch

    import zarr
    from dask.array.core import normalize_chunks as real_normalize_chunks

    shape = (100, 64, 64)
    dtype = np.dtype("uint16")
    full_data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    # Shard = 40 z-slices. We inject dask_write_chunks of 15 z-slices:
    # 15 < 40 and 15 does not divide 40, so each chunk write lands at a misaligned
    # offset inside the shard and the un-written remainder stays as zeros.
    outer_shards = (40, 64, 64)
    inner_chunks = (5, 64, 64)
    sub_shard_z = 15
    injected_chunks = (
        (sub_shard_z,) * (shape[0] // sub_shard_z) + (shape[0] % sub_shard_z,),
        (shape[1],),
        (shape[2],),
    )

    # Only intercept the specific chunks="auto" call made by _write_dask_to_existing_zarr.
    # All other normalize_chunks calls (used by dask internals such as rechunk) are
    # forwarded to the real implementation so dask's graph construction is not broken.
    def normalize_chunks_side_effect(chunks, *args, **kwargs):
        if chunks == "auto":
            return injected_chunks
        return real_normalize_chunks(chunks, *args, **kwargs)

    data = da.from_array(full_data, chunks=outer_shards)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---- part 1: confirm to_zarr corrupts data with injected sub-shard chunks ----
        old_path = Path(tmpdir) / "old.zarr"
        root = zarr.open_group(str(old_path), mode="w", zarr_format=3)
        arr = root.create_array(
            "0",
            shape=shape,
            chunks=inner_chunks,
            shards=outer_shards,
            dtype=dtype,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch(
                "dask.array.core.normalize_chunks",
                side_effect=normalize_chunks_side_effect,
            ):
                data.to_zarr(arr, compute=True)
        old_result = arr[:]
        assert not np.array_equal(old_result, full_data), (
            "Expected to_zarr to corrupt data with injected sub-shard chunks, "
            "but it did not — the regression test is no longer exercising the bug"
        )

        # ---- part 2: confirm dataset_to_zarr produces correct output under the same mock ----
        # If the writer regresses to to_zarr, the injected sub-shard chunks will corrupt
        # the output and the data equality assertion will fail.
        sub_shard_dataset, _ = _make_dataset(
            shape,
            chunks=(sub_shard_z, shape[1], shape[2]),
            data=full_data,
        )
        new_path = Path(tmpdir) / "new.zarr"
        with patch(
            "dask.array.core.normalize_chunks",
            side_effect=normalize_chunks_side_effect,
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                sub_shard_dataset,
                new_path,
                chunks=inner_chunks,
                shards=outer_shards,
            )
        read_dataset = anu_ctlab_io.Dataset.from_path(new_path)
        result = read_dataset.data.compute()

        assert result.shape == shape
        assert np.array_equal(result, full_data), (
            "Output contains incorrect values (possibly zeros) due to misaligned chunk writes"
        )


def test_no_false_warning_with_remainder_chunks(_make_dataset):
    """Test that no false-positive warning occurs when shards don't divide evenly into array size.

    This is a regression test for the Dask Zarr write warning bug that was fixed in:
    https://github.com/dask/dask/pull/12262

    The old Dask code would warn even when final chunks are at array boundaries (safe case).
    The fixed Dask code only warns when chunks are truly misaligned within the array.
    """
    # Use a shape where automatic chunking produces remainder chunks
    shape = (105, 50, 60)
    dataset, data = _make_dataset(
        shape,
        data=np.zeros(shape, dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "remainder_test.zarr"

        # Use automatic chunking with parameters that produce non-divisible shards
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunk_size_mb=0.5,  # Small chunks
                max_shard_size_mb=1.0,  # Small shards to trigger remainder
            )

            perf_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, PerformanceWarning)
            ]
            assert len(perf_warnings) == 0, (
                f"Unexpected performance warnings raised: {[str(w.message) for w in perf_warnings]}"
            )

            user_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            assert len(user_warnings) == 1
            assert (
                "chunk_size_mb, max_shard_size_mb are ignored when writing Zarr"
                in str(user_warnings[0].message)
            )

        # Verify the write succeeded and data is correct
        assert output_path.exists()
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.parametrize(
    ("shape", "chunks", "shards", "expected_chunks", "expected_shards"),
    [
        ((40, 65, 97), "auto", "auto", (32, 32, 32), (64, 96, 128)),
        ((1, 600, 700), "auto", "auto", (1, 256, 256), (1, 768, 768)),
        ((21, 200, 300), "auto", "auto", (21, 32, 32), (21, 224, 320)),
        ((60, 40, 50), (10, 0, 25), (30, 0, 0), (10, 40, 25), (30, 40, 50)),
        ((1000, 512, 512), "auto", "auto", (32, 32, 32), (512, 512, 512)),
        ((10, 20, 30), (5, 20, 30), (10, 20, 30), (5, 20, 30), (10, 20, 30)),
        ((40, 65, 97), "auto", None, (32, 32, 32), None),
        ((10, 20, 30), "auto", "auto", (10, 20, 30), (10, 20, 30)),
        ((10, 20, 30), (5, 0, 0), None, (5, 20, 30), None),
        ((40, 65, 97), (32, 32, 32), (32, 0, 0), (32, 32, 32), (32, 96, 128)),
    ],
)
def test_resolve_zarr_layout(
    shape,
    chunks,
    shards,
    expected_chunks,
    expected_shards,
):
    chunks, shards = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=shape,
        chunks=chunks,
        shards=shards,
    )

    assert chunks == expected_chunks
    assert shards == expected_shards


@pytest.mark.parametrize(
    ("elements", "expected_chunks"),
    [
        (32**3, (32, 32, 32)),
        (64**3, (64, 64, 64)),
        (64**3 - 1, (64, 64, 64)),
    ],
)
def test_integer_chunks_are_element_based(elements, expected_chunks):
    chunks, _ = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(1024, 1024, 1024),
        chunks=elements,
        shards=None,
    )

    assert chunks == expected_chunks


@pytest.mark.parametrize(
    ("elements", "expected_chunks"),
    [
        (32**2, (1, 32, 32)),
        (64**2, (1, 64, 64)),
        (64**2 - 1, (1, 64, 64)),
    ],
)
def test_integer_chunks_are_element_based_for_2d(elements, expected_chunks):
    chunks, _ = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(1, 1024, 1024),
        chunks=elements,
        shards=None,
    )

    assert chunks == expected_chunks


@pytest.mark.parametrize(
    ("shape", "expected_chunks", "expected_shards"),
    [
        ((1024, 1024, 1024), (32, 32, 32), (512, 512, 512)),
        ((1, 65536, 65536), (1, 256, 256), (1, 8192, 8192)),
    ],
)
def test_auto_uses_default_element_targets(shape, expected_chunks, expected_shards):
    chunks, shards = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=shape,
        chunks="auto",
        shards="auto",
    )

    assert chunks == expected_chunks
    assert shards == expected_shards


def test_integer_shards_are_chunk_multiples():
    chunks, shards = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(100, 100, 100),
        chunks=(10, 20, 25),
        shards=32**3,
    )

    assert chunks == (10, 20, 25)
    assert shards == (40, 40, 50)


def test_integer_shards_are_chunk_multiples_for_2d():
    chunks, shards = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(1, 100, 100),
        chunks=(1, 20, 25),
        shards=32**2,
    )

    assert chunks == (1, 20, 25)
    assert shards == (1, 40, 50)


def test_integer_shards_support_zero_chunk_sentinels():
    chunks, shards = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(100, 100, 100),
        chunks=(0, 20, 25),
        shards=32**3,
    )

    assert chunks == (32, 20, 25)
    assert shards == (32, 40, 50)


@pytest.mark.parametrize("elements", [0, -1, True])
def test_invalid_element_targets_raise(elements):
    with pytest.raises(ValueError, match="elements must be a positive integer"):
        anu_ctlab_io.zarr._writer._resolve_zarr_layout(
            shape=(100, 100, 100),
            chunks=elements,
            shards=None,
        )


def test_zero_chunk_and_shard_axes_expand_to_span_targets(_make_dataset):
    """A zero in shards spans the array; a zero in chunks spans the resolved shard axis."""
    import zarr

    dataset, _ = _make_dataset((60, 40, 50))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "shapes_with_zero.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            chunks=(10, 0, 25),
            shards=(30, 0, 0),
            ome_zarr_version=None,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.chunks == (10, 40, 25)
        assert array.shards == (30, 40, 50)


def test_default_zarr_chunking_is_element_based(_make_dataset):
    """Default Zarr writing uses element-based cubic chunks and shards."""
    import zarr

    dataset, _ = _make_dataset((40, 65, 97))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "default_layout.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.chunks == (32, 32, 32)
        assert array.shards == (64, 96, 128)


def test_write_with_auto_chunks(_make_dataset):
    """Passing chunks='auto' with shards=None uses the default unsharded chunk layout."""
    import zarr

    dataset, data = _make_dataset((10, 20, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "auto_chunks.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
            chunks="auto",
            shards=None,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.shards is None
        assert array.chunks == (10, 20, 32)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_write_with_auto_chunks_and_shards(_make_dataset):
    """Passing chunks='auto' and shards='auto' uses the default sharded layout."""
    import zarr

    shape = (100, 20, 60)
    dataset, data = _make_dataset(shape, chunks=(10, 20, 30))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "auto_chunks_and_shards.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
            chunks="auto",
            shards="auto",
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.chunks == (32, 20, 32)
        assert array.shards == (128, 20, 64)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_size_parameters_and_explicit_shapes(_make_dataset):
    """Deprecated size parameters warn and explicit shapes still win."""
    import zarr

    dataset, data = _make_dataset((60, 40, 50))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "explicit_shapes_with_ignored_sizes.zarr"
        with pytest.warns(
            UserWarning,
            match="chunk_size_mb, max_shard_size_mb are ignored when writing Zarr",
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                ome_zarr_version=None,
                chunks=(10, 40, 50),
                shards=(30, 40, 50),
                chunk_size_mb=0.005,
                max_shard_size_mb=1.0,
            )

        array = zarr.open_array(output_path, mode="r")
        assert array.chunks == (10, 40, 50)
        assert array.shards == (30, 40, 50)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


@pytest.mark.parametrize(
    "ome_zarr_version", [anu_ctlab_io.zarr.OMEZarrVersion.v05, None]
)
def test_write_slice_thumbnails(_make_dataset, tmp_path, ome_zarr_version):
    import zarr

    shape = (5, 7, 9)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data[0, 0, 0] = 1_000_000  # Outside all three middle slices.
    dataset, _ = _make_dataset(shape, data=data, dtype=np.float32, datatype=None)
    output_path = tmp_path / "images.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset, output_path, ome_zarr_version=ome_zarr_version
    )

    node = zarr.open(output_path, zarr_format=3)
    attrs = dict(node.attrs)
    assert any(
        convention["uuid"] == "49326c01-1180-4743-b15f-f7157038a6ab"
        for convention in attrs["zarr_conventions"]
    )
    assert len(attrs["thumbnails"]) == 3

    planes = {
        "xy": data[shape[0] // 2, :, :],
        "xz": data[:, shape[1] // 2, :],
        "yz": data[:, :, shape[2] // 2],
    }
    combined = np.concatenate([plane.ravel() for plane in planes.values()])
    lower, upper = np.percentile(combined, (1.0, 99.0))

    for plane_name, plane in planes.items():
        path = (
            output_path
            / "thumbnails"
            / f"middle_{plane_name}_{plane.shape[1]}x{plane.shape[0]}.jpg"
        )
        assert path.exists()
        with Image.open(path) as thumbnail:
            assert thumbnail.mode == "L"
            assert thumbnail.size == (plane.shape[1], plane.shape[0])

    metadata_by_path = {entry["path"]: entry for entry in attrs["thumbnails"]}
    xy_metadata = metadata_by_path[f"thumbnails/middle_xy_{shape[2]}x{shape[1]}.jpg"]
    assert xy_metadata["width"] == shape[2]
    assert xy_metadata["height"] == shape[1]
    assert xy_metadata["attributes"]["slice_axis"] == "z"
    assert xy_metadata["attributes"]["slice_index"] == shape[0] // 2
    assert xy_metadata["attributes"]["lower_value"] == pytest.approx(lower)
    assert xy_metadata["attributes"]["upper_value"] == pytest.approx(upper)
    assert xy_metadata["attributes"]["upper_value"] < 1_000_000


def test_write_full_slice_thumbnails(_make_dataset, tmp_path):
    shape = (3, 4, 600)
    dataset, _ = _make_dataset(shape, datatype=None)
    output_path = tmp_path / "resized.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, slice_thumbnails="full")

    with Image.open(output_path / "thumbnails/middle_xy_600x4.jpg") as full:
        assert full.size == (600, 4)
    with Image.open(output_path / "thumbnails/middle_xy_512x3.jpg") as thumbnail:
        assert thumbnail.size == (512, 3)


def test_write_full_slice_thumbnails_deduplicates_same_size(_make_dataset, tmp_path):
    dataset, _ = _make_dataset((3, 4, 5), datatype=None)
    output_path = tmp_path / "deduplicated.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, slice_thumbnails="full")

    import zarr

    root = zarr.open_group(output_path, zarr_format=3)
    assert len(root.attrs["thumbnails"]) == 3
    assert all(
        entry["attributes"]["roles"] == ["thumbnail", "full_resolution"]
        for entry in root.attrs["thumbnails"]
    )
    assert len(list((output_path / "thumbnails").glob("*.jpg"))) == 3


def test_write_thumbnails_reordered_dimensions(tmp_path):
    shape = (7, 9, 5)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    dataset = anu_ctlab_io.Dataset(
        da.from_array(data),
        dimension_names=("y", "x", "z"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
    )
    output_path = tmp_path / "reordered.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, ome_zarr_version=None)

    with Image.open(
        output_path / f"thumbnails/middle_xy_{shape[1]}x{shape[0]}.jpg"
    ) as image:
        assert image.size == (shape[1], shape[0])
    with Image.open(
        output_path / f"thumbnails/middle_xz_{shape[1]}x{shape[2]}.jpg"
    ) as image:
        assert image.size == (shape[1], shape[2])
    with Image.open(
        output_path / f"thumbnails/middle_yz_{shape[0]}x{shape[2]}.jpg"
    ) as image:
        assert image.size == (shape[0], shape[2])


def test_write_thumbnails_excludes_invalid_values(tmp_path):
    import zarr

    shape = (3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data[shape[0] // 2, 0, 0] = np.nan
    data[shape[0] // 2, 0, 1] = anu_ctlab_io.DataType.TOMO_FLOAT.mask_value
    dataset = anu_ctlab_io.Dataset(
        da.from_array(data),
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=anu_ctlab_io.DataType.TOMO_FLOAT,
    )
    output_path = tmp_path / "invalid.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, ome_zarr_version=None)

    array = zarr.open_array(output_path, zarr_format=3)
    metadata = {entry["path"]: entry for entry in array.attrs["thumbnails"]}[
        "thumbnails/middle_xy_5x4.jpg"
    ]
    assert metadata["attributes"]["upper_value"] < 100
    with Image.open(output_path / metadata["path"]) as diagnostic:
        pixels = np.asarray(diagnostic)
        assert pixels[0, 0] < 10
        assert pixels[0, 1] < 10


def test_write_thumbnails_all_invalid_is_black(tmp_path):
    data = np.full((3, 4, 5), np.nan, dtype=np.float32)
    dataset = anu_ctlab_io.Dataset(
        da.from_array(data),
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
    )
    output_path = tmp_path / "all_invalid.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, ome_zarr_version=None)

    with Image.open(output_path / "thumbnails/middle_xy_5x4.jpg") as image:
        assert np.asarray(image).max() < 10


def test_write_thumbnails_constant_data_is_black(_make_dataset, tmp_path):
    dataset, _ = _make_dataset(
        (3, 4, 5), data=np.full((3, 4, 5), 42, dtype=np.uint16), datatype=None
    )
    output_path = tmp_path / "constant.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, ome_zarr_version=None)

    with Image.open(output_path / "thumbnails/middle_xy_5x4.jpg") as image:
        assert np.asarray(image).max() < 10


def test_thumbnail_generation_requires_named_xyz_dimensions(tmp_path):
    dataset = anu_ctlab_io.Dataset(
        da.zeros((3, 4, 5)),
        dimension_names=("a", "b", "c"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
    )
    output_path = tmp_path / "invalid_dimensions.zarr"

    with pytest.raises(ValueError, match="exactly one named x, y, and z dimension"):
        anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path)
    assert not output_path.exists()


def test_disable_slice_thumbnail_generation(_make_dataset, tmp_path):
    dataset, _ = _make_dataset((3, 4, 5))
    output_path = tmp_path / "disabled.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset, output_path, slice_thumbnails=False, ome_zarr_version=None
    )

    import zarr

    array = zarr.open_array(output_path, zarr_format=3)
    assert "zarr_conventions" not in array.attrs
    assert "thumbnails" not in array.attrs
    assert not (output_path / "thumbnails").exists()


def test_invalid_slice_thumbnail_mode(_make_dataset, tmp_path):
    dataset, _ = _make_dataset((3, 4, 5))

    with pytest.raises(ValueError, match="slice_thumbnails must be"):
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset, tmp_path / "invalid.zarr", slice_thumbnails="invalid"
        )
