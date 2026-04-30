import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    from dask.array.core import PerformanceWarning

    import anu_ctlab_io.zarr

    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False

import anu_ctlab_io

pytestmark = pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")


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
            ome_zarr_version=anu_ctlab_io.zarr.OMEZarrVersion.v05,
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


def test_write_zarr_deprecated_size_args_warn_and_use_default_layout(_make_dataset):
    """Deprecated size-based parameters warn and fall back to the default layout."""
    import zarr

    shape = (100, 20, 60)
    dataset, data = _make_dataset(shape, chunks=(10, 20, 30))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_sharded.zarr"

        with pytest.warns(
            UserWarning,
            match="chunk_size_mb, max_shard_size_mb is ignored when writing Zarr",
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                dataset_id="test_sharded_dataset",
                max_shard_size_mb=0.02,
                chunk_size_mb=0.005,
            )

        array = zarr.open_array(output_path / "0", mode="r")
        assert array.chunks == (32, 20, 32)
        assert array.shards == (32, 20, 64)

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
            ome_zarr_version=anu_ctlab_io.zarr.OMEZarrVersion.v05,
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


@pytest.mark.xfail(
    reason="This was the behaviour in <=1.2.2, but this library now ignores size parameters"
)
def test_error_both_shapes_and_sizes(_make_dataset):
    """Test that providing both shapes and sizes raises ValueError."""
    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.raises(
            ValueError, match="Cannot specify both chunks/shards and chunk_size_mb"
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks=(5, 20, 30),
                shards=(10, 20, 30),
                chunk_size_mb=5.0,
            )


@pytest.mark.xfail(
    reason="This was the behaviour in <=1.2.2, but this library now ignores size parameters"
)
def test_error_shapes_and_max_shard_size(_make_dataset):
    """Test that providing shapes with max_shard_size_mb raises ValueError."""
    dataset, _ = _make_dataset(
        (10, 20, 30),
        data=np.zeros((10, 20, 30), dtype=np.uint16),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"

        with pytest.raises(
            ValueError, match="Cannot specify both chunks/shards and chunk_size_mb"
        ):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                output_path,
                chunks=(5, 20, 30),
                shards=(10, 20, 30),
                max_shard_size_mb=100.0,
            )


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

            # Verify the deprecated size warning is the only warning emitted.
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
                "chunk_size_mb, max_shard_size_mb is ignored when writing Zarr"
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
        ((40, 65, 97), "auto", "auto", (32, 32, 32), (32, 96, 128)),
        ((1, 600, 700), "auto", "auto", (1, 256, 256), (1, 768, 768)),
        ((21, 200, 300), "auto", "auto", (21, 32, 32), (21, 224, 320)),
        ((60, 40, 50), (10, 0, 25), (30, 0, 0), (10, 40, 25), (30, 40, 50)),
        ((1000, 512, 512), "auto", "auto", (32, 32, 32), (32, 512, 512)),
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


def test_default_zarr_chunking_spans_xy_domain(_make_dataset):
    """Default Zarr writing uses 32^3 subchunks with full-domain XY chunks (shards)."""
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
        assert array.shards == (32, 96, 128)


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
        assert array.shards == (32, 20, 64)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_size_parameters_warn_and_explicit_shapes_still_win(_make_dataset):
    """Deprecated size parameters are ignored when explicit shapes are also provided."""
    import zarr

    dataset, data = _make_dataset((60, 40, 50))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "explicit_shapes_with_ignored_sizes.zarr"
        with pytest.warns(
            UserWarning,
            match="chunk_size_mb, max_shard_size_mb is ignored when writing Zarr",
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
