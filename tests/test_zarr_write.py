import json
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    from dask.array.core import PerformanceWarning

    import anu_ctlab_io.zarr
    from anu_ctlab_io.zarr import OMEZarrVersion
except ImportError:
    pytest.skip("Requires 'zarr' extra", allow_module_level=True)

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False

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


def test_default_ome_zarr_writes_multiscale_levels(_make_dataset, tmp_path):
    """Default OME-Zarr output writes a multiscale pyramid when chunks allow it."""
    import zarr

    shape = (16, 16, 16)
    dataset, data = _make_dataset(shape, chunks=(4, 4, 4))
    output_path = tmp_path / "multiscale.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
    )

    root = zarr.open_group(output_path, mode="r")
    assert sorted(root.array_keys()) == ["0", "1", "2"]

    read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
    assert read_dataset.data.shape == shape
    assert np.array_equal(read_dataset.data.compute(), data.compute())


def test_multiscale_stops_when_array_fits_in_level_0_chunk(_make_dataset, tmp_path):
    """Unsharded pyramids stop once the current level fits in one level-0 chunk."""
    import zarr

    dataset, _ = _make_dataset((4, 4, 4), chunks=(4, 4, 4))
    output_path = tmp_path / "fits_in_chunk.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
    )

    assert sorted(zarr.open_group(output_path, mode="r").array_keys()) == ["0"]


def test_multiscale_stops_when_array_fits_in_level_0_subchunk(_make_dataset, tmp_path):
    """Sharded pyramids stop once the current level fits in one level-0 subchunk."""
    import zarr

    dataset, _ = _make_dataset((2, 2, 2), chunks=(2, 2, 2))
    output_path = tmp_path / "fits_in_subchunk.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks=(2, 2, 2),
        shards=(2, 2, 2),
    )

    assert sorted(zarr.open_group(output_path, mode="r").array_keys()) == ["0"]


def test_multiscale_stop_threshold_uses_level_0_storage_unit(_make_dataset, tmp_path):
    """The stop threshold does not shrink with each pyramid level's layout."""
    import zarr

    dataset, _ = _make_dataset((16, 16, 16), chunks=(8, 8, 8))
    output_path = tmp_path / "level_0_threshold.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
    )

    root = zarr.open_group(output_path, mode="r")
    assert sorted(root.array_keys()) == ["0", "1"]
    assert zarr.open_array(output_path / "0", mode="r").chunks == (8, 8, 8)
    assert zarr.open_array(output_path / "1", mode="r").shape == (8, 8, 8)


def test_ome_zarr_multiscale_metadata_transforms(_make_dataset, tmp_path):
    """Level transforms use 2x scales and half-voxel translations."""
    import zarr

    dataset, _ = _make_dataset((16, 16, 16), chunks=(4, 4, 4), voxel_size=(2, 3, 4))
    output_path = tmp_path / "metadata.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
    )

    ome = zarr.open_group(output_path, mode="r").attrs["ome"]
    multiscale = ome["multiscales"][0]
    assert multiscale["coordinateTransformations"] == [
        {"type": "scale", "scale": [2.0, 3.0, 4.0]}
    ]

    datasets = multiscale["datasets"]
    assert [dataset["path"] for dataset in datasets] == ["0", "1", "2"]
    assert datasets[0]["coordinateTransformations"] == [
        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
    ]
    assert datasets[1]["coordinateTransformations"] == [
        {"type": "scale", "scale": [2.0, 2.0, 2.0]},
        {"type": "translation", "translation": [0.5, 0.5, 0.5]},
    ]
    assert datasets[2]["coordinateTransformations"] == [
        {"type": "scale", "scale": [4.0, 4.0, 4.0]},
        {"type": "translation", "translation": [1.5, 1.5, 1.5]},
    ]


def test_plain_zarr_output_remains_single_scale(_make_dataset, tmp_path):
    """Plain Zarr output is still a single array."""
    import zarr

    dataset, _ = _make_dataset((8, 8, 8), chunks=(4, 4, 4))
    output_path = tmp_path / "plain.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        ome_zarr_version=None,
    )

    assert isinstance(zarr.open(output_path, mode="r"), zarr.Array)


def test_multiscale_stops_on_lower_level_layout_incompatibility(
    _make_dataset, tmp_path
):
    """Level >0 layout incompatibility stops the pyramid without metadata gaps."""
    import zarr

    dataset, _ = _make_dataset((8, 8, 8), chunks=(4, 4, 4))
    output_path = tmp_path / "stopped.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks=(2, 2, 2),
        shards=None,
    )

    root = zarr.open_group(output_path, mode="r")
    assert sorted(root.array_keys()) == ["0", "1"]
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    assert [dataset["path"] for dataset in datasets] == ["0", "1"]


def test_multiscale_level_zero_layout_incompatibility_raises(_make_dataset, tmp_path):
    """Level 0 keeps raising for explicit layouts incompatible with the input grid."""
    dataset, _ = _make_dataset((8, 8, 8), chunks=(4, 4, 4))

    with pytest.raises(ValueError, match="evenly divide"):
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            tmp_path / "invalid_level0.zarr",
            chunks=(3, 3, 3),
            shards=None,
        )


def test_multiscale_write_uses_source_chunk_tasks(_make_dataset, tmp_path):
    """Multiscale writes do not rechunk or build separate Dask downsample arrays."""
    from unittest.mock import patch

    dataset, _ = _make_dataset((8, 8, 8), chunks=(4, 4, 4))

    with (
        patch.object(da.Array, "rechunk", side_effect=AssertionError),
        patch.object(da, "coarsen", side_effect=AssertionError),
        patch.object(da, "store", side_effect=AssertionError),
    ):
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            tmp_path / "no_rechunk.zarr",
            chunks="auto",
            shards=None,
        )


def test_multiscale_allows_odd_chunk_sizes_by_rounding_up(_make_dataset, tmp_path):
    """Odd source chunks can still downsample; each chunk rounds up locally."""
    import zarr

    data = np.arange(10**3, dtype=np.uint16).reshape((10, 10, 10))
    dataset, _ = _make_dataset(data.shape, chunks=(5, 5, 5), data=data)
    output_path = tmp_path / "odd_chunks.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
    )

    root = zarr.open_group(output_path, mode="r")
    assert sorted(root.array_keys()) == ["0", "1", "2"]

    level_1 = zarr.open_array(output_path / "1", mode="r")
    level_2 = zarr.open_array(output_path / "2", mode="r")
    level_1_chunks = ((3, 3), (3, 3), (3, 3))
    level_2_chunks = ((2, 2), (2, 2), (2, 2))
    assert level_1.shape == (6, 6, 6)
    assert level_1.chunks == (3, 3, 3)
    assert level_2.shape == (4, 4, 4)
    assert level_2.chunks == (2, 2, 2)

    def starts(chunks):
        axis_starts = []
        for axis_chunks in chunks:
            offset = 0
            current_axis_starts = []
            for chunk in axis_chunks:
                current_axis_starts.append(offset)
                offset += chunk
            axis_starts.append(tuple(current_axis_starts))
        return tuple(axis_starts)

    def chunkwise_strided(array, input_chunks, output_chunks):
        output = np.empty(tuple(sum(axis) for axis in output_chunks), dtype=array.dtype)
        input_starts = starts(input_chunks)
        output_starts = starts(output_chunks)
        for chunk_index in np.ndindex(tuple(len(axis) for axis in input_chunks)):
            input_slices = tuple(
                slice(
                    input_starts[axis][index],
                    input_starts[axis][index] + input_chunks[axis][index],
                    2,
                )
                for axis, index in enumerate(chunk_index)
            )
            output_slices = tuple(
                slice(
                    output_starts[axis][index],
                    output_starts[axis][index] + output_chunks[axis][index],
                )
                for axis, index in enumerate(chunk_index)
            )
            output[output_slices] = array[input_slices]
        return output

    expected_level_1 = chunkwise_strided(data, ((5, 5), (5, 5), (5, 5)), level_1_chunks)
    expected_level_2 = chunkwise_strided(
        expected_level_1, level_1_chunks, level_2_chunks
    )
    assert np.array_equal(level_1[:], expected_level_1)
    assert np.array_equal(level_2[:], expected_level_2)


def test_mean_downsampling_preserves_dtype_and_values(_make_dataset, tmp_path):
    """Mean downsampling rounds and casts integer levels back to the source dtype."""
    import zarr

    data = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
    dataset, _ = _make_dataset(data.shape, chunks=(2, 2, 2), data=data)
    output_path = tmp_path / "mean.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks=(1, 1, 1),
        shards=None,
        downsample_method="mean",
    )

    level = zarr.open_array(output_path / "1", mode="r")
    assert level.dtype == np.dtype("uint16")
    assert level[:] == np.array([[[4]]], dtype=np.uint16)


def test_default_downsampling_is_strided_for_all_datatypes(_make_dataset, tmp_path):
    """Default downsampling takes every second voxel regardless of datatype."""
    import zarr

    data = np.array([7, 2, 2, 1, 2, 1, 1, 2], dtype=np.uint8).reshape(2, 2, 2)
    dataset, _ = _make_dataset(
        data.shape,
        chunks=(2, 2, 2),
        data=data,
        datatype=anu_ctlab_io.DataType.SEGMENTED,
    )
    output_path = tmp_path / "strided.zarr"

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks=(1, 1, 1),
        shards=None,
    )

    assert zarr.open_array(output_path / "1", mode="r")[:] == np.array(
        [[[7]]], dtype=np.uint8
    )


@pytest.mark.parametrize(
    ("datatype", "method", "expected"),
    [
        (anu_ctlab_io.DataType.SEGMENTED, "mean", 2),
        (anu_ctlab_io.DataType.TOMO, "mode", 1),
    ],
)
def test_mean_and_mode_downsample_methods(
    _make_dataset, tmp_path, datatype, method, expected
):
    """Explicit mean and mode downsampling choices are available."""
    import zarr

    dtype = np.uint8 if datatype == anu_ctlab_io.DataType.SEGMENTED else np.uint16
    data = np.array([1, 3, 3, 1, 3, 1, 1, 3], dtype=dtype).reshape(2, 2, 2)
    dataset, _ = _make_dataset(
        data.shape,
        chunks=(2, 2, 2),
        data=data,
        datatype=datatype,
    )

    anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        tmp_path / f"{method}.zarr",
        chunks=(1, 1, 1),
        shards=None,
        downsample_method=method,
    )

    assert zarr.open_array(tmp_path / f"{method}.zarr" / "1", mode="r")[:] == np.array(
        [[[expected]]], dtype=dtype
    )


def test_multiscale_compute_false_returns_one_delayed(_make_dataset, tmp_path):
    """compute=False returns a single delayed object that writes every level."""
    import zarr

    dataset, _ = _make_dataset((4, 4, 4), chunks=(2, 2, 2))
    output_path = tmp_path / "delayed.zarr"

    result = anu_ctlab_io.zarr.dataset_to_zarr(
        dataset,
        output_path,
        chunks="auto",
        shards=None,
        compute=False,
    )

    assert result is not None
    result.compute()
    assert sorted(zarr.open_group(output_path, mode="r").array_keys()) == ["0", "1"]


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
            multiscale=False,
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
            multiscale=False,
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
            multiscale=False,
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
                multiscale=False,
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
    ("shape", "chunks", "shards", "expected_chunks", "expected_subchunks"),
    [
        ((40, 65, 97), "auto", "auto", (64, 96, 128), (32, 32, 32)),
        ((1, 600, 700), "auto", "auto", (1, 768, 768), (1, 256, 256)),
        ((21, 200, 300), "auto", "auto", (21, 224, 320), (21, 32, 32)),
        ((60, 40, 50), (10, 0, 25), (30, 0, 0), (30, 40, 50), (10, 40, 25)),
        ((1000, 512, 512), "auto", "auto", (512, 512, 512), (32, 32, 32)),
        ((10, 20, 30), (5, 20, 30), (10, 20, 30), (10, 20, 30), (5, 20, 30)),
        ((40, 65, 97), "auto", None, (40, 65, 97), None),
        ((10, 20, 30), "auto", "auto", (10, 20, 30), (10, 20, 30)),
        ((10, 20, 30), (5, 0, 0), None, (5, 20, 30), None),
        ((40, 65, 97), (32, 32, 32), (32, 0, 0), (32, 96, 128), (32, 32, 32)),
    ],
)
def test_resolve_zarr_layout(
    shape,
    chunks,
    shards,
    expected_chunks,
    expected_subchunks,
):
    resolved_chunks, resolved_subchunks = (
        anu_ctlab_io.zarr._writer._resolve_zarr_layout(
            shape=shape,
            chunks=shards if shards is not None else chunks,
            subchunks=chunks if shards is not None else None,
        )
    )

    assert resolved_chunks == expected_chunks
    assert resolved_subchunks == expected_subchunks


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
        subchunks=None,
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
        subchunks=None,
    )

    assert chunks == expected_chunks


@pytest.mark.parametrize(
    ("shape", "expected_chunks", "expected_subchunks"),
    [
        ((1024, 1024, 1024), (512, 512, 512), (32, 32, 32)),
        ((1, 65536, 65536), (1, 8192, 8192), (1, 256, 256)),
    ],
)
def test_auto_uses_default_element_targets(shape, expected_chunks, expected_subchunks):
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=shape,
        chunks="auto",
        subchunks="auto",
    )

    assert chunks == expected_chunks
    assert subchunks == expected_subchunks


def test_integer_shards_are_chunk_multiples():
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(100, 100, 100),
        chunks=32**3,
        subchunks=(10, 20, 25),
    )

    assert chunks == (40, 40, 50)
    assert subchunks == (10, 20, 25)


def test_integer_shards_are_chunk_multiples_for_2d():
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(1, 100, 100),
        chunks=32**2,
        subchunks=(1, 20, 25),
    )

    assert chunks == (1, 40, 50)
    assert subchunks == (1, 20, 25)


def test_integer_shards_support_zero_chunk_sentinels():
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(100, 100, 100),
        chunks=32**3,
        subchunks=(0, 20, 25),
    )

    assert chunks == (32, 40, 50)
    assert subchunks == (32, 20, 25)


def test_input_aligned_integer_chunk_target_uses_dask_chunk_divisors():
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(40, 50, 60),
        chunks="auto",
        subchunks=32**3,
        aligned_chunks=((20, 20), (50,), (60,)),
    )

    assert chunks == (20, 50, 60)
    assert subchunks == (20, 25, 30)


def test_input_aligned_integer_shard_target_allows_multiple_shards_per_dask_chunk():
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=(32, 32, 32),
        chunks=8**3,
        subchunks=4**3,
        aligned_chunks=((16, 16), (16, 16), (16, 16)),
    )

    assert chunks == (8, 8, 8)
    assert subchunks == (4, 4, 4)


@pytest.mark.parametrize(
    ("shape", "aligned_chunks", "expected_chunks", "expected_subchunks"),
    [
        # One chunk on XY, subchunk doesn't evenly divide. Zarr chunk expands to accommodate.
        ((32, 2914, 2914), ((32,), (2914,), (2914,)), (32, 2944, 2944), (32, 32, 32)),
        ((32, 997, 997), ((32,), (997,), (997,)), (32, 1024, 1024), (32, 32, 32)),
        # Multiple chunks on XY with prime number sizes. Mango cannot create this, but this checks rechunking still doesn't occur even though the subchunk size is highly suboptimal
        (
            (32, 997 * 2, 997 * 2),
            ((32,), (997, 997), (997, 997)),
            (32, 997, 997),
            (32, 1, 1),
        ),
    ],
)
def test_resolve_zarr_layout_edge_and_prime(
    shape, aligned_chunks, expected_chunks, expected_subchunks
):
    chunks, subchunks = anu_ctlab_io.zarr._writer._resolve_zarr_layout(
        shape=shape,
        chunks="auto",
        subchunks="auto",
        aligned_chunks=aligned_chunks,
    )

    assert chunks == expected_chunks
    assert subchunks == expected_subchunks


def test_normalize_explicit_shapes_uses_internal_chunk_subchunk_order():
    chunks, subchunks = anu_ctlab_io.zarr._writer._normalize_explicit_shapes(
        shape=(60, 40, 50),
        chunks=(30, 0, 0),
        subchunks=(10, 0, 25),
    )

    assert chunks == (30, 40, 50)
    assert subchunks == (10, 40, 25)


@pytest.mark.parametrize("elements", [0, -1, True])
def test_invalid_element_targets_raise(elements):
    with pytest.raises(ValueError, match="elements must be a positive integer"):
        anu_ctlab_io.zarr._writer._resolve_zarr_layout(
            shape=(100, 100, 100),
            chunks=elements,
            subchunks=None,
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
        assert array.chunks == (10, 20, 60)

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


def test_input_aligned_chunks_match_dask_chunks_for_sharded_write(_make_dataset):
    """Input-aligned sharded writes use dask chunks as shards and divisor chunks."""
    import zarr

    shape = (40, 50, 60)
    dataset, data = _make_dataset(shape, chunks=(20, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "input_aligned_sharded.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
            input_aligned_chunks=True,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.shards == (20, 50, 60)
        assert array.chunks == (20, 25, 30)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_input_aligned_chunks_match_dask_chunks_for_unsharded_write(_make_dataset):
    """Input-aligned unsharded writes use dask chunks as Zarr chunks."""
    import zarr

    shape = (40, 50, 60)
    dataset, data = _make_dataset(shape, chunks=(20, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "input_aligned_unsharded.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
            shards=None,
            input_aligned_chunks=True,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.shards is None
        assert array.chunks == (20, 50, 60)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_input_aligned_chunks_allow_array_edge_remainders(_make_dataset):
    """Final dask chunks may be smaller at the array edge."""
    import zarr

    shape = (45, 55, 65)
    dataset, data = _make_dataset(shape, chunks=(20, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "input_aligned_edges.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            ome_zarr_version=None,
            input_aligned_chunks=True,
        )

        array = zarr.open_array(output_path, mode="r")
        assert array.shards == (20, 50, 60)
        assert array.chunks == (20, 25, 30)

        loaded_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert np.array_equal(loaded_dataset.data.compute(), data.compute())


def test_input_aligned_chunks_reject_irregular_internal_dask_chunks():
    """Only the final chunk on each axis may be a smaller remainder."""
    data = da.zeros((50, 50, 60), chunks=((20, 10, 20), (50,), (60,)))
    dataset = anu_ctlab_io.Dataset(
        data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="regular aligned chunks"):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                Path(tmpdir) / "irregular.zarr",
                ome_zarr_version=None,
                input_aligned_chunks=True,
            )


def test_input_aligned_chunks_validates_explicit_tuple_layout(_make_dataset):
    """Explicit tuple layouts must be compatible with the dask chunk grid."""
    dataset, _ = _make_dataset((40, 50, 60), chunks=(20, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="evenly divide"):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                Path(tmpdir) / "misaligned.zarr",
                ome_zarr_version=None,
                chunks=(7, 25, 30),
                shards=(20, 50, 60),
                input_aligned_chunks=True,
            )


def test_input_aligned_chunks_skips_writer_rechunk(_make_dataset):
    """Validated input-aligned writes do not rechunk before storing."""
    from unittest.mock import patch

    dataset, _ = _make_dataset((40, 50, 60), chunks=(20, 50, 60))

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(da.Array, "rechunk", side_effect=AssertionError):
            anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                Path(tmpdir) / "no_rechunk.zarr",
                ome_zarr_version=None,
                slice_thumbnails=False,
                input_aligned_chunks=True,
            )


@pytest.mark.parametrize(
    ("shape", "chunks"),
    [
        ((68, 2914, 2914), ((32,) * 2 + (4,), (2914,), (2914,))),
        ((68, 997, 997), ((32,) * 2 + (4,), (997,), (997,))),
        ((68, 997 * 2, 997 * 2), ((32,) * 2 + (4,), (997, 997), (997, 997))),
    ],
)
def test_input_aligned_large_array_writes_skip_rechunk(shape, chunks):
    """Large input-aligned writes build the store graph without rechunking."""
    from unittest.mock import patch

    data = da.zeros(shape, chunks=chunks, dtype=np.uint16)
    dataset = anu_ctlab_io.Dataset(
        data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(da.Array, "rechunk", side_effect=AssertionError):
            result = anu_ctlab_io.zarr.dataset_to_zarr(
                dataset,
                Path(tmpdir) / "no_large_rechunk.zarr",
                ome_zarr_version=None,
                input_aligned_chunks=True,
                compute=False,
            )

    assert result is not None


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


def test_read_plain_zarr_array_without_mango():
    """Test reading a plain Zarr v3 array without mango attributes."""
    import zarr

    shape = (10, 20, 30)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "plain.zarr"
        # Create a plain Zarr v3 array without any mango metadata
        za = zarr.open_array(
            output_path,
            mode="w",
            shape=shape,
            dtype=np.float32,
            zarr_format=3,
            dimension_names=("z", "y", "x"),
        )
        za[:] = data

        # Read it back via dataset_from_zarr
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify defaults for missing mango attributes
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data)
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.VOXEL
        assert read_dataset.voxel_size == (1.0, 1.0, 1.0)
        assert read_dataset._datatype is None
        assert read_dataset.history == {}
        assert read_dataset._dataset_id is None


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_plain_zarr_to_netcdf_requires_datatype():
    """Test that converting a plain Zarr array to NetCDF requires explicit datatype."""
    import zarr

    shape = (10, 20, 30)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "plain.zarr"
        # Create a plain Zarr v3 array without mango metadata
        za = zarr.open_array(
            zarr_path,
            mode="w",
            shape=shape,
            dtype=np.uint16,
            zarr_format=3,
            dimension_names=("z", "y", "x"),
        )
        za[:] = data

        # Load the plain Zarr array
        dataset = anu_ctlab_io.Dataset.from_path(zarr_path)
        assert dataset._datatype is None

        # Attempt to write to NetCDF without datatype should fail
        netcdf_path = Path(tmpdir) / "output.nc"
        with pytest.raises(ValueError, match="datatype must be provided"):
            dataset.to_path(netcdf_path, filetype="NetCDF")

        # Writing with explicit datatype should succeed
        netcdf_path = Path(tmpdir) / "tomo_output.nc"
        dataset.to_path(netcdf_path, filetype="NetCDF", datatype="tomo")
        assert netcdf_path.exists()

        # Verify the written NetCDF can be read back
        read_dataset = anu_ctlab_io.Dataset.from_path(netcdf_path)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data)
