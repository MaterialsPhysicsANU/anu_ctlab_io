import pytest

from anu_ctlab_io import VoxelUnit


def test_from_str():
    assert VoxelUnit.from_str("m") == VoxelUnit.M
    assert VoxelUnit.from_str("um") == VoxelUnit.UM
    assert VoxelUnit.from_str("µm") == VoxelUnit.UM
    assert VoxelUnit.from_str("voxel") == VoxelUnit.VOXEL
    with pytest.raises(ValueError):
        VoxelUnit.from_str("unknown")
