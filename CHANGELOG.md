# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation using Sphinx
- justfile to build docs and run tests
- Support for units to `VoxelUnit` to match MANGO support: `angstrom`, `centimeters`, `voxels`
- Tests for `DataType.from_basename`

### Changed

- Replace README.md with README.rst
- Improve typehinting
- Privatise unintentionally public modules: `dataset`, `datatype`, `parse_history`, `version`, `voxel_properties`
- Privatise objects not intended to be public:
  - `_datatype.DataTypeProperties`
  - `_datatype.DATATYPE_PROPERTIES`
  - `netcdf._read_netcdf`
- Rename `_datatype.storage_dtypes` to `_datatype.StorageDType` (PEP8)
- Capitalise `DataType` members (PEP8)
- Capitalise `VoxelUnit` members (PEP8)
- Move zarr and netcdf dependencies to extras
- Improve `tox` testing configuration, including testing with and without extras
- Update changelog formatting to better match the Keep a Changelog format.

### Removed

- Remove features which were deprecated in 0.2.0:
  - NetCDFDataset
  - Xarray dataset support
- Remove `UnknownVoxelUnitException`

## [0.2.0] - 2025-09-07

### Added

- Support for reading ANU CTLab Zarr files
- Direct access to the data loaded as Dask arrays

### Changed

- Interface changed to use the Dataset class for both Zarr and NetCDF files
- Xarray output deprecated (to be removed in 1.0.0)

## [0.0.1]

### Added

- Support for reading ANU CTLab NetCDF files
- Support for outputting Xarray data

[unreleased]: https://github.com/MaterialsPhysicsANU/anu_ctlab_io/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/MaterialsPhysicsANU/anu_ctlab_io/releases/tag/v0.2.0
