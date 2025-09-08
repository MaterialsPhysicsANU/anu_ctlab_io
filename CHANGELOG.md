# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0

### Added

- Support for reading ANU CTLab Zarr files
- Direct access to the data loaded as Dask arrays

### Changed

- Interface changed to use the Dataset class for both Zarr and NetCDF files
- Xarray output deprecated (to be removed in 1.0.0)

## 0.0.1

### Added

- Support for reading ANU CTLab NetCDF files
- Support for outputting Xarray data
