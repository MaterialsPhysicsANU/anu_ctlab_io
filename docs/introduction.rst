anu_ctlab_io package
--------------------
python I/O for the ANU CTLab array storage format(s).

Introduction
============

This package loads data provided in the specific NetCDF and Zarr formats produced by the ANU CTLab.
The intention is that the consumers of the data we produce should be able to load that data and then
work with that directly in standard scientific python workflows, rather than needing to use the
pre-existing MANGO toolchain.

The specific netcdf and zarr loading functionality is contained in extras, so you will
generally want to install this package as either ``anu_ctlab_io[netcdf]`` or ``anu_ctlab_io[zarr]``
depending on the format of data you want to read.
The ``all`` extra enables all supported formats.

Examples
========
.. code-block :: python3

    import anu_ctlab_io
    from dask_image import ndfilters

    dataset = anu_ctlab_io.Dataset.from_path(<path-to-your-data>)
    blurred = ndfilters.gaussian_filter(dataset.data, sigma=3)
    print(blurred.mean().compute())
