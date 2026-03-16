"""Raw binary format support for :any:`anu_ctlab_io`.

Provides write-only support for headerless raw binary files (C-order, little-endian).
No optional extras are required for this format.
"""

from anu_ctlab_io.raw._writer import dataset_to_raw

__all__ = ["dataset_to_raw"]
