from __future__ import annotations

import numpy as np

from openamundsen_da.util.storage_policy import percent_to_uint8_nodata


def test_percent_to_uint8_nodata_encodes_percent_raster() -> None:
    values = np.array([[0.2, 49.6, 100.0], [255.0, np.nan, 120.0]], dtype=np.float32)

    out = percent_to_uint8_nodata(values, nodata_value=255.0)

    assert out.dtype == np.dtype("uint8")
    assert out.tolist() == [
        [0, 50, 100],
        [255, 255, 100],
    ]
