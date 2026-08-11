from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from openamundsen_da.methods.viz.maps.panel_renderers import (
    _prior_wet_fraction_array,
    _single_domain_scf_model_probability_array,
)
from openamundsen_da.util.map_support import (
    load_map_support_field,
    validate_map_support,
    write_map_support,
)


def test_retained_map_support_round_trips_and_feeds_render_sources(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    date = pd.Timestamp("2023-03-01")
    scf = np.asarray([[0.0, 0.5], [1.0, np.nan]], dtype=float)
    wet = np.asarray([[0.2, 0.4], [0.8, np.nan]], dtype=float)
    output = write_map_support(
        project,
        dates=[date],
        fields={
            "scf_prior_probability": [scf],
            "wet_snow_prior_probability": [wet],
        },
    )
    assert output.is_file()
    assert validate_map_support(
        project,
        dates=[date],
        fields={"scf_prior_probability", "wet_snow_prior_probability"},
    ) == output
    np.testing.assert_allclose(
        load_map_support_field(project, date=date, field="scf_prior_probability"),
        scf,
        equal_nan=True,
    )

    context = SimpleNamespace(project_dir=project, roi_mask=np.ones((2, 2), dtype=bool))
    np.testing.assert_allclose(
        _single_domain_scf_model_probability_array(
            context=context,
            source="prior_probability",
            date=date,
            derived_cache={},
        ),
        scf,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _prior_wet_fraction_array(context=context, date=date, derived_cache={}),
        wet,
        equal_nan=True,
    )
