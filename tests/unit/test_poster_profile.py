from __future__ import annotations

import base64
from pathlib import Path

import pytest
import ruamel.yaml

from openamundsen_da.methods.viz.common import save_figure_png
from openamundsen_da.methods.viz.common import PosterRenderStyle, PosterTypography
from openamundsen_da.methods.viz.maps.config import LayoutSpec, MapPanelSpec, MapRecipe
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR
import openamundsen_da.methods.viz.maps.layout as map_layout
import openamundsen_da.methods.viz.maps.panel_renderers as panel_renderers_module
from openamundsen_da.methods.viz.maps.render import _scaled_map_style
import openamundsen_da.methods.viz.poster as poster_mod
from openamundsen_da.methods.viz.poster import (
    PosterDaEventsConfig,
    PosterSetupOverviewConfig,
    load_poster_config,
    measure_poster_svg_targets,
    poster_da_event_recipe,
    poster_map_recipes,
    poster_setup_overview_recipe,
    render_poster_profile,
    render_poster_maps,
    write_poster_target_sizes,
)


def _panel(kind: str, row: int, col: int, *, source: str | None = None, title: str | None = None) -> MapPanelSpec:
    return MapPanelSpec(kind=kind, row=row, col=col, source=source, title=title or f"{kind}-{col}")


def test_poster_map_typography_increases_vertical_colorbar_tick_padding() -> None:
    original = map_layout._COLORBAR_TICK_PAD

    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(14.2, 12.0, 10.0))):
        assert map_layout._COLORBAR_TICK_PAD == pytest.approx(8.0)

    assert map_layout._COLORBAR_TICK_PAD == original


def test_poster_map_typography_applies_title_pad_override() -> None:
    assert map_layout._TITLE_PAD_OVERRIDE is None
    assert map_layout._Y_TICK_LABEL_STRIDE_MULTIPLIER == 1

    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        assert map_layout._TITLE_PAD_OVERRIDE == pytest.approx(8.0)
        assert map_layout._Y_TICK_LABEL_STRIDE_MULTIPLIER == 2
        assert map_layout._POSTER_VERTICAL_COLORBAR_TICKS_ENABLED is True
        assert map_layout._POSTER_VERTICAL_COLORBAR_UNIT_HEADER_ENABLED is True

    assert map_layout._TITLE_PAD_OVERRIDE is None
    assert map_layout._Y_TICK_LABEL_STRIDE_MULTIPLIER == 1
    assert map_layout._POSTER_VERTICAL_COLORBAR_TICKS_ENABLED is False
    assert map_layout._POSTER_VERTICAL_COLORBAR_UNIT_HEADER_ENABLED is False


def test_poster_map_typography_thins_y_coordinate_labels_only() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (0.0, 10_000.0, 0.0, 10_000.0)
    fig, (default_ax, poster_ax) = plt.subplots(ncols=2, figsize=(6.0, 3.0))
    map_layout.apply_map_axis_style(default_ax, extent, title=None, show_grid=True)
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        map_layout.apply_map_axis_style(poster_ax, extent, title=None, show_grid=True)

    default_y_labels = [label.get_text() for label in default_ax.get_yticklabels() if label.get_text()]
    poster_y_labels = [label.get_text() for label in poster_ax.get_yticklabels() if label.get_text()]
    default_x_labels = [label.get_text() for label in default_ax.get_xticklabels() if label.get_text()]
    poster_x_labels = [label.get_text() for label in poster_ax.get_xticklabels() if label.get_text()]
    assert poster_x_labels == default_x_labels
    assert poster_y_labels == default_y_labels[::2]
    assert len(poster_ax.get_yticks()) == len(default_ax.get_yticks())
    assert all(tick.tick1line.get_visible() for tick in poster_ax.yaxis.get_major_ticks())
    plt.close(fig)


def test_default_attached_vertical_colorbar_uses_full_height_axis() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    mappable = ax.imshow(np.arange(4).reshape(2, 2), vmin=0.0, vmax=100.0)
    map_layout.attach_colorbar(
        ax,
        mappable,
        label="fractional snow cover [%]",
        ticks=(0, 20, 40, 60, 80, 100),
        layout="vertical",
    )

    child_axes = getattr(ax, "_oa_child_axes")
    assert len(child_axes) == 1
    colorbar_ax = child_axes[0]
    fig.canvas.draw()
    assert not colorbar_ax.child_axes
    assert [tick.get_text() for tick in colorbar_ax.get_yticklabels()][-1] == "100"
    plt.close(fig)


def test_poster_attached_percent_colorbar_uses_five_tick_rhythm() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    mappable = ax.imshow(np.arange(4).reshape(2, 2), vmin=0.0, vmax=100.0)
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        map_layout.attach_colorbar(
            ax,
            mappable,
            label="fractional snow cover [%]",
            ticks=(0, 20, 40, 60, 80, 100),
            layout="vertical",
        )

    _, colorbar_ax = getattr(ax, "_oa_child_axes")[-2:]
    fig.canvas.draw()
    assert [tick.get_text() for tick in colorbar_ax.get_yticklabels()] == ["0", "25", "50", "75"]
    assert colorbar_ax.get_yticks()[-1] == pytest.approx(100.0)
    top_tick = colorbar_ax.yaxis.get_major_ticks()[-1]
    assert top_tick.label2.get_text() == "100"
    assert top_tick.label2.get_visible() is False
    plt.close(fig)


def test_poster_attached_nonpercent_colorbar_keeps_domain_ticks() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    mappable = ax.imshow(np.arange(4).reshape(2, 2), vmin=0.01, vmax=1.5)
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        map_layout.attach_colorbar(
            ax,
            mappable,
            label="snow depth [m]",
            ticks=(0.01, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5),
            layout="vertical",
        )

    _, colorbar_ax = getattr(ax, "_oa_child_axes")[-2:]
    fig.canvas.draw()
    tick_labels = [tick.get_text() for tick in colorbar_ax.get_yticklabels()]
    assert tick_labels != ["0", "25", "50", "75", "100"]
    assert colorbar_ax.get_yticks()[-1] == pytest.approx(1.5)
    top_tick = colorbar_ax.yaxis.get_major_ticks()[-1]
    assert top_tick.label2.get_text() == "1.50"
    assert top_tick.label2.get_visible() is False
    plt.close(fig)


def test_poster_attached_diverging_colorbar_uses_five_symmetric_ticks() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    mappable = ax.imshow(
        np.array([[-0.6, -0.3], [0.3, 0.6]]),
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6),
    )
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        map_layout.attach_colorbar(ax, mappable, label="increment [m]", layout="vertical")

    _, colorbar_ax = getattr(ax, "_oa_child_axes")[-2:]
    fig.canvas.draw()
    assert [tick.get_text() for tick in colorbar_ax.get_yticklabels()] == [
        "-0.6",
        "-0.3",
        "0",
        "0.3",
    ]
    top_tick = colorbar_ax.yaxis.get_major_ticks()[-1]
    assert top_tick.label2.get_text() == "0.6"
    assert top_tick.label2.get_visible() is False
    plt.close(fig)


def test_poster_attached_vertical_colorbar_uses_unit_header_without_tick_overlap() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    mappable = ax.imshow(np.arange(4).reshape(2, 2), vmin=0.0, vmax=100.0)
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        map_layout.attach_colorbar(
            ax,
            mappable,
            label="fractional snow cover [%]",
            ticks=(0, 20, 40, 60, 80, 100),
            layout="vertical",
        )

    container_ax, colorbar_ax = getattr(ax, "_oa_child_axes")[-2:]
    fig.canvas.draw()
    assert container_ax.texts[0].get_text() == "[%]"
    assert len(container_ax.child_axes) == 1
    assert colorbar_ax.get_position().height / container_ax.get_position().height == pytest.approx(0.88, abs=0.01)
    renderer = fig.canvas.get_renderer()
    map_bbox = ax.get_window_extent(renderer)
    container_bbox = container_ax.get_window_extent(renderer)
    colorbar_bbox = colorbar_ax.get_window_extent(renderer)
    unit_bbox = container_ax.texts[0].get_window_extent(renderer)
    assert container_bbox.y1 <= map_bbox.y1 + 0.5
    assert unit_bbox.y1 <= map_bbox.y1 + 0.5
    assert unit_bbox.y0 >= colorbar_bbox.y1 - 0.5
    assert unit_bbox.x0 == pytest.approx(container_bbox.x0, abs=0.5)
    top_tick = colorbar_ax.yaxis.get_major_ticks()[-1]
    assert top_tick.label2.get_visible() is False
    plt.close(fig)


def test_poster_map_typography_thins_dense_vertical_colorbar_labels() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 14.0, 13.5))):
        fig, ax = plt.subplots(figsize=(1.0, 1.8))
        mappable = ax.imshow(np.arange(4).reshape(2, 2))
        cbar = plt.colorbar(mappable, ax=ax, orientation="vertical")
        cbar.set_ticks(range(7))
        cbar.set_ticklabels(("0.01", "0.25", "0.50", "0.75", "1", "1.25", "1.5"))
        fig.canvas.draw()
        map_layout._thin_vertical_colorbar_ticklabels(cbar)
        visible = [label.get_text() for label in cbar.ax.get_yticklabels() if label.get_visible()]
        plt.close(fig)

    assert visible[0] == "0.01"
    assert visible[-1] == "1.5"
    assert len(visible) < 7


def test_poster_colorbar_panel_normalizes_percent_ticks() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(1.0, 2.0))
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=100.0), cmap="Greys")
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        panel_renderers_module.render_colorbar_panel(
            ax,
            panel=MapPanelSpec(kind="colorbar", row=0, col=1, source="fsc"),
            artifacts={
                "fsc": {
                    "mappable": mappable,
                    "colorbar_style": {
                        "label": "fractional snow cover [%]",
                        "ticks": (0, 20, 40, 60, 80, 100),
                    },
                }
            },
        )

    colorbar_ax = ax.child_axes[0]
    assert [tick.get_text() for tick in colorbar_ax.get_yticklabels()] == ["0", "25", "50", "75", "100"]
    plt.close(fig)


def test_poster_colorbar_panel_normalizes_diverging_ticks() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(1.0, 2.0))
    mappable = ScalarMappable(norm=TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6), cmap="coolwarm")
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        panel_renderers_module.render_colorbar_panel(
            ax,
            panel=MapPanelSpec(kind="colorbar", row=0, col=1, source="increment"),
            artifacts={
                "increment": {
                    "mappable": mappable,
                    "colorbar_style": {"label": "increment [m]"},
                }
            },
        )

    colorbar_ax = ax.child_axes[0]
    assert [tick.get_text() for tick in colorbar_ax.get_yticklabels()] == [
        "-0.6",
        "-0.3",
        "0",
        "0.3",
        "0.6",
    ]
    plt.close(fig)


def test_default_colorbar_panel_draws_unit_above_full_height_bar() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(1.0, 2.0))
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    panel_renderers_module.render_colorbar_panel(
        ax,
        panel=MapPanelSpec(kind="colorbar", row=0, col=1, source="snow_depth"),
        artifacts={
            "snow_depth": {
                "mappable": mappable,
                "colorbar_style": {"label": "snow depth [m]", "ticks": (0.0, 1.0)},
            }
        },
    )

    assert len(ax.child_axes) == 0
    assert ax.texts[0].get_text() == "[m]"
    assert ax.texts[0].get_position()[1] == pytest.approx(1.02)
    plt.close(fig)


def test_poster_colorbar_panel_uses_header_gap_for_unit_label() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(1.0, 2.0))
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    with _scaled_map_style(PosterRenderStyle(typography=PosterTypography(15.0, 13.0, 12.0))):
        assert panel_renderers_module._COLORBAR_PANEL_UNIT_HEADER_ENABLED is True
        panel_renderers_module.render_colorbar_panel(
            ax,
            panel=MapPanelSpec(kind="colorbar", row=0, col=1, source="snow_depth"),
            artifacts={
                "snow_depth": {
                    "mappable": mappable,
                    "colorbar_style": {"label": "snow depth [m]", "ticks": (0.0, 1.0)},
                }
            },
        )
    assert panel_renderers_module._COLORBAR_PANEL_UNIT_HEADER_ENABLED is False

    fig.canvas.draw()
    assert len(ax.child_axes) == 1
    child_bbox = ax.child_axes[0].get_position()
    parent_bbox = ax.get_position()
    assert child_bbox.height < parent_bbox.height
    assert child_bbox.y1 < parent_bbox.y1
    assert ax.texts[0].get_text() == "[m]"
    assert ax.texts[0].get_position()[1] == pytest.approx(1.0)
    plt.close(fig)


def test_poster_config_parses_target_sizes(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
maps:
  setup_overview:
    enabled: true
    layout:
      ncols: 2
    target_size_mm: [75.220833, 178.40262]
  da_events:
    enabled: true
    names: [da_6]
    target_size_mm: [250.78568, 156.94586]
plots:
  result_overview_custom:
    enabled: true
    target_size_mm: [429.34836, 186.90407]
    layout:
      h_pad: 0.24
      hspace: 0.08
      panel_height_factor: 0.8
      align_first_xtick_left: true
    panels:
      - panel: fSC
theme:
  scale: 1.55
  typography:
    title_pt: 15.0
    label_pt: 14.0
    support_pt: 13.5
  linework:
    panel_box_pt: 0.55
""",
        encoding="utf-8",
    )

    config = load_poster_config(config_path)

    assert config.setup_overview.target_size is not None
    assert config.setup_overview.target_size.width_mm == pytest.approx(75.220833)
    assert config.da_events.names == ("da_6",)
    assert config.da_events.target_size is not None
    assert config.da_events.target_size.height_mm == pytest.approx(156.94586)
    assert config.result_overview_custom.target_size is not None
    assert config.result_overview_custom.target_size.width_mm == pytest.approx(429.34836)
    assert config.result_overview_custom.h_pad == pytest.approx(0.24)
    assert config.result_overview_custom.hspace == pytest.approx(0.08)
    assert config.result_overview_custom.panel_height_factor == pytest.approx(0.8)
    assert config.result_overview_custom.align_first_xtick_left is True
    assert config.theme.scale == pytest.approx(1.55)
    assert config.theme.typography is not None
    assert config.theme.typography.title_pt == pytest.approx(15.0)
    assert config.theme.typography.label_pt == pytest.approx(14.0)
    assert config.theme.typography.support_pt == pytest.approx(13.5)
    assert config.theme.linework is not None
    assert config.theme.linework.panel_box_pt == pytest.approx(0.55)


def test_poster_config_defaults_theme_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
plots:
  result_overview_custom:
    enabled: true
    panels:
      - panel: fSC
""",
        encoding="utf-8",
    )

    config = load_poster_config(config_path)

    assert config.theme.scale == 1.0


def test_poster_config_rejects_non_positive_theme_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
theme:
  scale: 0
plots:
  result_overview_custom:
    enabled: true
    panels:
      - panel: fSC
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Poster theme.scale"):
        load_poster_config(config_path)


def test_poster_config_rejects_non_positive_result_panel_height_factor(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
plots:
  result_overview_custom:
    enabled: true
    layout:
      panel_height_factor: 0
    panels:
      - panel: fSC
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Poster result_overview_custom.layout.panel_height_factor"):
        load_poster_config(config_path)


def test_poster_config_rejects_incomplete_typography(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
theme:
  typography:
    title_pt: 14.2
    support_pt: 10.0
plots:
  result_overview_custom:
    enabled: true
    panels:
      - panel: fSC
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Poster theme.typography.label_pt"):
        load_poster_config(config_path)


def test_poster_setup_overview_drops_aspect_panel() -> None:
    recipe = MapRecipe(
        name="setup_overview",
        title="Setup overview",
        figure_title="Setup overview",
        layout=LayoutSpec(nrows=1, ncols=4),
        panels=(
            _panel("overview", 0, 0),
            _panel("dem", 0, 1),
            _panel("landcover", 0, 2),
            _panel("aspect", 0, 3),
        ),
    )

    poster_recipe = poster_setup_overview_recipe(
        recipe,
        PosterSetupOverviewConfig(enabled=True, drop_panel_kinds=("aspect",)),
    )

    assert poster_recipe.figure_title is None
    assert poster_recipe.layout.ncols == 3
    assert [panel.kind for panel in poster_recipe.panels] == ["overview", "dem", "landcover"]
    assert [panel.col for panel in poster_recipe.panels] == [0, 1, 2]


def test_poster_setup_overview_can_keep_panels_and_reflow_to_one_column() -> None:
    recipe = MapRecipe(
        name="setup_overview",
        title="Setup overview",
        figure_title="Setup overview",
        layout=LayoutSpec(nrows=1, ncols=4),
        panels=(
            _panel("overview", 0, 0),
            _panel("dem", 0, 1),
            _panel("landcover", 0, 2),
            _panel("aspect", 0, 3),
        ),
    )

    poster_recipe = poster_setup_overview_recipe(
        recipe,
        PosterSetupOverviewConfig(
            enabled=True,
            keep_panel_kinds=("dem", "landcover"),
            ncols=1,
        ),
    )

    assert poster_recipe.figure_title is None
    assert poster_recipe.layout.nrows == 2
    assert poster_recipe.layout.ncols == 1
    assert [panel.kind for panel in poster_recipe.panels] == ["dem", "landcover"]
    assert [(panel.row, panel.col) for panel in poster_recipe.panels] == [(0, 0), (1, 0)]


def test_poster_setup_overview_can_keep_panels_and_reflow_to_one_row() -> None:
    recipe = MapRecipe(
        name="setup_overview",
        title="Setup overview",
        figure_title="Setup overview",
        layout=LayoutSpec(nrows=1, ncols=4),
        panels=(
            _panel("overview", 0, 0),
            _panel("dem", 0, 1),
            _panel("landcover", 0, 2),
            _panel("aspect", 0, 3),
        ),
    )

    poster_recipe = poster_setup_overview_recipe(
        recipe,
        PosterSetupOverviewConfig(
            enabled=True,
            keep_panel_kinds=("dem", "landcover"),
            ncols=2,
        ),
    )

    assert poster_recipe.figure_title is None
    assert poster_recipe.layout.nrows == 1
    assert poster_recipe.layout.ncols == 2
    assert [panel.kind for panel in poster_recipe.panels] == ["dem", "landcover"]
    assert [(panel.row, panel.col) for panel in poster_recipe.panels] == [(0, 0), (0, 1)]


def test_poster_da_event_drops_open_loop_column_and_keeps_response_panels() -> None:
    recipe = MapRecipe(
        name="da_4",
        title="DA 4",
        figure_title="DA 4 - station snow depth",
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=LayoutSpec(nrows=1, ncols=4),
        panels=(
            _panel("snow_depth", 0, 0, source="open_loop", title="Open-loop snow depth"),
            _panel("snow_depth", 0, 1, source="ensemble_mean", title="Prior mean snow depth"),
            _panel("snow_depth", 0, 2, source="analysis_mean", title="Posterior mean snow depth"),
            _panel("snow_depth", 0, 3, source="analysis_increment", title="Snow-depth increment"),
        ),
    )

    poster_recipe = poster_da_event_recipe(recipe, PosterDaEventsConfig(enabled=True))

    assert poster_recipe.figure_title is None
    assert poster_recipe.layout.ncols == 3
    assert [panel.source for panel in poster_recipe.panels] == [
        "ensemble_mean",
        "analysis_mean",
        "analysis_increment",
    ]
    assert [panel.col for panel in poster_recipe.panels] == [0, 1, 2]
    assert poster_recipe.output_subdir == GENERATED_DA_MAPS_SUBDIR


def test_poster_da_event_abbreviates_long_titles() -> None:
    recipe = MapRecipe(
        name="da_6",
        title="DA 6",
        figure_title="DA 6 - wet snow line",
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=LayoutSpec(nrows=2, ncols=3),
        panels=(
            _panel("wet_snow_elevation_fraction", 0, 0, title="Prior elevation band WSF"),
            _panel("wet_snow_elevation_fraction", 0, 1, title="Posterior elevation band WSF"),
            _panel("wet_snow_elevation_fraction", 0, 2, title="Observed elevation band WSF"),
            _panel("fsc", 1, 0, title="Prior snow cover probability"),
            _panel("fsc", 1, 1, title="Posterior snow cover probability"),
            _panel("fsc", 1, 2, title="Satellite FSC observation"),
        ),
    )

    poster_recipe = poster_da_event_recipe(recipe, PosterDaEventsConfig(enabled=True, drop_first_column=False))

    assert [panel.title for panel in poster_recipe.panels] == [
        "Prior elev. band WSF",
        "Post. elev. band WSF",
        "Obs. elev. band WSF",
        "Prior snow-cover prob.",
        "Post. snow-cover prob.",
        "Satellite FSC obs.",
    ]


def test_poster_wet_snow_line_event_uses_paper_compacting_before_dropping_open_loop() -> None:
    panels = []
    for row, kind in (
        (0, "wet_snow_line"),
        (1, "wet_snow_elevation_fraction"),
        (2, "snow_depth"),
    ):
        for col, source in enumerate(("open_loop", "prior_probability", "posterior_probability", None)):
            panels.append(_panel(kind, row, col, source=source))
    recipe = MapRecipe(
        name="da_6",
        title="DA 6",
        figure_title="DA 6 - wet snow line",
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=LayoutSpec(nrows=3, ncols=4),
        panels=tuple(panels),
    )

    poster_recipe = poster_da_event_recipe(recipe, PosterDaEventsConfig(enabled=True))

    assert poster_recipe.layout.nrows == 3
    assert poster_recipe.layout.ncols == 3
    assert {panel.kind for panel in poster_recipe.panels} == {
        "wet_snow_line",
        "wet_snow_elevation_fraction",
        "snow_depth",
    }
    assert {panel.row for panel in poster_recipe.panels} == {0, 1, 2}
    assert {panel.col for panel in poster_recipe.panels} == {0, 1, 2}
    assert all(panel.source != "open_loop" for panel in poster_recipe.panels)


def _da_recipe(name: str) -> MapRecipe:
    return MapRecipe(
        name=name,
        title=name.upper(),
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=LayoutSpec(nrows=1, ncols=2),
        panels=(
            _panel("fsc", 0, 0, source="open_loop"),
            _panel("fsc", 0, 1, source="ensemble_mean"),
        ),
    )


def test_poster_da_events_names_filters_generated_maps(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
maps:
  da_events:
    enabled: true
    names: [da_6]
plots: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(poster_mod, "generated_da_map_recipes", lambda _project_dir: (_da_recipe("da_5"), _da_recipe("da_6")))

    config = load_poster_config(config_path)
    recipes = poster_map_recipes(tmp_path, config)

    assert [recipe.name for recipe in recipes] == ["da_6"]


def test_poster_da_events_names_rejects_missing_map(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
maps:
  da_events:
    enabled: true
    names: [da_9]
plots: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(poster_mod, "generated_da_map_recipes", lambda _project_dir: (_da_recipe("da_6"),))

    config = load_poster_config(config_path)
    with pytest.raises(ValueError, match="Poster da_events.names not found: da_9"):
        poster_map_recipes(tmp_path, config)


def test_poster_da_events_without_names_keeps_all_generated_maps(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
maps:
  da_events:
    enabled: true
plots: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(poster_mod, "generated_da_map_recipes", lambda _project_dir: (_da_recipe("da_5"), _da_recipe("da_6")))

    config = load_poster_config(config_path)
    recipes = poster_map_recipes(tmp_path, config)

    assert [recipe.name for recipe in recipes] == ["da_5", "da_6"]


def _data_href(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def test_measure_poster_svg_targets_matches_embedded_png_hashes(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    poster_root = project_dir / "results" / "poster"
    setup_png = poster_root / "maps" / "setup_overview.png"
    da_6_png = poster_root / "maps" / GENERATED_DA_MAPS_SUBDIR / "da_6.png"
    da_8_png = poster_root / "maps" / GENERATED_DA_MAPS_SUBDIR / "da_8.png"
    overview_png = poster_root / "plots" / "results" / "result_overview_custom.png"
    for path, payload in (
        (setup_png, b"setup-current-png"),
        (da_6_png, b"da-6-current-png"),
        (da_8_png, b"da-8-current-png"),
        (overview_png, b"overview-current-png"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    svg_path = tmp_path / "poster.svg"
    svg_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="594mm" height="841mm" viewBox="0 0 594 841">
  <g transform="translate(10,20)">
    <image x="1" y="2" width="75.220833" height="178.40262" xlink:href="{_data_href(setup_png.read_bytes())}" />
    <image x="10" y="200" width="250.78566" height="156.94586" xlink:href="{_data_href(da_8_png.read_bytes())}" />
    <image x="300" y="200" width="250.78570" height="156.94586" xlink:href="{_data_href(da_6_png.read_bytes())}" />
    <image x="20" y="400" width="429.34836" height="186.90407" xlink:href="{_data_href(overview_png.read_bytes())}" />
  </g>
</svg>""",
        encoding="utf-8",
    )

    sizes = measure_poster_svg_targets(project_dir, svg_path)

    assert set(sizes) == {"setup_overview", "da_events", "result_overview_custom"}
    assert sizes["setup_overview"].width_mm == pytest.approx(75.220833)
    assert sizes["setup_overview"].height_mm == pytest.approx(178.40262)
    assert sizes["da_events"].width_mm == pytest.approx(250.78568)
    assert sizes["da_events"].height_mm == pytest.approx(156.94586)
    assert sizes["result_overview_custom"].width_mm == pytest.approx(429.34836)


def test_measure_poster_svg_targets_matches_linked_wsl_file_uris(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    poster_root = project_dir / "results" / "poster"
    setup_png = poster_root / "maps" / "setup_overview.png"
    da_png = poster_root / "maps" / GENERATED_DA_MAPS_SUBDIR / "da_6.png"
    overview_png = poster_root / "plots" / "results" / "result_overview_custom.png"
    for path in (setup_png, da_png, overview_png):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"current-{path.name}".encode("utf-8"))

    def wsl_uri(path: Path) -> str:
        rel = path.relative_to(poster_root)
        host_project = Path("/home/franz/workspace/dev_examples/rofental_tdew/projects/project_2022_2023")
        host_path = host_project / "results" / "poster" / rel
        return f"file:////wsl.localhost/Ubuntu{host_path.as_posix()}"

    svg_path = tmp_path / "poster.svg"
    svg_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="594mm" height="841mm" viewBox="0 0 594 841">
  <g transform="translate(2,3)">
    <image width="78.24" height="185.53" xlink:href="{wsl_uri(setup_png)}" />
    <image width="250.78567" height="156.94586" xlink:href="{wsl_uri(da_png)}" />
    <image width="429.34836" height="186.90407" xlink:href="{wsl_uri(overview_png)}" />
  </g>
</svg>""",
        encoding="utf-8",
    )

    sizes = measure_poster_svg_targets(project_dir, svg_path)

    assert sizes["setup_overview"].width_mm == pytest.approx(78.24)
    assert sizes["setup_overview"].height_mm == pytest.approx(185.53)
    assert sizes["da_events"].width_mm == pytest.approx(250.78567)
    assert sizes["result_overview_custom"].height_mm == pytest.approx(186.90407)


def test_write_poster_target_sizes_updates_config(tmp_path: Path) -> None:
    config_path = tmp_path / "poster.yml"
    config_path.write_text(
        """
maps:
  setup_overview:
    enabled: true
  da_events:
    enabled: true
plots:
  result_overview_custom:
    enabled: true
    panels:
      - panel: fSC
""",
        encoding="utf-8",
    )
    sizes = {
        "setup_overview": poster_mod.PosterTargetSize(75.220833, 178.40262),
        "da_events": poster_mod.PosterTargetSize(250.78568, 156.94586),
        "result_overview_custom": poster_mod.PosterTargetSize(429.34836, 186.90407),
    }

    write_poster_target_sizes(config_path, sizes)

    yaml = ruamel.yaml.YAML(typ="safe")
    config = yaml.load(config_path.read_text(encoding="utf-8"))
    assert config["maps"]["setup_overview"]["target_size_mm"] == [75.220833, 178.40262]
    assert config["maps"]["da_events"]["target_size_mm"] == [250.78568, 156.94586]
    assert config["plots"]["result_overview_custom"]["target_size_mm"] == [429.34836, 186.90407]


def test_save_figure_png_target_size_preserves_artist_linewidth(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    (line,) = ax.plot([0, 1], [0, 1], lw=2.5)
    output = tmp_path / "target.png"

    save_figure_png(fig, output, dpi=100, target_size_in=(2.0, 1.0))
    plt.close(fig)

    with Image.open(output) as image:
        assert image.size == (200, 100)
    assert line.get_linewidth() == 2.5


def test_render_poster_profile_passes_three_panel_result_overview_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True)
    (project_dir / "poster.yml").write_text(
        """
theme:
  scale: 1.4
  typography:
    title_pt: 14.2
    label_pt: 12.0
    support_pt: 10.0
  linework:
    panel_box_pt: 0.45
plots:
  result_overview_custom:
    enabled: true
    target_size_mm: [429.34836, 186.90407]
    layout:
      h_pad: 0.24
      hspace: 0.08
      panel_height_factor: 0.8
      align_first_xtick_left: true
    panels:
      - panel: fSC
        title: Snow cover fraction
      - panel: WSLA
        title: Wet snow line
      - panel: station-sd
        station_id: proviantdepot
        title: Snow depth (Proviantdepot 2659 m)
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(poster_mod, "render_poster_maps", lambda **_kwargs: [])

    def fake_plot_result_overview_cli(argv: list[str], *, configure_logger: bool = True) -> int:
        del configure_logger
        assert "--no-paper-mirror" in argv
        assert argv[argv.index("--style-scale") + 1] == "1.4"
        assert argv[argv.index("--poster-title-pt") + 1] == "14.2"
        assert argv[argv.index("--poster-label-pt") + 1] == "12"
        assert argv[argv.index("--poster-support-pt") + 1] == "10"
        assert argv[argv.index("--poster-panel-box-pt") + 1] == "0.45"
        assert argv[argv.index("--target-size-mm") + 1 : argv.index("--target-size-mm") + 3] == [
            "429.348",
            "186.904",
        ]
        assert argv[argv.index("--poster-h-pad") + 1] == "0.24"
        assert argv[argv.index("--poster-hspace") + 1] == "0.08"
        assert argv[argv.index("--poster-panel-height-factor") + 1] == "0.8"
        assert "--poster-align-first-xtick-left" in argv
        config_path = Path(argv[argv.index("--custom-config") + 1])
        output_path = Path(argv[argv.index("--output") + 1])
        yaml = ruamel.yaml.YAML(typ="safe")
        captured["panels"] = yaml.load(config_path.read_text(encoding="utf-8"))["panels"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("poster png", encoding="utf-8")
        return 0

    monkeypatch.setattr(poster_mod, "plot_result_overview_cli", fake_plot_result_overview_cli)

    outputs = render_poster_profile(project_dir=project_dir)

    expected = project_dir / "results" / "poster" / "plots" / "results" / "result_overview_custom.png"
    assert outputs == [expected]
    assert expected.read_text(encoding="utf-8") == "poster png"
    panels = captured["panels"]
    assert [panel["panel"] for panel in panels] == ["fSC", "WSLA", "station-sd"]
    assert panels[2]["station_id"] == "proviantdepot"
    assert not (project_dir / "results" / "paper" / "poster").exists()


def test_render_poster_maps_passes_target_sizes(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "poster.yml"
    config_path.write_text(
        """
theme:
  scale: 1.4
  typography:
    title_pt: 14.2
    label_pt: 12.0
    support_pt: 10.0
  linework:
    panel_box_pt: 0.45
maps:
  setup_overview:
    enabled: true
    layout:
      ncols: 2
    target_size_mm: [75.220833, 178.40262]
  da_events:
    enabled: true
    target_size_mm: [250.78568, 156.94586]
""",
        encoding="utf-8",
    )
    config = load_poster_config(config_path)
    recipes = (
        MapRecipe(
            name="setup_overview",
            title="Setup overview",
            layout=LayoutSpec(nrows=1, ncols=1),
            panels=(_panel("dem", 0, 0),),
        ),
        MapRecipe(
            name="da_6",
            title="DA 6",
            output_subdir=GENERATED_DA_MAPS_SUBDIR,
            layout=LayoutSpec(nrows=1, ncols=1),
            panels=(_panel("snow_depth", 0, 0),),
        ),
    )
    captured: dict[str, tuple[tuple[float, float] | None, object]] = {}

    monkeypatch.setattr(poster_mod, "poster_map_recipes", lambda _project_dir, _config: recipes)
    monkeypatch.setattr(poster_mod, "_collect_shared_model_vmax", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(poster_mod, "load_static_context", lambda _project_dir: object())

    def fake_render_map_recipe(
        *,
        recipe,
        output_path,
        target_size_in=None,
        poster_style=None,
        **_kwargs,
    ):
        captured[recipe.name] = (target_size_in, poster_style)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("png", encoding="utf-8")
        return output_path

    monkeypatch.setattr(poster_mod, "render_map_recipe", fake_render_map_recipe)

    outputs = render_poster_maps(project_dir=project_dir, config=config, max_workers=1)

    assert len(outputs) == 2
    assert captured["setup_overview"][0] == pytest.approx((75.220833 / 25.4, 178.40262 / 25.4))
    setup_style = captured["setup_overview"][1]
    assert setup_style.scale == pytest.approx(1.4)
    assert setup_style.typography.title_pt == pytest.approx(14.2)
    assert setup_style.typography.label_pt == pytest.approx(12.0)
    assert setup_style.typography.support_pt == pytest.approx(10.0)
    assert setup_style.linework.panel_box_pt == pytest.approx(0.45)
    assert captured["da_6"][0] == pytest.approx((250.78568 / 25.4, 156.94586 / 25.4))
    assert captured["da_6"][1].typography.title_pt == pytest.approx(14.2)
