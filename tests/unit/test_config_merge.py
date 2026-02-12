import unittest
import sys
import types

# Provide a tiny stub so importing core.config does not depend on a full
# openamundsen installation in this unit test context.
openamundsen_pkg = types.ModuleType("openamundsen")
openamundsen_conf = types.ModuleType("openamundsen.conf")
openamundsen_conf.parse_config = lambda cfg: cfg
openamundsen_util = types.ModuleType("openamundsen.util")
openamundsen_util.read_yaml_file = lambda _: {}
openamundsen_util.to_yaml = lambda _: ""
sys.modules.setdefault("openamundsen", openamundsen_pkg)
sys.modules["openamundsen.conf"] = openamundsen_conf
sys.modules["openamundsen.util"] = openamundsen_util

from openamundsen_da.core.config import merge_configs


class MergeConfigsTests(unittest.TestCase):
    def test_step_overrides_project_and_setup(self):
        setup = {
            "timestep": "3H",
            "output_data": {"timeseries": {"format": "csv"}},
        }
        project = {
            "output_data": {"timeseries": {"format": "netcdf"}},
        }
        step = {
            "output_data": {"timeseries": {"format": "memory"}},
        }

        merged = merge_configs(setup, project, step)

        self.assertEqual(merged["timestep"], "3H")
        self.assertEqual(merged["output_data"]["timeseries"]["format"], "memory")

    def test_nested_dict_is_shallow_merged_with_project_precedence(self):
        setup = {
            "output_data": {
                "timeseries": {"format": "csv"},
                "grids": {"format": "netcdf"},
            }
        }
        project = {"output_data": {"timeseries": {"write_freq": "D"}}}

        merged = merge_configs(setup, project, {})

        self.assertIn("timeseries", merged["output_data"])
        self.assertIn("grids", merged["output_data"])
        self.assertEqual(merged["output_data"]["timeseries"]["write_freq"], "D")
        self.assertEqual(merged["output_data"]["grids"]["format"], "netcdf")


if __name__ == "__main__":
    unittest.main()
