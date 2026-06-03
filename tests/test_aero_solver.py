import os
import tempfile
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from canopy.aero_solver import run_aerodynamic_analysis, map_aerodynamic_loads
from canopy.fem_solver import parse_mesh_for_mapping, build_ccx_deck


class TestAeroSolver(unittest.TestCase):

    def test_map_aerodynamic_loads_pure(self):
        # 1. Define inputs with simple geometry
        # 2 panels: at (0, 0, 0) with force (0, 0, 10), and (1, 0, 0) with force (0, 0, 20)
        aero_centroids = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        aero_forces = np.array([
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 20.0]
        ])

        # Skin nodes at exactly the same points, plus an extra one
        nodes_dict = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (2.0, 0.0, 0.0)
        }
        skin_node_ids = {1, 2, 3}

        # Run mapping
        mapped = map_aerodynamic_loads(
            aero_centroids=aero_centroids,
            aero_forces=aero_forces,
            nodes_dict=nodes_dict,
            skin_node_ids=skin_node_ids,
            num_neighbors=2,
            power=2.0
        )

        # Total forces should match perfectly (conserved)
        total_mapped = np.sum(list(mapped.values()), axis=0)
        np.testing.assert_allclose(total_mapped, [0.0, 0.0, 30.0], atol=1e-7)

        # Force at node 1 should be close to 10, node 2 close to 20
        self.assertIn(1, mapped)
        self.assertIn(2, mapped)
        self.assertNotIn(3, mapped)

    @patch('subprocess.run')
    def test_run_aerodynamic_analysis_mock(self, mock_run):
        # Mock subprocess run to execute successfully
        mock_process = MagicMock()
        mock_process.return_value.returncode = 0
        mock_run.return_value = mock_process

        # Create dummy wing with necessary methods
        class MockWing:
            def to_vertex_grids(self, *args, **kwargs):
                # Return small grid
                X = np.zeros((3, 4))
                Y = np.ones((3, 4))
                Z = np.zeros((3, 4))
                return X, Y, Z

        wing = MockWing()

        with tempfile.TemporaryDirectory() as tmpdir:
            # We mock the json load since solve_flowpanel.jl would write this file
            loads_json_path = os.path.join(tmpdir, "aero_loads.json")
            dummy_data = {
                "centroids": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "forces": [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
                "areas": [1.0, 1.0],
                "Cps": [-0.5, -0.2]
            }
            with open(loads_json_path, 'w') as f:
                json.dump(dummy_data, f)

            # We intercept subprocess.run to copy the dummy loads file to the correct place
            # so the solver doesn't fail reading it.
            def side_effect(*args, **kwargs):
                # The second arg is output path in the actual call
                out_path = args[0][-1]
                with open(out_path, 'w') as f:
                    json.dump(dummy_data, f)
                return mock_process

            mock_run.side_effect = side_effect

            res = run_aerodynamic_analysis(
                wing=wing,
                aoa=4.0,
                magVinf=30.0,
                rho=1.225,
                num_points_profile=4,
                debug=False,
                temp_dir=tmpdir
            )

            # Assert results were read correctly
            self.assertEqual(len(res["centroids"]), 2)
            self.assertEqual(res["forces"][0][2], 10.0)
            self.assertEqual(res["Cps"][1], -0.2)

    @patch('subprocess.run')
    def test_run_aerodynamic_analysis_with_spanwise(self, mock_run):
        # Mock subprocess run to execute successfully
        mock_process = MagicMock()
        mock_process.return_value.returncode = 0
        mock_run.return_value = mock_process

        # Create dummy segment specs
        class MockSegment:
            def __init__(self, span, num_sections):
                self.span = span
                self.num_sections = num_sections

        class MockWing:
            def __init__(self):
                self.segments = [MockSegment(10.0, 5), MockSegment(20.0, 10)]

            def to_vertex_grids(self, *args, **kwargs):
                # Inside to_vertex_grids, verify num_sections was temporarily modified:
                # Total spanwise points is 15. Proportional:
                # Seg 1: 15 * (10 / 30) = 5
                # Seg 2: 15 * (20 / 30) = 10
                assert self.segments[0].num_sections == 5
                assert self.segments[1].num_sections == 10
                return np.zeros((3, 4)), np.ones((3, 4)), np.zeros((3, 4))

        wing = MockWing()

        with tempfile.TemporaryDirectory() as tmpdir:
            loads_json_path = os.path.join(tmpdir, "aero_loads.json")
            dummy_data = {
                "centroids": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "forces": [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
                "areas": [1.0, 1.0],
                "Cps": [-0.5, -0.2]
            }
            with open(loads_json_path, 'w') as f:
                json.dump(dummy_data, f)

            def side_effect(*args, **kwargs):
                out_path = args[0][-1]
                with open(out_path, 'w') as f:
                    json.dump(dummy_data, f)
                return mock_process

            mock_run.side_effect = side_effect

            res = run_aerodynamic_analysis(
                wing=wing,
                aoa=4.0,
                magVinf=30.0,
                rho=1.225,
                num_points_profile=4,
                num_points_spanwise=15,
                debug=False,
                temp_dir=tmpdir
            )

            # Check that original sections were successfully restored
            self.assertEqual(wing.segments[0].num_sections, 5)
            self.assertEqual(wing.segments[1].num_sections, 10)

    def test_parse_mesh_for_mapping(self):
        # Create a mock INP file
        mesh_content = """**
*NODE, NSET=NALL
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*ELEMENT, TYPE=S4, ELSET=WingSkin
1, 1, 2, 3, 1
*ELSET, ELSET=WingSkin
1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as f:
            f.write(mesh_content)
            tmp_name = f.name

        try:
            nodes, skin_nodes = parse_mesh_for_mapping(tmp_name)
            self.assertEqual(len(nodes), 3)
            self.assertEqual(nodes[1], (0.0, 0.0, 0.0))
            self.assertEqual(skin_nodes, {1, 2, 3})
        finally:
            os.remove(tmp_name)


if __name__ == '__main__':
    unittest.main()
