"""
Tests for the TreeGenerator crossing-avoidance logic.

Verifies that:
  - Single-trunk: trunk segments are never rejected by crossing checks
  - Multi-trunk: both trunks remain fully intact even when branches overlap
  - Branch-branch crossing is correctly detected and pruned
"""

import unittest
import numpy as np

from fractal_wing.structures import Seg, SubParams, BranchDef, Station, TrunkSpec, segments_cross
from fractal_wing.generator import TreeGenerator


class MockWingAdapter:
    """Minimal wing adapter for testing — a simple rectangular domain."""

    def __init__(self, chord=10.0, span=15.0):
        self._chord = chord
        self.b = span
        self.sf = 0.0
        self.sr = 1.0

    def c(self, e):
        return self._chord

    def xle(self, e):
        return 0.0

    def xte(self, e):
        return self._chord

    def xfs(self, e):
        return self.sf * self._chord

    def xrs(self, e):
        return self.sr * self._chord

    def xdf(self, e):
        return 0.0

    def xdr(self, e):
        return self._chord

    def xdc(self, e):
        return self._chord * 0.5

    def inside(self, x, y):
        return -0.5 <= x <= self._chord + 0.5 and -0.5 <= y <= self.b + 0.5

    def clip(self, p1, p2):
        return p2

    def cla(self, sc=1.0):
        return np.arctan2(sc * self.b, self.xdc(sc) - self.xdc(0))

    def z_at(self, x, y):
        return 0.05, -0.05


class TestCrossingAvoidance(unittest.TestCase):

    def _make_wing(self):
        return MockWingAdapter(chord=10.0, span=15.0)

    def test_trunk_segments_always_placed_single_trunk(self):
        """Trunk segments (level 0) must never be interrupted — single trunk."""
        wing = self._make_wing()
        gen = TreeGenerator(wing)

        # Dense branching that will produce lots of crossings
        stations = [
            Station(position=p, branches=[
                BranchDef(angle=45, length=5.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
                BranchDef(angle=-45, length=5.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
            ])
            for p in np.linspace(0.05, 0.95, 10)
        ]

        spec = TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=False,
        )
        segs = gen.generate(spec)

        # Extract trunk segments (level 0)
        trunk_segs = [s for s in segs if s.level == 0]
        # There should be n_stations + 1 trunk segments (between each pair of stations + root-to-first + last-to-tip)
        expected_trunk_count = len(stations) + 1
        self.assertEqual(len(trunk_segs), expected_trunk_count,
                         f"Expected {expected_trunk_count} trunk segments, got {len(trunk_segs)}. "
                         f"Trunk segments should never be rejected by crossing checks.")

    def test_trunk_segments_always_placed_multi_trunk(self):
        """Both trunks must remain intact in multi-trunk mode with crossing OFF."""
        wing = self._make_wing()
        gen = TreeGenerator(wing)

        n_stations = 8
        stations_fn = lambda: [
            Station(position=p, branches=[
                BranchDef(angle=50, length=4.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
                BranchDef(angle=-50, length=4.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
            ])
            for p in np.linspace(0.05, 0.95, n_stations)
        ]

        specs = [
            TrunkSpec(
                chord_frac=0.25, span_cov=1.0, thick=0.004,
                stations=stations_fn(), allow_crossing=False,
            ),
            TrunkSpec(
                chord_frac=0.75, span_cov=1.0, thick=0.004,
                stations=stations_fn(), allow_crossing=False,
            ),
        ]

        segs = gen.generate_multi(specs)
        trunk_segs = [s for s in segs if s.level == 0]

        # Each trunk should have n_stations + 1 segments
        expected_per_trunk = n_stations + 1
        expected_total = expected_per_trunk * 2
        self.assertEqual(len(trunk_segs), expected_total,
                         f"Expected {expected_total} trunk segments (2 trunks × {expected_per_trunk}), "
                         f"got {len(trunk_segs)}. Trunk segments must never be rejected.")

    def test_unprotected_trunk_dynamic_termination(self):
        """Unprotected trunks (protect_trunk=False) should stop growing when encountering a crossing."""
        wing = self._make_wing()
        gen = TreeGenerator(wing)

        n_stations = 8
        stations_fn = lambda: [
            Station(position=p, branches=[
                BranchDef(angle=50, length=4.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
                BranchDef(angle=-50, length=4.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
            ])
            for p in np.linspace(0.05, 0.95, n_stations)
        ]

        specs = [
            TrunkSpec(
                chord_frac=0.25, span_cov=1.0, thick=0.004,
                stations=stations_fn(), allow_crossing=False,
                protect_trunk=True, # Primary trunk is protected
            ),
            TrunkSpec(
                chord_frac=0.75, span_cov=1.0, thick=0.004,
                stations=stations_fn(), allow_crossing=False,
                protect_trunk=False, # Secondary trunk stops if it hits a crossing
            ),
        ]

        segs = gen.generate_multi(specs)
        
        trunk1_segs = [s for s in segs if s.level == 0 and s.p0[0] < 5.0]
        trunk2_segs = [s for s in segs if s.level == 0 and s.p0[0] > 5.0]

        expected_trunk1 = n_stations + 1
        
        self.assertEqual(len(trunk1_segs), expected_trunk1, "Primary protected trunk should be fully intact.")
        self.assertLess(len(trunk2_segs), expected_trunk1, "Secondary unprotected trunk should terminate early due to crossing constraints.")

    def test_crossing_off_reduces_branch_count(self):
        """Crossing OFF should produce fewer total segments than crossing ON."""
        wing = self._make_wing()

        stations = [
            Station(position=p, branches=[
                BranchDef(angle=40, length=4.0,
                          sub=SubParams(max_depth=3, mode='sympodial')),
                BranchDef(angle=-40, length=4.0,
                          sub=SubParams(max_depth=3, mode='sympodial')),
            ])
            for p in np.linspace(0.05, 0.95, 12)
        ]

        gen_on = TreeGenerator(wing)
        segs_on = gen_on.generate(TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=True,
        ))

        gen_off = TreeGenerator(wing)
        segs_off = gen_off.generate(TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=False,
        ))

        self.assertGreater(len(segs_on), len(segs_off),
                           "Crossing OFF should prune branches, resulting in fewer total segments.")

    def test_no_branch_crossings_when_off(self):
        """When crossing is OFF, no two branch segments should cross each other."""
        wing = self._make_wing()
        gen = TreeGenerator(wing)

        stations = [
            Station(position=p, branches=[
                BranchDef(angle=35, length=3.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
                BranchDef(angle=-35, length=3.0,
                          sub=SubParams(max_depth=2, mode='sympodial')),
            ])
            for p in np.linspace(0.1, 0.9, 8)
        ]

        spec = TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=False,
        )
        segs = gen.generate(spec)

        # Check that no pair of segments crosses
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if segments_cross(segs[i].p0, segs[i].p1,
                                  segs[j].p0, segs[j].p1):
                    self.fail(
                        f"Segments {segs[i].sid} (level {segs[i].level}) and "
                        f"{segs[j].sid} (level {segs[j].level}) cross each other "
                        f"when allow_crossing=False"
                    )

    def test_segments_cross_function(self):
        """Verify the geometric crossing detection function itself."""
        # Two clearly crossing segments
        a0 = np.array([0.0, 0.0])
        a1 = np.array([1.0, 1.0])
        b0 = np.array([0.0, 1.0])
        b1 = np.array([1.0, 0.0])
        self.assertTrue(segments_cross(a0, a1, b0, b1))

        # Two parallel segments (should not cross)
        c0 = np.array([0.0, 0.0])
        c1 = np.array([1.0, 0.0])
        d0 = np.array([0.0, 1.0])
        d1 = np.array([1.0, 1.0])
        self.assertFalse(segments_cross(c0, c1, d0, d1))

        # Two segments sharing an endpoint (should not cross due to 0.01 margin)
        e0 = np.array([0.0, 0.0])
        e1 = np.array([1.0, 0.0])
        f0 = np.array([1.0, 0.0])
        f1 = np.array([1.0, 1.0])
        self.assertFalse(segments_cross(e0, e1, f0, f1))


if __name__ == '__main__':
    unittest.main()
