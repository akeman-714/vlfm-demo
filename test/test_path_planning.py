# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.path_planning import (
    downsample_path,
    plan_path,
    rc_to_xy,
    snap_to_navigable,
    xy_to_rc,
)


def test_xy_rc_roundtrip_matches_basemap() -> None:
    bm = BaseMap(size=1000, pixels_per_meter=20)
    # Origin is pinned to the pixel center (500, 500).
    assert xy_to_rc(bm, np.array([0.0, 0.0])) == (500, 500)
    # xy -> rc -> xy should return the original within one pixel (0.05 m @ 20px/m).
    for xy in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, -3.0), (-4.5, 7.25)]:
        rc = xy_to_rc(bm, np.array(xy))
        xy_back = rc_to_xy(bm, rc)
        np.testing.assert_allclose(xy_back, xy, atol=0.05)


def test_xy_to_rc_clips_out_of_bounds() -> None:
    bm = BaseMap(size=1000, pixels_per_meter=20)
    # Way outside the +-25 m map; must clip into [0, size).
    r, c = xy_to_rc(bm, np.array([1000.0, -1000.0]))
    assert 0 <= r < 1000 and 0 <= c < 1000


def test_snap_to_navigable_returns_nearest_free_cell() -> None:
    navigable = np.ones((5, 5), dtype=bool)
    navigable[2, 2] = False
    snapped = snap_to_navigable(navigable, (2, 2))
    assert snapped is not None
    assert navigable[snapped]
    # nearest free cell is one step away
    assert abs(snapped[0] - 2) + abs(snapped[1] - 2) == 1


def test_snap_to_navigable_none_when_no_free_cell() -> None:
    navigable = np.zeros((5, 5), dtype=bool)
    assert snap_to_navigable(navigable, (2, 2)) is None


def test_plan_path_routes_around_wall() -> None:
    navigable = np.ones((11, 11), dtype=bool)
    # Vertical wall at col 5 spanning rows 0..7; gap left at rows 8..10.
    navigable[0:8, 5] = False
    start, goal = (5, 1), (5, 9)
    path = plan_path(navigable, start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal
    # Every cell on the path is navigable (never cuts through the wall).
    assert all(navigable[r, c] for r, c in path)
    # Must detour down past the wall's end (row >= 8) to get through the gap.
    assert max(r for r, _ in path) >= 8


def test_plan_path_none_when_goal_walled_off() -> None:
    navigable = np.ones((7, 7), dtype=bool)
    # Fully enclose (3, 3) with obstacles -> unreachable through free space.
    navigable[2, 2:5] = False
    navigable[4, 2:5] = False
    navigable[2:5, 2] = False
    navigable[2:5, 4] = False
    assert plan_path(navigable, (0, 0), (3, 3)) is None


def test_plan_path_trivial_when_start_equals_goal() -> None:
    navigable = np.ones((5, 5), dtype=bool)
    assert plan_path(navigable, (2, 2), (2, 2)) == [(2, 2)]


def test_downsample_path_keeps_endpoints_and_thins() -> None:
    straight = [(0, i) for i in range(40)]
    out = downsample_path(straight, spacing_px=10)
    assert out[0] == straight[0]
    assert out[-1] == straight[-1]
    assert len(out) < len(straight)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
