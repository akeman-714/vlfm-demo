# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Grid path-planning helpers for global navigation (Module 3).

Everything in this file works purely on a 2D boolean ``navigable`` array indexed
as ``[row, col]`` -- the *same* indexing used by ``ObstacleMap`` (see
``obstacle_map.py:101`` which writes ``self._map[px[:, 1], px[:, 0]]``).

World ``(x, y)`` <-> array ``(row, col)`` conversion is delegated entirely to
``BaseMap._xy_to_px`` / ``BaseMap._px_to_xy`` (``base_map.py:35-60``) via the
``xy_to_rc`` / ``rc_to_xy`` adapters below, so the y-flip + row-flip is never
re-implemented here.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

RC = Tuple[int, int]

# Per-pixel cost assigned to non-navigable cells. Large enough that any path
# crossing even a single obstacle pixel dwarfs a fully-navigable path
# (a 1000x1000 free path costs < ~3000), so we can detect "had to cut through
# an obstacle" cheaply. float64 keeps the MCP sum well clear of overflow.
OBSTACLE_COST = 1e6


def xy_to_rc(base_map: Any, xy: np.ndarray) -> RC:
    """World ``(x, y)`` [meters] -> array ``(row, col)`` index, clipped to bounds."""
    px = base_map._xy_to_px(np.asarray(xy, dtype=np.float32).reshape(1, 2))[0]
    # _xy_to_px returns (col, row); ObstacleMap indexes _map[px[:,1], px[:,0]].
    row, col = int(px[1]), int(px[0])
    size = base_map._map.shape[0]
    row = min(max(row, 0), size - 1)
    col = min(max(col, 0), size - 1)
    return row, col


def rc_to_xy(base_map: Any, rc: RC) -> np.ndarray:
    """Array ``(row, col)`` index -> world ``(x, y)`` [meters]."""
    # _px_to_xy expects px as (col, row), the inverse of _xy_to_px.
    px = np.array([[rc[1], rc[0]]], dtype=int)
    return base_map._px_to_xy(px)[0]


def snap_to_navigable(navigable: np.ndarray, rc: RC) -> Optional[RC]:
    """Return ``rc`` itself if navigable, else the nearest navigable cell.

    Returns ``None`` only if the whole map is non-navigable.
    """
    size_r, size_c = navigable.shape
    r = min(max(int(rc[0]), 0), size_r - 1)
    c = min(max(int(rc[1]), 0), size_c - 1)
    if navigable[r, c]:
        return r, c
    if not navigable.any():
        return None
    # distance_transform_edt(input) returns, for each foreground (non-zero)
    # cell, the index of the nearest background (zero) cell. With
    # input = ~navigable, "background" == navigable, so we get the nearest
    # navigable pixel to (r, c).
    _, inds = distance_transform_edt(~navigable, return_indices=True)
    return int(inds[0, r, c]), int(inds[1, r, c])


def plan_path(
    navigable: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    obstacle_cost: float = OBSTACLE_COST,
) -> Optional[List[RC]]:
    """A* / min-cost path on the grid from ``start_rc`` to ``goal_rc``.

    Returns a list of ``(row, col)`` cells (start first, goal last), or ``None``
    if no fully-navigable path exists (i.e. the only route cuts through known
    obstacles). Caller is responsible for snapping ``start_rc`` / ``goal_rc`` to
    navigable cells first if desired.
    """
    if start_rc == goal_rc:
        return [start_rc]

    cost = np.where(navigable, 1.0, obstacle_cost).astype(np.float64)
    try:
        indices, _ = route_through_array(
            cost,
            list(start_rc),
            list(goal_rc),
            fully_connected=True,  # 8-connectivity (diagonal moves allowed)
            geometric=True,  # true Euclidean step lengths
        )
    except Exception:
        return None
    if not indices:
        return None

    rows = np.fromiter((p[0] for p in indices), dtype=int, count=len(indices))
    cols = np.fromiter((p[1] for p in indices), dtype=int, count=len(indices))
    # If the route had to pass through any non-navigable cell, the endpoints are
    # not connected through free space -> report "no path" so the caller can
    # fall back (e.g. to frontier exploration).
    if not navigable[rows, cols].all():
        return None

    return [(int(r), int(c)) for r, c in indices]


def downsample_path(path_rc: List[RC], spacing_px: int = 12) -> List[RC]:
    """Thin a dense per-pixel path down to waypoints ~``spacing_px`` apart.

    Always keeps the first and last cell. Uses Chebyshev spacing so diagonal
    runs are not over-sampled.
    """
    if len(path_rc) <= 2:
        return list(path_rc)
    out: List[RC] = [path_rc[0]]
    last = path_rc[0]
    for p in path_rc[1:-1]:
        if max(abs(p[0] - last[0]), abs(p[1] - last[1])) >= spacing_px:
            out.append(p)
            last = p
    out.append(path_rc[-1])
    return out
