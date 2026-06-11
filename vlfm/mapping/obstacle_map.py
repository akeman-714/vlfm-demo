# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        super().reset()
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def _recompute_navigable(self) -> None:
        """Rebuild the robot-radius-padded navigable map from the obstacle map."""
        self._navigable_map = 1 - cv2.dilate(
            self._map.astype(np.uint8),
            self._navigable_kernel,
            iterations=1,
        ).astype(bool)

    def save(self, path: Union[str, os.PathLike[str]], start_pose: Optional[Sequence[float]] = None) -> None:
        """Persist the monotonic obstacle grid and a small compatibility header."""
        map_path = Path(path)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = map_path.with_name(f".{map_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            with tmp_path.open("wb") as f:
                np.savez_compressed(
                    f,
                    obstacle_map=self._map.astype(bool),
                    size=np.array([self.size], dtype=np.int64),
                    pixels_per_meter=np.array([self.pixels_per_meter], dtype=np.int64),
                    start_pose=np.asarray(start_pose if start_pose is not None else [], dtype=np.float64),
                    ts=np.array([time.time()], dtype=np.float64),
                )
            os.replace(tmp_path, map_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def load_and_merge(self, path: Union[str, os.PathLike[str]]) -> int:
        """Load a persisted obstacle grid, OR it into this map, and return loaded pixels."""
        map_path = Path(path)
        if not map_path.exists():
            return 0

        try:
            with np.load(map_path, allow_pickle=False) as data:
                loaded_size = int(np.asarray(data["size"]).reshape(-1)[0])
                loaded_ppm = int(np.asarray(data["pixels_per_meter"]).reshape(-1)[0])
                loaded_map = np.asarray(data["obstacle_map"], dtype=bool)
        except (KeyError, OSError, ValueError) as e:
            print(f"[persist] could not read {map_path}: {e}")
            return 0

        if loaded_size != self.size or loaded_ppm != self.pixels_per_meter or loaded_map.shape != self._map.shape:
            print(
                "[persist] skipped incompatible map "
                f"{map_path}: size={loaded_size}, ppm={loaded_ppm}, shape={loaded_map.shape}; "
                f"expected size={self.size}, ppm={self.pixels_per_meter}, shape={self._map.shape}"
            )
            return 0

        loaded_count = int(loaded_map.sum())
        self._map |= loaded_map
        self._recompute_navigable()
        return loaded_count

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._recompute_navigable()

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            # Keep the agent's current component plus any other component large
            # enough to be a real explored region (>= area_thresh). This prevents a
            # momentary false obstacle (e.g. depth glitches when clipping through
            # geometry) from splitting the map and erasing previously-explored rooms.
            # This assumes a single-floor scene, which is what this pipeline targets.
            keep_idxs = {best_idx}
            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) >= self._area_thresh_in_pixels:
                    keep_idxs.add(idx)
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            for idx in keep_idxs:
                cv2.drawContours(new_area, contours, idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
