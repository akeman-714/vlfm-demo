# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.utils.goal_plan import Goal, GoalQueue, decompose
from vlfm.utils.object_memory import recall_object, remember_object
from vlfm.utils.path_planning import downsample_path, plan_path, rc_to_xy, snap_to_navigable, xy_to_rc
from vlfm.vlm.attribute_verifier import AttributeVerifierClient, heuristic_verify, parse_objectnav_instruction
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


# --- Module 3: global A* navigation (opt-in via VLFM_GLOBAL_NAV=1) ---
# A* plans discrete waypoints on the obstacle map; the trained PointNav policy
# drives between consecutive waypoints with learned local obstacle avoidance
# (see _navigate_global). No geometric controller / stuck-recovery is needed.
# Waypoint-reached tolerance (meters). Kept above the 0.25m forward step so we
# advance to the next waypoint instead of oscillating around it.
_NAV_ARRIVE_RADIUS = 0.4
# Spacing (pixels @ 20px/m -> 0.6m) used to thin the dense A* path into waypoints.
_NAV_WAYPOINT_SPACING_PX = 12
# Replan if the goal (e.g. an object whose point cloud keeps shifting) drifts more
# than this many meters from the goal we last planned for.
_NAV_GOAL_DRIFT_M = 0.5
# Periodic safety replan (in steps) to absorb newly observed obstacles/free space.
_NAV_REPLAN_PERIOD = 25


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self._verifier = AttributeVerifierClient(port=int(os.environ.get("ATTR_VERIFIER_PORT", "12186")))

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._remembered_goal: Union[np.ndarray, None] = None
        self._memory_recall_attempted = False
        self._memory_written: Set[str] = set()
        self._last_persist_save_step = -1
        # Module 4: ordered multi-goal plan (opt-in via VLFM_GOAL_SEQUENCE).
        self._goal_queue: Union[GoalQueue, None] = None
        self._multi_goal = False
        self._done_initializing = False
        self._called_stop = False
        # Module 3: global-navigation state (lazily planned by _navigate_global).
        self._global_path: Union[List[np.ndarray], None] = None
        self._path_goal: Union[np.ndarray, None] = None
        self._waypoint_idx = 0
        self._last_plan_step = 0
        self._instruction = ""
        self._predicate = ""
        self._predicate_parse_reason = ""
        self._verification_enabled = False
        self._attribute_verified = False
        self._verify_calls = 0
        self._last_verify_result = ""
        self._last_target_crop: Optional[np.ndarray] = None
        self._last_target_crop_step = -1
        self._last_target_bbox: Optional[np.ndarray] = None
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._remembered_goal = None
        self._memory_recall_attempted = False
        self._memory_written = set()
        self._last_persist_save_step = -1
        self._goal_queue = None
        self._multi_goal = False
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        self._global_path = None
        self._path_goal = None
        self._waypoint_idx = 0
        self._last_plan_step = 0
        self._instruction = ""
        self._predicate = ""
        self._predicate_parse_reason = ""
        self._verification_enabled = False
        self._attribute_verified = False
        self._verify_calls = 0
        self._last_verify_result = ""
        self._last_target_crop = None
        self._last_target_crop_step = -1
        self._last_target_bbox = None
        if self._compute_frontiers:
            self._obstacle_map.reset()
            self._load_persistent_obstacles()
        self._did_reset = True

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        use_global = os.environ.get("VLFM_GLOBAL_NAV") == "1"
        debug_goal = self._debug_nav_goal() if use_global else None

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif self._multi_goal:  # Module 4: ordered multi-goal plan
            mode, pointnav_action = self._act_multi_goal(observations, robot_xy)
        elif debug_goal is not None:  # Verification hook: drive to a fixed goal via A*
            mode = "navigate-debug"
            debug_conservative = os.environ.get("VLFM_NAV_DEBUG_CONSERVATIVE", "1") == "1"
            pointnav_action = self._navigate_to(debug_goal, observations, conservative=debug_conservative)
        elif goal is not None:  # Found the target object -> go to it
            mode = "navigate"
            pointnav_action = self._navigate_to(goal[:2], observations, conservative=False)
            if self._called_stop:
                verified_action = self._verify_on_arrival(observations, robot_xy)
                if verified_action is not None:
                    mode = "verify-reject"
                    pointnav_action = verified_action
        elif use_global and self._remembered_goal is not None:  # Recalled from memory (Module 2)
            mode = "navigate-memory"
            memory_conservative = os.environ.get("VLFM_MEMORY_NAV_CONSERVATIVE", "1") == "1"
            pointnav_action = self._navigate_to(
                self._remembered_goal,
                observations,
                conservative=memory_conservative,
                fallback_to_pointnav=True,
            )
        else:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)

        mode, pointnav_action = self._guard_attribute_stop(mode, pointnav_action, observations, robot_xy)
        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        if not self._multi_goal:
            self._maybe_remember_object(goal)
        self._maybe_save_persistent_obstacles()
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            seq = os.environ.get("VLFM_GOAL_SEQUENCE")
            if seq and seq.strip():
                self._goal_queue = GoalQueue(decompose(seq))
                self._multi_goal = len(self._goal_queue) > 0
            if self._multi_goal:
                self._apply_current_goal()
            else:
                self._target_object = observations["objectgoal"]
                self._configure_attribute_query(str(observations["objectgoal"]))
                self._maybe_recall_object_memory()
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _configure_attribute_query(self, default_noun: str) -> None:
        query = (
            os.environ.get("VLFM_OBJECTNAV_QUERY")
            or os.environ.get("VLFM_ATTR_QUERY")
            or os.environ.get("VLFM_NAV_QUERY")
            or ""
        ).strip()
        direct_predicate = os.environ.get("VLFM_ATTR_PREDICATE", "").strip()
        direct_noun = os.environ.get("VLFM_ATTR_NOUN", "").strip()

        if direct_predicate:
            self._instruction = query or direct_predicate
            self._predicate = direct_predicate
            self._target_object = direct_noun or default_noun
            self._predicate_parse_reason = "env"
        elif query:
            parsed = parse_objectnav_instruction(
                query,
                default_noun=direct_noun or default_noun,
                timeout=float(os.environ.get("VLFM_ATTR_PARSE_TIMEOUT", "8")),
            )
            self._instruction = parsed.original
            self._predicate = parsed.predicate
            self._predicate_parse_reason = parsed.reason
            if parsed.noun:
                self._target_object = parsed.noun
        else:
            self._instruction = ""
            self._predicate = ""
            self._predicate_parse_reason = ""

        self._verification_enabled = bool(self._predicate) and os.environ.get("VLFM_ATTR_VERIFY", "1") != "0"
        if self._verification_enabled:
            print(
                f"[attr] query={self._instruction!r} noun={self._target_object!r} "
                f"predicate={self._predicate!r} parse={self._predicate_parse_reason}",
                flush=True,
            )

    def _get_value_target_text(self) -> str:
        if self._predicate and os.environ.get("VLFM_ATTR_USE_VALUE_TEXT", "0") == "1":
            predicate = self._predicate.strip()
            lower = predicate.lower()
            if lower.startswith("a "):
                return predicate[2:].strip()
            if lower.startswith("an "):
                return predicate[3:].strip()
            return predicate
        return self._target_object

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _maybe_recall_object_memory(self) -> None:
        if self._memory_recall_attempted:
            return
        self._memory_recall_attempted = True

        path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
        if not path or not self._target_object:
            return

        remembered_goal = recall_object(path, self._target_object)
        if remembered_goal is None:
            return

        self._remembered_goal = remembered_goal
        print(f"[memory] recalled {self._target_object} at {remembered_goal.tolist()} from {path}")

    def _maybe_remember_object(self, goal: Union[None, np.ndarray]) -> None:
        path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
        if (
            not path
            or not self._called_stop
            or goal is None
            or not self._target_object
            or self._target_object in self._memory_written
        ):
            return

        remember_object(path, self._target_object, goal[:2], start_pose=self._get_memory_start_pose())
        self._memory_written.add(self._target_object)
        print(f"[memory] remembered {self._target_object} at {goal[:2].tolist()} in {path}")

    def _should_verify_on_arrival(self) -> bool:
        return self._verification_enabled and bool(self._predicate) and bool(self._target_object)

    def _verify_on_arrival(self, observations: "TensorDict", robot_xy: np.ndarray) -> Union[None, Tensor]:
        if not self._should_verify_on_arrival():
            return None

        max_calls = int(os.environ.get("VLFM_ATTR_MAX_VERIFY_CALLS", "5"))
        if self._verify_calls >= max_calls:
            self._last_verify_result = f"verify skipped: max calls {max_calls} reached"
            print(f"[attr] {self._last_verify_result}", flush=True)
            return None

        crop = self._get_arrival_crop()
        if crop is None:
            self._last_verify_result = "verify skipped: no target crop"
            print(f"[attr] {self._last_verify_result}", flush=True)
            if os.environ.get("VLFM_ATTR_FAIL_OPEN", "1") == "1":
                return None
            return self._reject_and_continue(observations, robot_xy, reason=self._last_verify_result)

        self._verify_calls += 1
        timeout = float(os.environ.get("VLFM_ATTR_VERIFY_TIMEOUT", "3.0"))
        verdict = self._verifier.verify(crop, self._predicate, timeout=timeout)
        if verdict is None:
            verdict = heuristic_verify(crop, self._predicate).to_json()

        match = bool(verdict.get("match"))
        source = str(verdict.get("source", "unknown"))
        reason = str(verdict.get("reason", "")).strip()
        self._last_verify_result = f"verify[{source}] match={match}: {reason}"
        print(f"[attr] {self._last_verify_result}", flush=True)
        if match:
            self._attribute_verified = True
            return None
        return self._reject_and_continue(observations, robot_xy, reason=self._last_verify_result)

    def _guard_attribute_stop(
        self,
        mode: str,
        action: Tensor,
        observations: "TensorDict",
        robot_xy: np.ndarray,
    ) -> Tuple[str, Tensor]:
        if not self._should_verify_on_arrival() or self._attribute_verified or not self._is_stop_action(action):
            return mode, action

        if self._called_stop:
            verified_action = self._verify_on_arrival(observations, robot_xy)
            if verified_action is None:
                return mode, action
            if self._is_stop_action(verified_action):
                return self._block_unverified_stop(mode)
            return "verify-reject", verified_action

        # _called_stop is False here: the STOP came from exploration giving up (no
        # frontiers / re-inspection exhausted), not from arriving at a target. Let it
        # terminate the episode instead of rewriting it into an in-place TURN_RIGHT,
        # which would livelock (spin without translating) until the step budget runs out.
        return mode, action

    def _is_stop_action(self, action: Any) -> bool:
        if isinstance(action, torch.Tensor) and isinstance(self._stop_action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
            stop_np = self._stop_action.detach().cpu().numpy()
            return action_np.shape == stop_np.shape and np.array_equal(action_np, stop_np)
        return action == self._stop_action

    def _block_unverified_stop(self, mode: str) -> Tuple[str, Tensor]:
        self._called_stop = False
        self._last_verify_result = f"verify guard blocked unverified STOP from mode={mode}"
        print(f"[attr] {self._last_verify_result}", flush=True)
        return "verify-guard", self._turn_right_action()

    def _turn_right_action(self) -> Tensor:
        if isinstance(self._stop_action, torch.Tensor):
            action = torch.zeros_like(self._stop_action)
            action[..., 0] = 3
            return action
        return "TURN_RIGHT"

    def _get_arrival_crop(self) -> Optional[np.ndarray]:
        if self._last_target_crop is None:
            return None
        max_age = int(os.environ.get("VLFM_ATTR_CROP_MAX_AGE", "80"))
        if self._num_steps - self._last_target_crop_step > max_age:
            return None
        return self._last_target_crop

    def _reject_and_continue(
        self,
        observations: "TensorDict",
        robot_xy: np.ndarray,
        reason: str = "",
    ) -> Tensor:
        reject_xy = self._last_goal if not np.array_equal(self._last_goal, np.zeros(2)) else robot_xy
        radius = float(os.environ.get("VLFM_ATTR_REJECT_RADIUS", "0.6"))
        self._object_map.reject_region(self._target_object, reject_xy, radius=radius)
        print(
            f"[attr] reject {self._target_object!r} around {np.round(reject_xy, 3).tolist()} "
            f"r={radius:.2f} reason={reason}",
            flush=True,
        )
        self._reset_per_goal_nav()
        return self._explore(observations)

    def _act_multi_goal(self, observations: "TensorDict", robot_xy: np.ndarray) -> Tuple[str, Tensor]:
        """Module 4: drive toward the current goal in the ordered plan. When a goal
        is reached (``_called_stop``), record it (objects only) and advance to the
        next goal, re-dispatching within the same step so we emit a moving action
        instead of terminating -- the episode STOPs only after the last goal."""
        for _ in range(len(self._goal_queue) + 1):
            g = self._goal_queue.current()
            mode, action = self._dispatch_current_goal(observations, robot_xy, g)
            if not self._called_stop:
                return mode, action
            # Reached the current goal.
            self._on_subgoal_reached(g, robot_xy)
            if not self._goal_queue.advance():
                print("[goal] all goals complete -> STOP", flush=True)
                return mode, action  # _called_stop stays True -> episode terminates
            self._called_stop = False  # reaching a sub-goal is not an episode stop
            self._reset_per_goal_nav()
            self._apply_current_goal()
            print(f"[goal] advance -> #{self._goal_queue.index} {self._goal_queue.current()}", flush=True)
        # Bounded loop exhausted (should not happen) -> stop defensively.
        self._called_stop = True
        return "goal-exhausted", self._stop_action

    def _dispatch_current_goal(
        self, observations: "TensorDict", robot_xy: np.ndarray, g: Goal
    ) -> Tuple[str, Tensor]:
        """One goal -> one action, mirroring the single-goal branch order: point ->
        conservative A*; detected object -> go to it; known (remembered) object ->
        A* to memory; unknown object -> explore."""
        idx = self._goal_queue.index
        if g.kind == "point":
            return f"goal{idx}-point", self._navigate_to(g.xy, observations, conservative=True)
        detected = self._get_target_object_location(robot_xy)
        if detected is not None:
            return f"goal{idx}-navigate", self._navigate_to(detected[:2], observations, conservative=False)
        if self._remembered_goal is not None:
            mem_conservative = os.environ.get("VLFM_MEMORY_NAV_CONSERVATIVE", "1") == "1"
            return f"goal{idx}-memory", self._navigate_to(
                self._remembered_goal, observations, conservative=mem_conservative, fallback_to_pointnav=True
            )
        return f"goal{idx}-explore", self._explore(observations)

    def _on_subgoal_reached(self, g: Goal, robot_xy: np.ndarray) -> None:
        """Persist a freshly reached object's location to memory (once per object)."""
        if g.kind != "object":
            return
        path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
        if not path or g.name in self._memory_written:
            return
        loc = self._get_target_object_location(robot_xy)
        if loc is None:
            return
        remember_object(path, g.name, loc[:2], start_pose=self._get_memory_start_pose())
        self._memory_written.add(g.name)
        print(f"[memory] remembered {g.name} at {loc[:2].tolist()} in {path}", flush=True)

    def _apply_current_goal(self) -> None:
        """Point the perception/navigation stack at the current goal: set
        ``_target_object`` (empty for point goals) and recall its memory."""
        if self._goal_queue is None:
            return
        g = self._goal_queue.current()
        if g is None:
            return
        if g.kind == "object":
            self._target_object = g.name
            path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
            self._remembered_goal = recall_object(path, g.name) if path else None
            tag = "known" if self._remembered_goal is not None else "unknown"
            print(f"[goal] #{self._goal_queue.index} object '{g.name}' ({tag})", flush=True)
        else:
            self._target_object = ""
            self._remembered_goal = None
            print(f"[goal] #{self._goal_queue.index} point {np.round(g.xy, 2).tolist()}", flush=True)

    def _reset_per_goal_nav(self) -> None:
        """Clear per-goal navigation state when switching goals (keeps the shared
        object map and obstacle map intact)."""
        self._global_path = None
        self._path_goal = None
        self._waypoint_idx = 0
        self._last_plan_step = self._num_steps
        self._last_goal = np.zeros(2)
        self._called_stop = False
        self._pointnav_policy.reset()

    def _load_persistent_obstacles(self) -> None:
        path = os.environ.get("VLFM_PERSIST_MAP_PATH")
        if not path or not self._compute_frontiers:
            return
        loaded_px = self._obstacle_map.load_and_merge(path)
        if loaded_px > 0:
            print(f"[persist] loaded {loaded_px} obstacle px from {path}")

    def _maybe_save_persistent_obstacles(self) -> None:
        path = os.environ.get("VLFM_PERSIST_MAP_PATH")
        if not path or not self._compute_frontiers:
            return
        if not self._called_stop and self._num_steps % int(os.environ.get("VLFM_PERSIST_SAVE_PERIOD", "10")) != 0:
            return
        if self._last_persist_save_step == self._num_steps:
            return
        self._obstacle_map.save(path, start_pose=self._get_memory_start_pose())
        self._last_persist_save_step = self._num_steps
        print(f"[persist] saved {int(self._obstacle_map._map.sum())} obstacle px to {path}")

    def _get_memory_start_pose(self) -> List[float]:
        start_yaw = self._observations_cache.get(
            "habitat_start_yaw",
            self._observations_cache.get("robot_heading", 0.0),
        )
        return [0.0, 0.0, float(start_yaw)]

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }
        if self._instruction:
            policy_info["instruction"] = f"instruction: {self._instruction}"
            policy_info["render_below_images"].append("instruction")
        if self._predicate:
            policy_info["attribute_predicate"] = f"predicate: {self._predicate}"
            policy_info["render_below_images"].append("attribute_predicate")
        if self._last_verify_result:
            policy_info["attribute_verify"] = self._last_verify_result
            policy_info["render_below_images"].append("attribute_verify")

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _debug_nav_goal(self) -> Union[np.ndarray, None]:
        """Verification hook (Module 3). If ``VLFM_NAV_DEBUG_GOAL="x,y"`` is set,
        return that episodic goal -- but only once ``self._num_steps`` reaches
        ``VLFM_NAV_DEBUG_AFTER`` (default 0), so the robot can first explore away
        from the origin before being driven (back) to the goal."""
        raw = os.environ.get("VLFM_NAV_DEBUG_GOAL")
        if not raw:
            return None
        if self._num_steps < int(os.environ.get("VLFM_NAV_DEBUG_AFTER", "0")):
            return None
        try:
            x, y = (float(v) for v in raw.split(",")[:2])
        except ValueError:
            print(f"[nav] ignoring malformed VLFM_NAV_DEBUG_GOAL={raw!r}")
            return None
        return np.array([x, y], dtype=np.float64)

    def _navigate_to(
        self,
        goal_xy: np.ndarray,
        observations: "TensorDict",
        conservative: bool = False,
        fallback_to_pointnav: bool = False,
    ) -> Tensor:
        """Navigate toward ``goal_xy``. With ``VLFM_GLOBAL_NAV=1`` this plans an
        A* path on the obstacle map and follows it; otherwise it preserves the
        original ``_pointnav(goal, stop=True)`` behavior. If global planning finds
        no navigable path, falls back to frontier exploration."""
        goal_xy = np.asarray(goal_xy, dtype=np.float64)[:2]
        if os.environ.get("VLFM_GLOBAL_NAV") != "1":
            return self._pointnav(goal_xy, stop=True)
        action = self._navigate_global(goal_xy, conservative=conservative)
        if action is not None:
            return action
        if fallback_to_pointnav:
            print("[nav] no global path found, falling back to PointNav toward remembered goal")
            return self._pointnav(goal_xy, stop=True)
        # A* found no navigable path (truly walled in) -> fall back to exploring.
        print("[nav] no global path found, falling back to exploration")
        return self._explore(observations)

    def _navigate_global(self, goal_xy: np.ndarray, conservative: bool = False) -> Union[Tensor, None]:
        """Plan (and cache) an A* path to ``goal_xy`` on the obstacle map and drive
        toward the next waypoint with the trained PointNav policy (which provides
        learned local obstacle avoidance between waypoints).

        Args:
            goal_xy: Episodic ``(x, y)`` goal in meters.
            conservative: If True, only traverse already-explored free space
                (``_navigable_map & explored_area``) -- used for returning to a
                remembered/home point. If False, treat unseen cells as navigable
                (optimistic) -- used for charging a freshly detected object.

        Returns:
            A discrete action tensor, the STOP action once within
            ``_pointnav_stop_radius`` of the goal, or ``None`` if no navigable
            path exists (caller should fall back to exploration).
        """
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]

        # Stop once we are close enough to the actual goal.
        rho_goal, _ = rho_theta(robot_xy, heading, goal_xy)
        if rho_goal < self._pointnav_stop_radius:
            self._called_stop = True
            return self._stop_action

        navigable = self._obstacle_map._navigable_map.astype(bool)
        if conservative:
            navigable = navigable & self._obstacle_map.explored_area.astype(bool)

        if self._needs_replan(goal_xy, navigable):
            path_xy = self._plan_path_xy(robot_xy, goal_xy, navigable)
            if path_xy is None:
                self._global_path = None
                return None
            self._global_path = path_xy
            self._path_goal = goal_xy
            self._waypoint_idx = 0
            self._last_plan_step = self._num_steps

        # Skip past any waypoints we have already reached.
        while self._waypoint_idx < len(self._global_path):
            if rho_theta(robot_xy, heading, self._global_path[self._waypoint_idx])[0] < _NAV_ARRIVE_RADIUS:
                self._waypoint_idx += 1
            else:
                break

        if self._waypoint_idx >= len(self._global_path):
            # Reached the end of the path but still outside the stop radius (the
            # goal was snapped onto/just outside an obstacle). Best effort: stop.
            self._called_stop = True
            return self._stop_action

        next_wp = self._global_path[self._waypoint_idx]
        if os.environ.get("VLFM_NAV_DEBUG_LOG") == "1":
            rho_wp, theta_wp = rho_theta(robot_xy, heading, next_wp)
            print(
                "[nav] "
                f"xy={np.round(robot_xy, 3).tolist()} "
                f"goal={np.round(goal_xy, 3).tolist()} "
                f"wp_idx={self._waypoint_idx}/{len(self._global_path)} "
                f"wp={np.round(next_wp, 3).tolist()} "
                f"rho_wp={rho_wp:.3f} theta_wp={theta_wp:.3f} rho_goal={rho_goal:.3f}",
                flush=True,
            )
        # A* sets the intermediate waypoint; the trained PointNav policy drives to
        # it with learned local obstacle avoidance. stop=False so it never emits
        # STOP at an intermediate waypoint -- the final STOP is the rho_goal check
        # above. _pointnav owns _last_goal, so its RNN resets only when the
        # waypoint actually advances (not every step).
        return self._pointnav(next_wp, stop=False)

    def _needs_replan(self, goal_xy: np.ndarray, navigable: np.ndarray) -> bool:
        """Decide whether to recompute the global path this step (vs. keep following
        the cached one)."""
        if self._global_path is None or self._path_goal is None:
            return True
        if self._waypoint_idx >= len(self._global_path):
            return True
        if np.linalg.norm(goal_xy - self._path_goal) > _NAV_GOAL_DRIFT_M:
            return True
        if self._num_steps - self._last_plan_step >= _NAV_REPLAN_PERIOD:
            return True
        # The waypoint we are currently steering toward turned non-navigable under
        # the latest observation (a newly seen obstacle).
        r, c = xy_to_rc(self._obstacle_map, self._global_path[self._waypoint_idx])
        return not navigable[r, c]

    def _plan_path_xy(
        self, robot_xy: np.ndarray, goal_xy: np.ndarray, navigable: np.ndarray
    ) -> Union[List[np.ndarray], None]:
        """Plan start->goal on ``navigable``, returning a thinned list of waypoint
        ``(x, y)`` points, or ``None`` if no navigable path exists."""
        if not navigable.any():
            return None
        start_rc = snap_to_navigable(navigable, xy_to_rc(self._obstacle_map, robot_xy))
        goal_rc = snap_to_navigable(navigable, xy_to_rc(self._obstacle_map, goal_xy))
        if start_rc is None or goal_rc is None:
            return None
        path_rc = plan_path(navigable, start_rc, goal_rc)
        if path_rc is None:
            return None
        path_rc = downsample_path(path_rc, _NAV_WAYPOINT_SPACING_PX)
        return [rc_to_xy(self._obstacle_map, rc) for rc in path_rc]

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1
            self._cache_target_crop(rgb, bbox_denorm, object_mask)
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_target_crop(self, rgb: np.ndarray, bbox_xyxy: np.ndarray, object_mask: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float64)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        margin = float(os.environ.get("VLFM_ATTR_CROP_MARGIN", "0.15"))
        x1 = int(max(0, np.floor(x1 - margin * box_w)))
        y1 = int(max(0, np.floor(y1 - margin * box_h)))
        x2 = int(min(width, np.ceil(x2 + margin * box_w)))
        y2 = int(min(height, np.ceil(y2 + margin * box_h)))
        if x2 <= x1 or y2 <= y1:
            return

        crop = rgb[y1:y2, x1:x2].copy()
        if os.environ.get("VLFM_ATTR_MASK_CROP", "0") == "1":
            mask_crop = object_mask[y1:y2, x1:x2] > 0
            crop = np.where(mask_crop[..., None], crop, 255).astype(np.uint8)
        self._last_target_crop = crop
        self._last_target_crop_step = self._num_steps
        self._last_target_bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
