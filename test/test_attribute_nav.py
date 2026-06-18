# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from types import SimpleNamespace

import numpy as np

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.vlm.attribute_verifier import _cloud_yellow_family_accept, heuristic_parse_instruction, heuristic_verify


class _FakePolicy(BaseObjectNavPolicy):
    def __init__(self) -> None:
        pass


def test_reject_region_removes_points_and_blocks_region() -> None:
    object_map = ObjectPointCloudMap(erosion_size=0)
    object_map.reset()
    object_map.clouds["cat"] = np.array(
        [
            [0.00, 0.00, 0.2, 1.0],
            [0.20, 0.00, 0.2, 1.0],
            [2.00, 0.00, 0.2, 1.0],
        ]
    )
    object_map.last_target_coord = np.array([0.1, 0.0])

    object_map.reject_region("cat", np.array([0.0, 0.0]), radius=0.5)

    assert object_map.clouds["cat"].shape == (1, 4)
    np.testing.assert_allclose(object_map.clouds["cat"][0, :2], [2.0, 0.0])
    assert object_map.last_target_coord is None
    assert object_map._in_rejected("cat", np.array([0.25, 0.0]))
    assert not object_map._in_rejected("cat", np.array([1.0, 0.0]))


def _make_fake_policy(verdict: dict) -> _FakePolicy:
    policy = _FakePolicy()
    policy._verification_enabled = True
    policy._attribute_verified = False
    policy._predicate = "a yellow cat"
    policy._target_object = "cat"
    policy._verify_calls = 0
    policy._last_verify_result = ""
    policy._last_target_crop = np.full((24, 24, 3), [220, 170, 80], dtype=np.uint8)
    policy._last_target_crop_step = 10
    policy._num_steps = 10
    policy._last_goal = np.array([1.0, 2.0])
    policy._called_stop = True
    policy._pointnav_policy = SimpleNamespace(reset=lambda: setattr(policy, "_pointnav_reset", True))
    policy._pointnav_reset = False
    policy._stop_action = "STOP"
    policy._explore = lambda observations: "EXPLORE"
    policy._verifier = SimpleNamespace(verify=lambda crop, predicate, timeout=3.0: verdict)

    rejected = []

    def reject_region(name, xy, radius=0.5):
        rejected.append((name, np.asarray(xy).copy(), radius))

    policy._object_map = SimpleNamespace(reject_region=reject_region)
    policy._rejected = rejected
    return policy


def test_verify_on_arrival_rejects_and_resumes_exploration() -> None:
    policy = _make_fake_policy({"match": False, "reason": "not yellow", "source": "test"})

    action = policy._verify_on_arrival({}, robot_xy=np.zeros(2))

    assert action == "EXPLORE"
    assert policy._called_stop is False
    assert policy._pointnav_reset is True
    assert policy._rejected[0][0] == "cat"
    np.testing.assert_allclose(policy._rejected[0][1], [1.0, 2.0])
    assert "match=False" in policy._last_verify_result


def test_verify_on_arrival_accepts_match_and_keeps_stop() -> None:
    policy = _make_fake_policy({"match": True, "reason": "yellow cat", "source": "test"})

    action = policy._verify_on_arrival({}, robot_xy=np.zeros(2))

    assert action is None
    assert policy._called_stop is True
    assert policy._attribute_verified is True
    assert policy._rejected == []
    assert "match=True" in policy._last_verify_result


def test_stop_guard_blocks_exploration_stop_before_attribute_match() -> None:
    policy = _make_fake_policy({"match": False, "reason": "not yellow", "source": "test"})
    policy._called_stop = False

    mode, action = policy._guard_attribute_stop("explore", "STOP", {}, robot_xy=np.zeros(2))

    assert mode == "verify-guard"
    assert action == "TURN_RIGHT"
    assert policy._called_stop is False
    assert "blocked unverified STOP" in policy._last_verify_result


def test_heuristic_parse_yellow_cat_query() -> None:
    parsed = heuristic_parse_instruction("找黄色猫", default_noun="")

    assert parsed.noun == "cat"
    assert parsed.predicate == "a yellow cat"


def test_heuristic_verify_yellow_crop() -> None:
    crop = np.full((32, 32, 3), [220, 170, 80], dtype=np.uint8)

    verdict = heuristic_verify(crop, "a yellow cat")

    assert verdict.match is True


def test_cloud_yellow_family_accepts_tan_cat_reason() -> None:
    reason = "The cat's fur is a light-brown/tan shade."

    assert _cloud_yellow_family_accept("a yellow cat", reason) is True
    assert _cloud_yellow_family_accept("a yellow cat", "The cat is gray.") is False


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
