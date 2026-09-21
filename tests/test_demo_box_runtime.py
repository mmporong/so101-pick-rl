import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_box_runtime import DemoBoxRuntime


def _runtime(num_envs=2):
    runtime = DemoBoxRuntime.__new__(DemoBoxRuntime)
    runtime.torch = torch
    runtime.num_envs = num_envs
    runtime.device = torch.device("cpu")
    runtime.dt = 1 / 60
    runtime.lower = torch.tensor([-2.0, -1.0, -0.5, -3.0, -4.0, -0.2])
    runtime.upper = torch.tensor([2.0, 3.0, 1.5, 1.0, 4.0, 1.8])
    runtime.previous_target = torch.zeros((num_envs, 6))
    runtime._closed = False

    origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])[:num_envs]
    q = torch.arange(num_envs * 6, dtype=torch.float32).reshape(num_envs, 6) / 10
    dq = -q
    body_pos = torch.zeros((num_envs, 1, 3))
    effort_limits = torch.full((num_envs, 6), 10.0)
    runtime.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=q,
            joint_vel=dq,
            body_pos_w=body_pos,
            joint_effort_limits=effort_limits,
        ),
        write_joint_effort_limit_to_sim=Mock(),
        set_joint_position_target=Mock(),
        set_joint_velocity_target=Mock(),
    )
    cube_local = torch.tensor([0.42, -0.12, 0.08])
    cube_pose = torch.zeros((num_envs, 13))
    cube_pose[:, :3] = origins + cube_local
    cube_pose[:, 3] = 1.0
    cube_pose[:, 7:13] = torch.arange(6, dtype=torch.float32)
    runtime.cube = SimpleNamespace(
        data=SimpleNamespace(
            root_state_w=cube_pose,
            root_pos_w=cube_pose[:, :3],
            default_mass=torch.full((num_envs, 1), 0.25),
        )
    )
    box_local = torch.tensor([0.58, -0.35, 0.0455])
    box_position = origins + box_local
    runtime.box = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=box_position,
            default_mass=torch.full((num_envs, 1), 1.0),
        )
    )
    runtime.scene = SimpleNamespace(
        env_origins=origins,
        rigid_objects={"cube": runtime.cube, "box_target": runtime.box},
        write_data_to_sim=Mock(),
        update=Mock(),
        reset=Mock(),
        reset_to=Mock(),
    )
    runtime.sim = SimpleNamespace(step=Mock(), forward=Mock())
    return runtime


class DemoBoxRuntimeTests(unittest.TestCase):
    def test_observation_order_and_environment_local_positions(self):
        runtime = _runtime()
        runtime.previous_target.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6))

        observation = runtime.observe()

        self.assertEqual(tuple(observation.shape), (2, 34))
        torch.testing.assert_close(observation[:, 0:6], runtime.robot.data.joint_pos)
        torch.testing.assert_close(observation[:, 6:12], runtime.robot.data.joint_vel)
        torch.testing.assert_close(observation[:, 12:18], runtime.previous_target)
        torch.testing.assert_close(observation[:, 18:21], torch.tensor([[0.42, -0.12, 0.08]]).repeat(2, 1))
        torch.testing.assert_close(observation[:, 21], torch.ones(2))
        torch.testing.assert_close(observation[:, 25:31], torch.arange(6, dtype=torch.float32).repeat(2, 1))
        torch.testing.assert_close(
            observation[:, 31:34], torch.tensor([[0.58, -0.35, 0.0455]]).repeat(2, 1)
        )

    def test_step_maps_action_endpoints_and_does_not_reset_state(self):
        runtime = _runtime()
        actions = torch.stack((-torch.ones(6), torch.ones(6)))

        observation = runtime.step(actions)

        target = runtime.robot.set_joint_position_target.call_args.args[0]
        torch.testing.assert_close(target[0], runtime.lower)
        torch.testing.assert_close(target[1], runtime.upper)
        torch.testing.assert_close(runtime.previous_target, target)
        torch.testing.assert_close(observation[:, 12:18], target)
        runtime.scene.reset.assert_not_called()
        runtime.scene.reset_to.assert_not_called()
        runtime.sim.forward.assert_not_called()
        runtime.scene.write_data_to_sim.assert_called_once_with()
        runtime.sim.step.assert_called_once_with(render=False)
        runtime.scene.update.assert_called_once_with(runtime.dt)

    def test_step_rejects_invalid_actions_before_any_scene_write(self):
        cases = (
            torch.zeros((1, 6)),
            torch.full((2, 6), float("nan")),
            torch.full((2, 6), 1.01),
        )
        for actions in cases:
            with self.subTest(shape=tuple(actions.shape)):
                runtime = _runtime()
                with self.assertRaises(ValueError):
                    runtime.step(actions)
                runtime.robot.set_joint_position_target.assert_not_called()
                runtime.robot.write_joint_effort_limit_to_sim.assert_not_called()
                runtime.scene.write_data_to_sim.assert_not_called()
                runtime.sim.step.assert_not_called()

    def test_reset_applies_recorded_previous_target_as_controller_target(self):
        runtime = _runtime()
        previous_target = torch.zeros((2, 6))
        initial_state = {"articulation": {"robot": {"joint_position": torch.zeros((2, 6))}}}

        runtime.reset(initial_state, previous_target)

        runtime.scene.reset_to.assert_called_once()
        runtime.robot.set_joint_position_target.assert_called_once()
        torch.testing.assert_close(runtime.robot.set_joint_position_target.call_args.args[0], previous_target)
        runtime.robot.set_joint_velocity_target.assert_called_once()
        runtime.scene.write_data_to_sim.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
