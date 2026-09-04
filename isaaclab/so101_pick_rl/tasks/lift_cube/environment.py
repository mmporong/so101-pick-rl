"""Environment lifecycle hooks for per-episode SO-101 task state."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class SO101LiftCubeEnv(ManagerBasedRLEnv):
    """Allocate required task state before managers inspect observation dimensions."""

    def load_managers(self) -> None:
        cube = self.scene["cube"]
        self._so101_cube_initial_z = cube.data.root_pos_w[:, 2].clone()
        super().load_managers()
