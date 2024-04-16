from typing import Tuple
import gym
from gym.core import Wrapper
from gym.spaces import Box
import numpy as np
import gym_sokoban

from dreamerv3.embodied.core.wrappers import ResizeImage

# gym_sokoban.envs is a list of env IDs that later gets shadowed by a module.
# Thus, this must come before import of gym_sokoban.envs....
gym_sokoban_envs = gym_sokoban.envs
# ...and this must come after access of gym_sokoban.envs.
from gym_sokoban.envs.sokoban_env import SokobanEnv

from dreamerv3.embodied.envs.from_gym import FromGym


class TinyWorldObsWrapper(Wrapper):
    """Forces SokobanEnv.reset()/.step() return TinyWorld obs.

    TinyWorld is their version of Sokoban with a simple semantic color scheme
    and 1 cell = 1 pixel resolution. This makes it possible for us to specify
    goals more easily."""

    def __init__(self, env: SokobanEnv):
        self.env = env
        super().__init__(env=self.env)
        height, width = self.env.dim_room
        self.observation_space = Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        self.observation_mode = "tiny_rgb_array"

    def step(self, *args, **kwargs):
        return self.env.step(*args, observation_mode=self.observation_mode, **kwargs)

    def reset(self, *args, **kwargs):
        # yes, .reset() and .render() use render_mode, while .step() uses
        # observation_mode
        return self.env.reset(*args, render_mode=self.observation_mode, **kwargs)


class EmbodiedSokoban(FromGym):
    def __init__(
        self, task: str, use_tiny_world: bool
    ) -> None:
        env_ids = [env["id"].removeprefix("Sokoban-") for env in gym_sokoban_envs]
        if task not in env_ids:
            raise ValueError(
                f"Unknown Sokoban task: '{task}'. Valid tasks are: {env_ids}"
            )
        env = gym.make(f"Sokoban-{task}")
        if use_tiny_world:
            assert isinstance(env, SokobanEnv), (
                "Expected SokobanEnv, got " f"{type(env)} for task '{task}'"
            )
            env = TinyWorldObsWrapper(env)
        super().__init__(env)

# wrap in ResizeImage to get correct size
class Sokoban(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(EmbodiedSokoban(*args, **kwargs), size=size)
