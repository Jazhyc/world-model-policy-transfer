from typing import cast, Any, TypeVar
import gymnasium
from minigrid.wrappers import RGBImgObsWrapper, ObservationWrapper, ImgObsWrapper
from dreamerv3.embodied.core.wrappers import ResizeImage
from dreamerv3.embodied.envs.from_gymnasium import FromGymnasium

WrapperObsType = TypeVar("WrapperObsType")

class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
        obs_space.spaces.pop('mission')

    def observation(self, observation: dict):
        observation.pop('mission')
        return observation
        


class WrappedMinigrid(FromGymnasium):
    def __init__(self, task: str, fully_observable: bool, hide_mission: bool):
        env = gymnasium.make(f"MiniGrid-{task}-v0", render_mode="rgb_array")
        if fully_observable:
            env = RGBImgObsWrapper(env)
        else:
            env = ImgObsWrapper(env)
        # if hide_mission:
        #     env = HideMission(env)
        super().__init__(env=env)


# also wrap in ResizeImage so that we can handle size kwarg
class Minigrid(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(WrappedMinigrid(*args, **kwargs), size=size)