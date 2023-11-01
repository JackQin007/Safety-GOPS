from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces

from quadrotor.GenEnv_OCP.pyth_base import Env, State
from quadrotor.GenEnv_OCP.robot.quadrotor import Quadrotor
from quadrotor.GenEnv_OCP.context.ref_traj import QuadContext


class QuadTracking(Env):
    def __init__(
        self,       
        **kwargs,
    ):
        self.robot: Quadrotor = Quadrotor(

        )
        self.context: QuadContext = QuadContext(
           
        )
        self.state_space = self.robot.state_space
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.seed()

    def reset(
        self,
    ) -> Tuple[np.ndarray, dict]:
        return self.robot.reset()

    def _get_obs(self) -> np.ndarray:
        return self.robot._get_obs()

    def _get_reward(self, action: np.ndarray) -> float:
        return self.robot._get_reward()

    def _get_terminated(self) -> bool:
        return self.robot._get_done()
    
    def _get_info(self) -> dict:
        return self.robot._get_info()

    def render(self, mode="human"):
        pass

def env_creator(**kwargs):
    return QuadTracking(**kwargs)


if __name__ == "__main__":
    # test consistency with old environment
    import numpy as np
    env_new = QuadTracking()
    seed = 1
    env_new.seed(seed)
    np.random.seed(seed)
    obs_new = env_new.reset()
    print("reset obs close:", obs_new)
    action = np.random.random(2)
    next_obs_new, reward_new, done_new, _ = env_new.step(action)
    print("step reward close:",  reward_new)
