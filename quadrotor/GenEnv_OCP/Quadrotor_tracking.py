from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces

from quadrotor.GenEnv_OCP.pyth_base import Env, State
from quadrotor.GenEnv_OCP.robot.quadrotor import Quadrotor
from quadrotor.GenEnv_OCP.context.ref_traj import QuadContext


class QuadTracking(Env):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_acc: float = 3.0,
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        self.robot: Quadrotor = Quadrotor(
            dt=dt,
            max_acc=max_acc,
            max_steer=max_steer,
        )
        self.context: QuadContext = QuadContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
        )
        
        self.state_space = spaces.Box(
            low=np.array([-self.x_threshold, -np.finfo(np.float32).max]), 
            high=np.array([self.x_threshold, np.finfo(np.float32).max]), 
            dtype=np.float32)
        
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.seed()




    def reset(
        self,
    ) -> Tuple[np.ndarray, dict]:
        return Quadrotor.reset()

    def _get_obs(self) -> np.ndarray:
        return Quadrotor._get_obs()

    def _get_reward(self, action: np.ndarray) -> float:
        return Quadrotor._get_reward()

    def _get_terminated(self) -> bool:
        return Quadrotor._get_done()
    
    def _get_info(self) -> dict:
        # return {
        #     **super()._get_info(),
        #     "ref_points": self.context.state.reference.copy(),
        #     "path_num": self.context.state.path_num,
        #     "u_num": self.context.state.speed_num,
        #     "ref_time": self.context.state.ref_time,
        #     "ref": self.context.state.reference[0].copy(),
        # }
        return Quadrotor._get_info()

    def render(self, mode="human"):
        pass

def env_creator(**kwargs):
    return QuadTracking(**kwargs)


if __name__ == "__main__":
    # test consistency with old environment
    import numpy as np
    from quadrotor.GenEnv_OCP.Quadrotor_tracking import Veh3DoFTracking
    env_new = Veh3DoFTracking()
    seed = 1
    env_new.seed(seed)
    np.random.seed(seed)
    obs_new, _ = env_new.reset()
    print("reset obs close:", obs_new)
    action = np.random.random(2)
    next_obs_new, reward_new, done_new, _ = env_new.step(action)
    print("step reward close:",  reward_new)
