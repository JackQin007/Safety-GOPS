from copy import deepcopy
from context.ref_traj import QuadContext
import numpy as np
from gym import spaces

class Quadrotor:
    def __init__(self, prior_prop={},rew_exponential=True, **kwargs ):
        
        self.rew_exponential = rew_exponential
        self.GRAVITY_ACC = 9.81
        self.CTRL_TIMESTEP = 0.01 
        self.TIMESTEP = 0.001  
        self.QUAD_TYPE = "ONE_D"  
        self.state = None
        self.dt = self.TIMESTEP
        self.x_threshold = 2
        self.context = QuadContext()
        self.ctrl_step_counter = 0 
        self.task = self.context.task
        self.GROUND_PLANE_Z = -0.05
        low = np.array([    
                -self.x_threshold, -np.finfo(np.float32).max,
              
            ])
        high = np.array([
                self.x_threshold, np.finfo(np.float32).max,
                
            ])
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                            'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([ 0,-5.0]), high=np.array([0,5.0]), dtype=np.float32)
    def f_xu(self,X,T):
        m = self.context.MASS
        g= self.GRAVITY_ACC
        X_dot = np.array([X[1], T[0] / m - g])  
        return X_dot

    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state
        else:
            self.state = np.array([0.0, 0.0])  # Default initial state
        return self._get_obs()
    
    def step(self, thrust):
        X_dot= self.f_xu(X=self.state,T=thrust)
        self.state += self.dt * X_dot
        self.action = thrust
        obs = self._get_obs()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        self.ctrl_step_counter += 1
        return obs, rew, done, info

    def _get_obs(self):
        return self.state
        
    def _get_reward(self):
        act_error = self.action - self.context.U_GOAL
        # Quadratic costs w.r.t state and action
        # TODO: consider using multiple future goal states for cost in tracking
        if self.task == 'STABILIZATION':
            state_error = self.state - self.context.X_GOAL
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            dist += np.sum(self.context.rew_act_weight * act_error * act_error)
        if self.task == 'TRAJ_TRACKING':
            wp_idx = min(self.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = self.state - self.context.X_GOAL[wp_idx]
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            dist += np.sum(self.context.rew_act_weight * act_error * act_error)
        rew = -dist
        # Convert rew to be positive and bounded [0,1].
        if self.rew_exponential:
            rew = np.exp(rew)
        return rew

    def _get_done(self):
        # Done if goal reached for stabilization task with quadratic cost.
        if self.task == 'STABILIZATION' :
            self.goal_reached = bool(np.linalg.norm(self.state - self.context.X_GOAL) < self.context.TASK_INFO['stabilization_goal_tolerance'])
            if self.goal_reached:
                return True 
        # Done if state is out-of-bounds.
        mask = np.array([1, 0])
        # Element-wise or to check out-of-bound conditions.
        self.out_of_bounds = np.logical_or(self.state < self.state_space.low,
                                        self.state > self.state_space.high)
        # Mask out un-included dimensions (i.e. velocities)
        self.out_of_bounds = np.any(self.out_of_bounds * mask)
        # Early terminate if needed.
        if self.out_of_bounds:
            return True

        return False

    def _get_info(self):
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        if self.task == 'STABILIZATION' :
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        if self.task == 'STABILIZATION':
            state_error = state - self.context.X_GOAL
        elif self.task == 'TRAJ_TRACKING':
            # TODO: should use angle wrapping
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
            state_error = state - self.context.X_GOAL[wp_idx]
        # Filter only relevant dimensions.
        self.info_mse_metric_state_weight = np.array([1, 0], ndmin=1, dtype=float)
        state_error = state_error * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        # if self.constraints is not None:
        #     info['constraint_values'] = self.constraints.get_values(self)
        #     info['constraint_violations'] = self.constraints.get_violations(self)
        return info
 
