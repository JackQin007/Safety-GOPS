from copy import deepcopy
from quadrotor.GenEnv_OCP.context.ref_traj import QuadContext
import numpy as np
from gym import spaces
from enum import IntEnum
from enum import Enum
import xml.etree.ElementTree as etxml
import math


NAME = 'quadrotor'

INERTIAL_PROP_RAND_INFO = {
    'M': {  # Nominal: 0.027
        'distrib': 'uniform',
        'low': 0.022,
        'high': 0.032
    },
    'Ixx': {  # Nominal: 1.4e-5
        'distrib': 'uniform',
        'low': 1.3e-5,
        'high': 1.5e-5
    },
    'Iyy': {  # Nominal: 1.4e-5
        'distrib': 'uniform',
        'low': 1.3e-5,
        'high': 1.5e-5
    },
    'Izz': {  # Nominal: 2.17e-5
        'distrib': 'uniform',
        'low': 2.07e-5,
        'high': 2.27e-5
    }
}

INIT_STATE_RAND_INFO = {
    'init_x': {
        'distrib': 'uniform',
        'low': -0.5,
        'high': 0.5
    },
    'init_x_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_y': {
        'distrib': 'uniform',
        'low': -0.5,
        'high': 0.5
    },
    'init_y_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_z': {
        'distrib': 'uniform',
        'low': 0.1,
        'high': 1.5
    },
    'init_z_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_phi': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_theta': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_psi': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_p': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_theta_dot': {  # TODO: replace with q.
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_q': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_r': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    }
}

class Cost(str, Enum):
    '''Reward/cost functions enumeration class.'''
    RL_REWARD = 'rl_reward'  # Default RL reward function.
    QUADRATIC = 'quadratic'  # Quadratic cost.


class Task(str, Enum):
    '''Environment tasks enumeration class.'''
    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.

class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.

class Quadrotor():
    def __init__(self, 
                 prior_prop={},
                 obs_goal_horizon = 0,
                 rew_exponential=True,  
                 quad_type = QuadType.THREE_D,
                 init_state=None,
                 task: Task = Task.STABILIZATION,
                 cost: Cost = Cost.RL_REWARD,
                 info_mse_metric_state_weight=None,
                 **kwargs ):
      
        self.obs_goal_horizon = obs_goal_horizon
        self.init_state = init_state
        self.QUAD_TYPE = QuadType(quad_type)
        self.L = 1.
        self.rew_exponential = rew_exponential
        self.GRAVITY_ACC = 9.81
        self.CTRL_TIMESTEP = 0.01 
        self.TIMESTEP = 0.001  
        self.state = None
        self.dt = self.TIMESTEP
        self.x_threshold = 2
        self.context = QuadContext()
        self.ctrl_step_counter = 0 
        self.task = self.context.task
        self.GROUND_PLANE_Z = -0.05
       
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                            'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        self.INIT_STATE_LABELS = {
            QuadType.ONE_D: ['init_x', 'init_x_dot'],
            QuadType.TWO_D: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
            QuadType.THREE_D: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                               'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_r']
        }
        self.TASK = Task(task)
        self.COST = Cost(cost)
        if info_mse_metric_state_weight is None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.info_mse_metric_state_weight = np.array([1, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), not implemented quad type.')
        else:
            if (self.QUAD_TYPE == QuadType.ONE_D and len(info_mse_metric_state_weight) == 2) or \
                    (self.QUAD_TYPE == QuadType.TWO_D and len(info_mse_metric_state_weight) == 6) or \
                    (self.QUAD_TYPE == QuadType.THREE_D and len(info_mse_metric_state_weight) == 12):
                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.')
        
        # Concatenate reference for RL.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # Include future goal state(s).
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = np.concatenate([low] * mul)
            high = np.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            low = np.concatenate([low] * 2)
            high = np.concatenate([high] * 2)
        # Define obs space exposed to the controller.
        # Note how the obs space can differ from state space (i.e. augmented with the next reference states for RL)
        self.URDF_PATH = '/home/qinshentao/code/Safety-GOPS/quadrotor/model/assets/cf2x.urdf'
        self.MASS, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3, \
        self.PWM2RPM_SCALE, \
        self.PWM2RPM_CONST, \
        self.MIN_PWM, \
        self.MAX_PWM = self._parse_urdf_parameters(self.URDF_PATH)
        self.Iyy = prior_prop.get('Iyy', self.J[1, 1])
        self.Ixx = prior_prop.get('Ixx', self.J[0, 0])
        self.Izz = prior_prop.get('Izz', self.J[2, 2])
        if self.QUAD_TYPE == QuadType.ONE_D:
            self.state_dim, self.action_dim = 2, 1
        elif self.QUAD_TYPE == QuadType.TWO_D:
            #X = np.cat((x, x_dot, z, z_dot, theta, theta_dot), dim=0)
            #U = np.cat((T1, T2))
            self.state_dim, self.action_dim = 6, 2
        elif self.QUAD_TYPE == QuadType.THREE_D:
            self.state_dim, self.action_dim = 12, 4
        self.NORMALIZED_RL_ACTION_SPACE = True
     

        self._set_action_space()
        self._set_observation_space()
    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM
     
    def _set_observation_space(self):
        '''Sets the observation space of the environment.'''
        self.x_threshold = 2
        self.y_threshold = 2
        self.z_threshold = 2
        self.phi_threshold_radians = 85 * math.pi / 180
        self.theta_threshold_radians = 85 * math.pi / 180
        self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        # Define obs/state bounds, labels and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # obs/state = {z, z_dot}.
            low = np.array([self.GROUND_PLANE_Z, -np.finfo(np.float32).max])
            high = np.array([self.z_threshold, np.finfo(np.float32).max])
            self.STATE_LABELS = ['z', 'z_dot']
            self.STATE_UNITS = ['m', 'm/s']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
            low = np.array([
                -self.x_threshold, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
                -self.theta_threshold_radians, -np.finfo(np.float32).max
            ])
            high = np.array([
                self.x_threshold, np.finfo(np.float32).max,
                self.z_threshold, np.finfo(np.float32).max,
                self.theta_threshold_radians, np.finfo(np.float32).max
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad', 'rad/s']
        elif self.QUAD_TYPE == QuadType.THREE_D:
            # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            low = np.array([
                -self.x_threshold, -np.finfo(np.float32).max,
                -self.y_threshold, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
                -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
                -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
            ])
            high = np.array([
                self.x_threshold, np.finfo(np.float32).max,
                self.y_threshold, np.finfo(np.float32).max,
                self.z_threshold, np.finfo(np.float32).max,
                self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
                np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                'phi', 'theta', 'psi', 'p', 'q', 'r']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        # Define action/input dimension, labels, and units.
        # import ipdb ; ipdb.set_trace()
        if self.QUAD_TYPE == QuadType.ONE_D:
            action_dim = 1
            self.ACTION_LABELS = ['T']
            self.ACTION_UNITS = ['N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            action_dim = 2
            self.ACTION_LABELS = ['T1', 'T2']
            self.ACTION_UNITS = ['N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D:
            action_dim = 4
            self.ACTION_LABELS = ['T1', 'T2', 'T3', 'T4']
            self.ACTION_UNITS = ['N', 'N', 'N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']

        n_mot = 4 / action_dim
        a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
        a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
        self.physical_action_bounds = (np.full(action_dim, a_low, np.float32),
                                       np.full(action_dim, a_high, np.float32))

        if self.NORMALIZED_RL_ACTION_SPACE:
            # Normalized thrust (around hover thrust).
            self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim
            self.action_space = spaces.Box(low=-np.ones(action_dim),
                                           high=np.ones(action_dim),
                                           dtype=np.float32)
        else:
            # Direct thrust control.
            self.action_space = spaces.Box(low=self.physical_action_bounds[0],
                                           high=self.physical_action_bounds[1],
                                           dtype=np.float32)

    def f_xu(self,X,U):
        m = self.context.MASS
        g= self.GRAVITY_ACC
        u_eq = m * g
        if self.QUAD_TYPE == QuadType.ONE_D:
            self.state_dim, self.action_dim = 2, 1
            X_dot = np.array([X[1], U[0] / m - g])  
            return X_dot
        # Add other cases for QUAD_TYPE (TWO_D, THREE_D) as needed
        elif self.QUAD_TYPE == QuadType.TWO_D:
            #X = np.cat((x, x_dot, z, z_dot, theta, theta_dot), dim=0)
            #U = np.cat((T1, T2))
            self.state_dim, self.action_dim = 6, 2
            X_dot = np.array([X[1],np.sin(X[4]) * (U[0] + U[1]) / m,X[3],np.cos(X[4]) * (U[0] + U[1]) / m - g,
                              X[-1],self.L * (U[1] - U[0]) / self.Iyy / np.sqrt(2.0)])
            return X_dot
        elif self.QUAD_TYPE == QuadType.THREE_D:
            self.state_dim, self.action_dim = 12, 4
            J = np.array([[self.Ixx, 0.0, 0.0],
                            [0.0, self.Iyy, 0.0],
                            [0.0, 0.0, self.Izz]])
            Jinv = np.array([[1.0 / self.Ixx, 0.0, 0.0],
                                [0.0, 1.0 / self.Iyy, 0.0],
                                [0.0, 0.0, 1.0 / self.Izz]])
            # gamma = self.KM / self.KF    ## gamma 是电机的转矩常数 KM 和推力常数 KF 的比值。
            gamma = 0.1
            # X = np.cat((x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body))
            # U = np.cat((f1, f2, f3, f4))
            def torchRotZ(psi):
                '''Rotation matrix about Z axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

                Args:
                psi: Scalar rotation

                Returns:
                R: torch Rotation matrix
                '''
                R = np.array([[np.cos(psi), -np.sin(psi), 0],
                                [np.sin(psi), np.cos(psi), 0],
                                [0, 0, 1]])
                return R

            def torchRotY(theta):
                '''Rotation matrix about Y axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

                Args:
                theta: Scalar rotation

                Returns:
                R: torch Rotation matrix
                '''
                R = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])
                return R

            def torchRotX(phi):
                '''Rotation matrix about X axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

                Args:
                phi: Scalar rotation

                Returns:
                R: torch Rotation matrix
                '''
                R = np.array([[1, 0, 0],
                                [0, np.cos(phi), -np.sin(phi)],
                                [0, np.sin(phi), np.cos(phi)]])
                return R

            Rob = torchRotZ(X[6]) @ torchRotY(X[7]) @ torchRotX(X[8])
            # import ipdb;ipdb.set_trace()
            pos_ddot = Rob @ np.concatenate((np.array([0.0]), np.array([0.0]), (U[0] + U[1] + U[2] + U[3]).reshape(1,))) / m - np.concatenate((np.array([0.0]), np.array([0.0]), np.array([g])))
            pos_dot = np.array([X[1],X[3],X[5]])
            Mb = np.array((self.L / np.sqrt(2.0) * (U[0] + U[1] - U[2] - U[3]),
                            self.L / np.sqrt(np.array(2.0)) * (-U[0] + U[1] + U[2] - U[3]),
                            gamma * (-U[0] + U[1] - U[2] + U[3])))
            def skew_matrix(angular_velocity):
                # Create a 3x3 skew-symmetric matrix from a 3D angular velocity vector
                return np.array([[0, -angular_velocity[2], angular_velocity[1]],
                                    [angular_velocity[2], 0, -angular_velocity[0]],
                                    [-angular_velocity[1], angular_velocity[0], 0]])
            rate_dot = Jinv @ (Mb - skew_matrix(np.array([X[9], X[10], X[11]])) @ J @ np.array([X[9], X[10], X[11]]))
            # Define the components of the rotation matrix
            R1 = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(X[6]), -np.sin(X[6])],
                            [0.0, np.sin(X[6]), np.cos(X[6])]])
            R2 = np.array([[np.cos(X[7]), 0.0, np.sin(X[7])],
                            [0.0, 1.0, 0.0],
                            [-np.sin(X[7]), 0.0, np.cos(X[7])]])
            R3 = np.array([[np.cos(X[8]), -np.sin(X[8]), 0.0],
                            [np.sin(X[8]), np.cos(X[8]), 0.0],
                            [0.0, 0.0, 1.0]])
            # Compute the angular velocity vector
            ang_dot = R1 @ R2 @ R3 @ np.array((X[9], X[10], X[11]))
            # Flatten ang_dot and rate_dot into one-dimensional tensors
            ang_dot_flat = ang_dot.reshape(-1)
            rate_dot_flat = rate_dot.reshape(-1)
            # Concatenate all tensors
            X_dot = np.concatenate((pos_dot, pos_ddot, ang_dot_flat, rate_dot_flat))
            return X_dot
     
    def reset(self, init_state=None):
        if init_state is None:
            for init_name in INIT_STATE_RAND_INFO:  # Default zero state.
                self.__dict__[init_name.upper()] = 0.
            self.state = np.ones(self.state_dim)
        else:
            if isinstance(init_state, np.ndarray):  # Full state as numpy array .
                for i, init_name in enumerate(self.INIT_STATE_LABELS[self.QUAD_TYPE]):
                    self.__dict__[init_name.upper()] = init_state[i]
            elif isinstance(init_state, dict):  # Partial state as dictionary.
                for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                    self.__dict__[init_name.upper()] = init_state.get(init_name, 0.)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), init_state incorrect format.')
        return self._get_obs()
    
    def step(self, thrust):
        X_dot= self.f_xu(X=self.state,U=thrust)
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
        if self.QUAD_TYPE == QuadType.ONE_D:
            mask = np.array([1, 0])
        if self.QUAD_TYPE == QuadType.TWO_D:
            mask = np.array([1, 0, 1, 0, 1, 0])
        if self.QUAD_TYPE == QuadType.THREE_D:
            mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
        # Element-wise or to check out-of-bound conditions.
        # import ipdb; ipdb.set_trace()
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

        # import ipdb; ipdb.set_trace()
        state_error = state_error * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        # if self.constraints is not None:
        #     info['constraint_values'] = self.constraints.get_values(self)
        #     info['constraint_violations'] = self.constraints.get_violations(self)
        return info
 
