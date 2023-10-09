from quadrotor.utils.registration import register

register(idx='ppo',
         entry_point='quadrotor.rl_controller.ppo:PPO',
         config_entry_point='quadrotor.rl_controller:ppo.yaml')