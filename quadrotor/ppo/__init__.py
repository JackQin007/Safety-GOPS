from utils.registration import register

register(idx='ppo',
         entry_point='ppo.ppo:PPO',
         config_entry_point='ppo:ppo.yaml')