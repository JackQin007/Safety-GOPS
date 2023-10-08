### 2D Quadrotor Trajectory Tracking with PPO

```bash
cd ~/code/quadrotor/
python3 rl_experiment.py \
    --algo ppo \
    --task quadrotor \
    --overrides \
        env/quadrotor_2D/quadrotor_2D_track.yaml \
        env/quadrotor_2D/ppo_quadrotor_2D.yaml \
    --kv_overrides \
        algo_config.training=False
```

### 3D Quadrotor Trajectory Tracking with PPO

```bash
cd ~/code/quadrotor/
python3 rl_experiment.py \
    --algo ppo \
    --task quadrotor \
    --overrides \
        env/quadrotor_3D/quadrotor_3D_track.yaml \
        env/quadrotor_3D/ppo_quadrotor_3D.yaml \
    --kv_overrides \
        algo_config.training=False
```


