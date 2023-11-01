# safety-GOPS
## quadrotor
### env (context_info) 
### model (agent_info) implementation by torch(CasADi origin)
### ppo (rl_algorithm)
### utils (for_registeration) Factory Pattern

### trajectory:
#### "8"
$$
 x(t) = A sin(\omega t) 
$$
$$
y(t) = A sin(\omega t) cos(\omega t) 
$$

其中：
- \( A \) 是缩放因子，它决定了轨迹的大小。
- \( $\omega$ \) 是轨迹的角频率，它与轨迹的周期 \( T \) 有关系：
$$
\omega = \frac{2\pi}{T} 
$$
- \( t \) 是时间。
