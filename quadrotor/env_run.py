from functools import partial
from quadrotor.utils.registration import make
from quadrotor.utils.configuration import ConfigFactory
from quadrotor.env.benchmark_env import Task, Environment
from quadrotor.rl_controller import *


def main():
    # 创建命令行参数解析器
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                        config.task,
                        **config.task_config)
    env = env_func()
    env.reset()
    obs, reward, done, info = env.step(action = [-1.8674284 ,-0.7863587])
    print('obs, reward, done, info',obs,reward,done,info)

if __name__ == '__main__':
    main()
   
