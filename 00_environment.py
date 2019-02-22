import logging
import os, sys

import gym
from gym.wrappers import Monitor
import gym_ple

# random agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

render = True  # 画面表示したい場合は True, そうでない場合は False にしてください

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

    if not render:
        outdir = '/tmp/random-agent-results'
        env = Monitor(env, directory=outdir, force=True)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                break

    # Dump result info to disk
    env.close()

    logger.info("Successfully ran RandomAgent")

