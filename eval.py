from pprint import pprint
import argparse
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec

import gymnasium as gym
from mapf_env import MAPFEnvMulti, MAPFEnvSingle

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
args = parser.parse_args()


if __name__ == '__main__':
	ray.init()

	env_config = {
		"num_agents": 500,
		"fov": 5,
		"map_size": 256,
		"mobility_thresh": 0.1,
		"dynamic_terrain": True
		"manual_render": True,
	}

	algo = ppo.PPO(env=MAPFEnvMulti, config={
		"framework": "torch",
		"env_config": env_config,
		"create_env_on_driver": True,
	})

	algo.restore(args.checkpoint_path)
	pprint(algo.evaluate())

