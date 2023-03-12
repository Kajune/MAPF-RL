from pprint import pprint
import argparse
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec

import gymnasium as gym
from mapf_env import MAPFEnvMulti, MAPFEnvSingle


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
parser.add_argument('--load_checkpoint', type=str)
args = parser.parse_args()


if __name__ == '__main__':
	ray.init()

	num_agents = 500

	env_config = {
		"num_agents": num_agents,
		"fov": 5,
		"map_size": 256,
		"mobility_thresh": 0.1,
		"dynamic_terrain": True,
		"manual_render": True,
	}


	algo = ppo.PPO(env=MAPFEnvMulti, config={
		"framework": "torch",
		"num_workers": 2,
		"num_gpus": 1,
		"env_config": env_config,
		"multiagent": {
			"policies": {
				"common": PolicySpec(),
			},
			"policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "common",
		},
		"disable_env_checking": True,
		"render_env": False,

		"train_batch_size": 512,
		"num_sgd_iter": 5,
	})

	"""
	algo = ppo.PPO(env=MAPFEnvSingle, config={
		"framework": "torch",
		"num_workers": 2,
		"num_gpus": 1,
		"env_config": env_config,
		"disable_env_checking": True,
		"render_env": False,

		"train_batch_size": 512,
		"num_sgd_iter": 5,
	})
	"""

	if args.load_checkpoint is not None:
		alog.restore(args.load_checkpoint)

	while True:
		pprint(algo.train())
		algo.save(args.checkpoint_path)
