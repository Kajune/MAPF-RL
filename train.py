from pprint import pprint
import argparse
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec

import gymnasium as gym
from mapf_env import MAPFEnv


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, choices=['train', 'eval'], default='train')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
args = parser.parse_args()


if __name__ == '__main__':
	ray.init()
	algo = ppo.PPO(env=MAPFEnv, config={
		"local_dir": "./log_dir",
		"framework": "torch",
		"num_workers": 2 if args.mode == "train" else 0,
		"num_gpus": 1,
		"env_config": {
			"num_agents": 500,
			"fov": 5,
			"map_size": 256,
			"mobility_thresh": 0.1,
			"manual_render": args.mode == "eval"
		},
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

		"eval_config": {
			"render_env": True
		}
	})


	if args.mode == "train":
		while True:
			pprint(algo.train())
			algo.save(args.checkpoint_path)

	else:
		algo.restore(args.checkpoint_path)
		pprint(algo.evaluate())
