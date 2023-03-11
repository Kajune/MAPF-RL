from enum import Enum
import numpy as np
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import cv2
from perlin import perlin


np.random.seed(0)


class MAPFEnv(MultiAgentEnv):
	Directions = {
		0: np.int32([-1, 0]),
		1: np.int32([1, 0]),
		2: np.int32([0, -1]),
		3: np.int32([0, 1]),
		4: np.int32([0, 0]),
	}


	class Agent:
		def __init__(self, terrain, pos, mobility_thresh):
			self.terrain = terrain
			self.pos = pos
			self.mobility_thresh = mobility_thresh
			self.cost = 0
			self.trajectory = []


		def move(self, action):
			self.trajectory.append(self.pos)
			new_pos = self.pos + MAPFEnv.Directions[action]

			if self.terrain[new_pos[0], new_pos[1]] > self.mobility_thresh:
				return 0

			new_pos[0] = min(max(new_pos[0], 0), self.terrain.shape[0] - 1)
			new_pos[1] = min(max(new_pos[1], 0), self.terrain.shape[1] - 1)
#			distance = np.linalg.norm(new_pos - self.pos)
			distance = 1
			cost = (self.terrain[new_pos[0], new_pos[1]] + self.terrain[self.pos[0], self.pos[1]]) / 2 * distance
			self.pos = new_pos
			self.cost += cost
			return cost



	def __init__(self, env_config):
		self.manual_render = env_config["manual_render"]

		self.num_agents = env_config["num_agents"]
		self.fov = env_config["fov"]
		self.agent_list = []

		self.map_size = env_config["map_size"]
		self.sg_margin = 10
		self.mobility_thresh = env_config["mobility_thresh"]
		self.horizon = self.map_size * 2
		self.count = 0

		self.terrain = self._generate_terrain((self.map_size, self.map_size))
		self.start = np.zeros_like(self.terrain)
		self.start[:,:self.sg_margin][self.terrain[:,:self.sg_margin] < self.mobility_thresh] = 1
		self.goal = np.zeros_like(self.terrain)
		self.goal[:,-self.sg_margin:][self.terrain[:,-self.sg_margin:] < self.mobility_thresh] = 1

		self.r_time = -0.1 / self.map_size
		self.r_goal = 1
		self.r_approach = 1 / self.map_size
		self.r_collision = -10 / self.map_size
		self.r_cost = -10 / self.map_size


	@property
	def observation_space(self):
		return gym.spaces.Box(0, np.inf, (((self.fov * 2 + 1) ** 2) * 2,), dtype=np.float32)


	@property
	def action_space(self):
		return gym.spaces.Discrete(len(self.Directions))


	def reset(self, *, seed=None, options=None):
		start_pos_cands = np.array(np.where(self.start > 0)).T
		pos = start_pos_cands[np.random.choice(np.arange(0, len(start_pos_cands)), self.num_agents, replace=False)]
		self.agent_list = [self.Agent(self.terrain, pos[i], self.mobility_thresh) for i in range(self.num_agents)]
		self.count = 0

		return self._get_obs(), {}


	def step(self, actions):
		pos_list_prev = np.int32([agent.pos for agent in self.agent_list])
		cost_list = {}

		for ai in actions:
			agent = self.agent_list[ai]
			if self.goal[agent.pos[0], agent.pos[1]] > 0:
				continue
			cost = agent.move(actions[ai])
			cost_list[ai] = cost

		pos_list_next = np.int32([agent.pos for agent in self.agent_list])
		is_goal = self.goal[pos_list_next[:,0], pos_list_next[:,1]] > 0

		rewards = {agent_index: 0 for agent_index in actions}
		terminated = {agent_index: False for agent_index in actions}
		truncated = {"__all__": False}

		goal_pos_list = np.array(np.where(self.goal > 0)).T
		distance_to_goal_prev = np.min(np.linalg.norm(goal_pos_list[np.newaxis,:,:] - pos_list_prev[:,np.newaxis,:], axis=2), axis=1)
		distance_to_goal_next = np.min(np.linalg.norm(goal_pos_list[np.newaxis,:,:] - pos_list_next[:,np.newaxis,:], axis=2), axis=1)
		num_collision = np.sum(np.all(pos_list_next[np.newaxis,:,:] == pos_list_next[:,np.newaxis,:], axis=2), axis=1) - 1

		for ai in actions:
			agent = self.agent_list[ai]

			if is_goal[ai]:
				# ゴールしたら報酬 & done
				rewards[ai] += self.r_goal
				terminated[ai] = True
				print("Agent %d Goal" % ai)
			else:
				# 時間経過で罰則
				rewards[ai] += self.r_time

			# コストに対して罰則
			rewards[ai] += self.r_cost * cost_list[ai]

			# ゴールに近づいたら報酬
			rewards[ai] += (distance_to_goal_prev[ai] - distance_to_goal_next[ai]) * self.r_approach

			# 衝突したら罰則
			rewards[ai] += self.r_collision * num_collision[ai]


		# 全てのエージェントがゴールしているか、時間切れになったらエピソード終了
		terminated['__all__'] = np.all(is_goal)
		truncated['__all__'] = self.count >= self.horizon
		if terminated['__all__']:
			print("Everyone goal")

		self.count += 1

		if self.manual_render:
			self.render()

		return self._get_obs(), rewards, terminated, truncated, {}


	def render(self):
		vis = cv2.cvtColor(self.terrain, cv2.COLOR_GRAY2BGR)
		vis[self.start > 0] = [0,1,0]
		vis[self.goal > 0] = [1,0,0]
		pos_list = np.int32([agent.pos for agent in self.agent_list])
		unique_pos_list = np.unique(pos_list, axis=0)
		vis[unique_pos_list[:,0], unique_pos_list[:,1]] = [0,1,1]

		duplicate_indices = np.sum(np.all(unique_pos_list[:,np.newaxis,:] == pos_list[np.newaxis,:,:], axis=2), axis=1) > 1
		duplicate_pos_list = unique_pos_list[duplicate_indices]
		vis[duplicate_pos_list[:,0], duplicate_pos_list[:,1]] = [0,0,1]

		cv2.imshow("", cv2.resize(vis, (512, 512)))
		cv2.waitKey(1)

		return (vis * 255).astype(np.uint8)


	def _get_obs(self):
		terrain_map = np.pad(self.terrain, [[self.fov, self.fov], [self.fov, self.fov]], "constant")
		agent_map = np.zeros_like(self.terrain)
		for ai in range(len(self.agent_list)):
			agent = self.agent_list[ai]
			agent_map[agent.pos[0], agent.pos[1]] = 1
		agent_map = np.pad(agent_map, [[self.fov, self.fov], [self.fov, self.fov]], "constant")

		obs_list = {}
		for ai in range(len(self.agent_list)):
			agent = self.agent_list[ai]
			if self.goal[agent.pos[0], agent.pos[1]] > 0:
				continue

			py, px = agent.pos
			terrain_obs = terrain_map[py:py+self.fov*2+1, px:px+self.fov*2+1]
			agent_obs = agent_map[py:py+self.fov*2+1, px:px+self.fov*2+1]
			obs = np.dstack((terrain_obs, agent_obs))
			obs_list[ai] = obs.ravel()

#			cv2.imshow("terrain", cv2.resize(terrain_obs, (256, 256)))
#			cv2.imshow("agent", cv2.resize(agent_obs, (256, 256)))
#			cv2.waitKey()

		return obs_list


	def _generate_terrain(self, size):
		bias = 0.25
		terrain = np.zeros(size, dtype=np.float32)
		for i in np.random.rand(2):
			x = np.linspace(0, i * size[0] / 64, size[0])
			y = np.linspace(0, i * size[1] / 64, size[1])
			terrain += perlin(np.array(np.meshgrid(x,y)))
		terrain += bias
		terrain /= np.max(terrain)
		terrain = np.clip(terrain, 0, 1)

		converted = cv2.cvtColor(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV_FULL)
		kernel = np.ones((3, 3), np.uint8)
		
		for width in [1, 2, 3]:
			region_size = size[0] // 2 // width
			ruler = 30
			min_element_size = 10
			num_iterations = 2

			slic = cv2.ximgproc.createSuperpixelSLIC(converted, cv2.ximgproc.SLICO, region_size, ruler)
			slic.iterate(num_iterations)
			slic.enforceLabelConnectivity(min_element_size)

			contour_mask = slic.getLabelContourMask(True)
			contour_mask = cv2.dilate(contour_mask, kernel, iterations=width // 2)
			terrain[0 < contour_mask] = 0

		return terrain


if __name__ == '__main__':
	env = MAPFEnv({
			"num_agents": 500,
			"fov": 5,
			"map_size": 256,
			"mobility_thresh": 0.1,
			"manual_render": False
		})

	obs, info = env.reset()
	while True:
#		env.render()
		actions = {agent: env.action_space.sample() for agent in obs}
		obs, reward, terminated, truncated, info = env.step(actions)
		if terminated['__all__'] or truncated['__all__']:
			break
