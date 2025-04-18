import numpy as np
import random
from tqdm import tqdm

num_states = 8
num_messages = 8
state_dist = [1.0 / num_states] * num_states
pop_size = 20
num_generations = 100
num_episodes_per_generation = 200
mutation_rate = 0.01
learning_rate = 0.5
lifespan_mean = 1.0
rng = np.random.default_rng()

class SignalingGameEnv:
    def __init__(self, num_states, num_messages, state_dist):
        self.num_states = num_states
        self.num_messages = num_messages
        self.state_dist = state_dist
        self.agent_names = ['sender', 'receiver']
        self.reset()

    def reset(self):
        self.state = rng.choice(self.num_states, p=self.state_dist)
        self.success = None
        self.done = False
        self.agent_selection = 'sender'

    def observe(self, agent_name):
        if agent_name == 'sender':
            return self.state
        elif agent_name == 'receiver':
            return self.message

    def step(self, action):
        assert not self.done
        agent_name = self.agent_selection
        if agent_name == 'sender':
            self.message = action
            self.agent_selection = 'receiver'
        elif agent_name == 'receiver':
            self.action = action
            self.success = (self.state == action)
            self.done = True

# === Gene / Agent / Creature ===
class Gene:
    def __init__(self, obs_dim, act_dim):
        self.initial_weight = rng.normal(0, 1, (obs_dim, act_dim))
        self.initial_bias = rng.normal(0, 1, act_dim)

    def crossover(self, other):
        new_gene = Gene(self.initial_weight.shape[0], self.initial_weight.shape[1])
        mask_w = rng.uniform(0, 1, self.initial_weight.shape) < 0.5
        mask_b = rng.uniform(0, 1, self.initial_bias.shape) < 0.5
        new_gene.initial_weight = np.where(mask_w, self.initial_weight, other.initial_weight)
        new_gene.initial_bias = np.where(mask_b, self.initial_bias, other.initial_bias)
        return new_gene

    def mutate(self, mutation_rate):
        mask_w = rng.uniform(0, 1, self.initial_weight.shape) < mutation_rate
        self.initial_weight[mask_w] = rng.normal(0, 1, mask_w.sum())
        mask_b = rng.uniform(0, 1, self.initial_bias.shape) < mutation_rate
        self.initial_bias[mask_b] = rng.normal(0, 1, mask_b.sum())
        return self

class ReinforceAgent:
    def __init__(self, gene, obs_dim, act_dim, learning_rate):
        self.weights = np.copy(gene.initial_weight)
        self.bias = np.copy(gene.initial_bias)
        self.learning_rate = learning_rate
        self.buffer = []
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def encode_obs(self, obs_idx):
        vec = np.zeros(self.obs_dim)
        vec[obs_idx] = 1.0
        return vec

    def get_action(self, obs_idx):
        obs = self.encode_obs(obs_idx)
        z = np.dot(obs, self.weights) + self.bias
        probs = np.exp(z - np.max(z))
        probs /= np.sum(probs)
        action = rng.choice(self.act_dim, p=probs)
        self.buffer.append((obs, action, probs[action]))
        return action

    def update(self, reward):
        for obs, action, prob in self.buffer:
            grad = -1 / prob
            self.weights[:, action] -= self.learning_rate * grad * obs * reward
            self.bias[action] -= self.learning_rate * grad * reward
        self.buffer = []

class Creature:
    def __init__(self, obs_dims, act_dims, learning_rate):
        self.genes = {
            'sender': Gene(obs_dims['sender'], act_dims['sender']),
            'receiver': Gene(obs_dims['receiver'], act_dims['receiver'])
        }
        self.agents = {
            'sender': ReinforceAgent(self.genes['sender'], obs_dims['sender'], act_dims['sender'], learning_rate),
            'receiver': ReinforceAgent(self.genes['receiver'], obs_dims['receiver'], act_dims['receiver'], learning_rate)
        }
        self.fitness = 0
        self.lifespan = rng.geometric(1 / lifespan_mean)

    def crossover_mutate(self, other, obs_dims, act_dims, learning_rate):
        new_genes = {
            role: self.genes[role].crossover(other.genes[role]).mutate(mutation_rate)
            for role in self.genes
        }
        offspring = Creature(obs_dims, act_dims, learning_rate)
        offspring.genes = new_genes
        offspring.agents = {
            role: ReinforceAgent(new_genes[role], obs_dims[role], act_dims[role], learning_rate)
            for role in new_genes
        }
        return offspring
    
# 近接性の定義
matching_bias = 'uniform'
num_groups = 5
matching_dists = []
if matching_bias == 'uniform':
    for s in range(pop_size):
        matching_dist = np.ones(pop_size) / pop_size
        matching_dists.append(matching_dist)
elif matching_bias == 'group':
    for s in range(pop_size):
        group = s % num_groups
        matching_dist = np.zeros(pop_size)
        matching_dist[group::num_groups] = 1
        matching_dist /= matching_dist.sum()
        matching_dists.append(matching_dist)

# 実行
env = SignalingGameEnv(num_states, num_messages, state_dist)
obs_dims = {'sender': num_states, 'receiver': num_messages}
act_dims = {'sender': num_messages, 'receiver': num_states}
population = [Creature(obs_dims, act_dims, learning_rate) for _ in range(pop_size)]

# あとで進化過程を確認するためのログ
log_success_rate = []
log_initial_success_rate = []
log_fitness_mean = []
log_fitness_var = []
#log_rewards_mean = {'sender': [], 'receiver': []}
#log_bias_suc_fail_diff_mean = {'sender': [], 'receiver': []}

# 進化計算
for gen in range(num_generations):
    initial_success = []
    # num_episodes_in_test = num_episodes_per_creature * pop_size
    num_episodes_in_test = 1000
    for episode in range(num_episodes_in_test):
        sender_idx = rng.integers(0, pop_size)
        sender = population[sender_idx]
        receiver = rng.choice(population, p=matching_dists[sender_idx])
        creatures = {'sender': sender, 'receiver': receiver}
        env.reset()
        while not env.done:
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            agent = creatures[agent_name].agents[agent_name]
            action = agent.get_action(obs)
            env.step(action)
        # コミュニケーション成功を記録
        initial_success.append(1 if env.success else 0)

    # 初期成功率をロギング
    log_initial_success_rate.append(np.mean(initial_success))

    # ログ出力
    print(f'gen {gen}: initial_success_rate {log_initial_success_rate[-1]}')

    log_success = []
    log_fitness = []
    #log_rewards = {'sender': [], 'receiver': []}
    #log_success_choice_biases = {'sender': [], 'receiver': []}
    #log_fail_choice_biases = {'sender': [], 'receiver': []}

    num_episodes_in_this_gen = num_episodes_per_generation * pop_size
    for episode in range(num_episodes_per_generation):
        sender, receiver = rng.choice(population, 2, replace=False)
        env.reset()
        pair = {'sender': sender, 'receiver': receiver}

        while not env.done:
            role = env.agent_selection
            obs = env.observe(role)
            act = pair[role].agents[role].get_action(obs)
            env.step(act)

        reward = 1 if env.success else 0
        for role in ['sender', 'receiver']:
            pair[role].fitness += reward
            pair[role].agents[role].update(reward)

    # 世代交代
    new_population = []
    for creature in population:
        creature.lifespan -= 1
        if creature.lifespan > 0:
            new_population.append(creature)
        else:
            p1, p2 = rng.choice(population, 2, replace=False)
            new_population.append(p1.crossover_mutate(p2, obs_dims, act_dims, learning_rate))

    population = new_population
    avg_fit = np.mean([c.fitness for c in population])
    print(f"[Gen {gen}] Avg Fitness: {avg_fit:.3f}")
