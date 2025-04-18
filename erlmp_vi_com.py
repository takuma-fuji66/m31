import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


# 試行回数の設定
num_trials = 1  # ここを変更して試行回数を指定

for trial in range(1, num_trials + 1):
    print(f"=== Trial {trial}/{num_trials} ===")

    seed = np.random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

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
          assert self.done == False, 'WARNING: The game is done. Call reset().'
          agent_name = self.agent_selection
          if agent_name == 'sender':
              self.message = action
              self.agent_selection = 'receiver'
          elif agent_name == 'receiver':
              self.action = action
              self.success = self.state == action
              self.done = True

    class Gene:
      def __init__(self, size, min_val, max_val, init_values='random'):
          # size は整数または整数のタプルです．
          self.size = size
          self.min_val = min_val
          self.max_val = max_val
          self.init_values = init_values
          if init_values == 'random':
              self.values = rng.uniform(self.min_val, self.max_val, size)
          else:
              self.values = np.full(size, init_values)

      def crossover(self, other_gene):
          new_gene = Gene(self.size, self.min_val, self.max_val, init_values=self.init_values)
          random_mask = rng.integers(0, 2, self.size)
          new_gene.values = np.where(random_mask, self.values, other_gene.values)
          return new_gene

      def mutate(self, mutation_rate):
          random_mask = rng.uniform(0, 1, self.size) < mutation_rate
          self.values = np.where(random_mask, rng.uniform(self.min_val, self.max_val, self.size), self.values)
          return self

    class UrnAgent:
      def __init__(self, bias_gene):
          self.bias_gene = bias_gene
          self.num_obs = bias_gene.size[0]
          self.num_choices = bias_gene.size[1]
          self.urn_balls = [np.ones(self.num_choices, dtype=float) for _ in range(self.num_obs)]
          self.urn_sum_balls = [self.num_choices for _ in range(self.num_obs)]
          self.train_buf = []
          # Apply innate bias
          for obs in range(self.num_obs):
              self.urn_balls[obs] += self.bias_gene.values[obs]
              self.urn_sum_balls[obs] = self.urn_balls[obs].sum()

      def get_action(self, obs):
          p = self.urn_balls[obs] / self.urn_sum_balls[obs]
          return rng.choice(np.arange(self.num_choices), p=p)

      def store_buffer(self, obs, choice):
          self.train_buf.append([obs, choice, 0]) # 観測，選択，報酬の組

      def update_reward(self, reward):
          if len(self.train_buf) > 0:
            self.train_buf[-1][2] += reward

      def train(self):
          for obs, choice, reward in self.train_buf:
              self.urn_balls[obs][choice] += reward
              self.urn_sum_balls[obs] += reward
          self.train_buf = []

      
      def visualize_urn(sender, receiver, gen = None, output_dir="urn", title="Urn Contents"):
        if gen is not None:
            title = f"{title} (Generation {gen})"

            # データ収集（Sender）
            sender_current_balls = np.array([urn for urn in sender.urn_balls])
            sender_genetic_bias = sender.bias_gene.values
            sender_num_obs = sender.num_obs
            sender_num_choices = sender.num_choices

            # データ収集（Receiver）
            receiver_current_balls = np.array([urn for urn in receiver.urn_balls])
            receiver_genetic_bias = receiver.bias_gene.values
            receiver_num_obs = receiver.num_obs
            receiver_num_choices = receiver.num_choices

            # プロットの準備
            fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={"hspace": 0.3})  # 2行1列で上下に分割
            bar_width = 0.35

            # Senderのプロット
            x_sender = np.arange(sender_num_obs * sender_num_choices)
            sender_current_balls_flat = sender_current_balls.flatten()
            sender_genetic_bias_flat = sender_genetic_bias.flatten()

            axes[0].bar(x_sender - bar_width / 2, sender_current_balls_flat, bar_width, label="Current Balls")
            axes[0].bar(x_sender + bar_width / 2, sender_genetic_bias_flat, bar_width, label="Genetic Bias")
            axes[0].set_title("Sender's Urn Contents")
            axes[0].set_xlabel("Observations & Choices")
            axes[0].set_ylabel("Number of Balls")
            axes[0].legend(loc="upper right")
            sender_x_labels = [f"{obs}->{choice}" for obs in range(sender_num_obs) for choice in range(sender_num_choices)]
            axes[0].set_xticks(x_sender)
            axes[0].set_xticklabels(sender_x_labels, rotation=90)

            # Receiverのプロット
            x_receiver = np.arange(receiver_num_obs * receiver_num_choices)
            receiver_current_balls_flat = receiver_current_balls.flatten()
            receiver_genetic_bias_flat = receiver_genetic_bias.flatten()

            axes[1].bar(x_receiver - bar_width / 2, receiver_current_balls_flat, bar_width, label="Current Balls")
            axes[1].bar(x_receiver + bar_width / 2, receiver_genetic_bias_flat, bar_width, label="Genetic Bias")
            axes[1].set_title("Receiver's Urn Contents")
            axes[1].set_xlabel("Observations & Choices")
            axes[1].set_ylabel("Number of Balls")
            axes[1].legend(loc="upper right")
            receiver_x_labels = [f"{obs}->{choice}" for obs in range(receiver_num_obs) for choice in range(receiver_num_choices)]
            axes[1].set_xticks(x_receiver)
            axes[1].set_xticklabels(receiver_x_labels, rotation=90)

            # グラフ全体のタイトル
            fig.suptitle(title, fontsize=16)

            # 保存先フォルダの作成
            os.makedirs(output_dir, exist_ok=True)

            # ファイル名の生成
            filename = f"gen{gen}_combined_urn.png"
            filepath = os.path.join(output_dir, filename)

            # グラフの保存
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # タイトルと各プロットの間隔調整
            plt.savefig(filepath)
            plt.close(fig)


    def str_agent(agent):
        str_urn = ''
        for obs in range(agent.num_obs):
            for choice in range(agent.num_choices):
                str_urn += f'{obs}->{choice}:{agent.urn_balls[obs][choice]:.0f}({agent.bias_gene.values[obs][choice]:.0f}), '
            str_urn += '\n'
        return str_urn
    

    class Creature:
      def __init__(self, bias_genes, lifespan_mean):
          self.bias_genes = bias_genes
          self.agents = dict()
          for agent_name, bias_gene in bias_genes.items():
              self.agents[agent_name] = UrnAgent(bias_gene)
          self.fitness = 0
          # (1 / lifespan_mean)をパラメータとした幾何分布から寿命をサンプルする
          self.lifespan = rng.geometric(1 / lifespan_mean)


    num_states = 2          # 状態数
    num_messages = 2        # 信号数
    state_dist = [1.0 / num_states] * num_states # 状態の確率分布
    env = SignalingGameEnv(num_states, num_messages, state_dist)

    pop_size = 20           # 集団内の個体数
    num_episodes_per_creature = 25 # 1世代における1個体あたりのエピソード数
    min_bias = 0.0          # 遺伝的な行動バイアスの最小値
    max_bias = 30.0         # 遺伝的な行動バイアスの最大値
    init_bias = 0.0         # 遺伝的な行動バイアスの初期値．乱数で初期化する場合は'random'
    success_fitness = {'sender': 1, 'receiver': 1} # コミュニケーション成功時の適応度増加量
    success_reward = {'sender': 1, 'receiver': 1} # コミュニケーション成功時の報酬量
    tornament_size = 5     # トーナメントサイズ
    do_crossover = True     # 交叉を行うかどうか．行う場合は親二人，行わない場合は親一人のみを選択
    mutation_rate = 0.001   # 突然変異率
    lifespan_mean = 1.0     # 平均寿命（幾何分布のパラメータの逆数）


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

    # 初期個体群の生成
    population = []
    for _ in range(pop_size):
        sender_bias_gene = Gene((num_states, num_messages), min_bias, max_bias, init_bias)
        receiver_bias_gene = Gene((num_messages, num_states), min_bias, max_bias, init_bias)
        bias_genes = {'sender': sender_bias_gene, 'receiver': receiver_bias_gene}
        population.append(Creature(bias_genes, lifespan_mean))

    # あとで進化過程を確認するためのログ
    log_success_rate = []
    log_fitness_mean = []
    log_fitness_var = []
    log_rewards_mean = {'sender': [], 'receiver': []}
    log_bias_suc_fail_diff_mean = {'sender': [], 'receiver': []}

    num_gen = 1000           # 世代数一旦1000で固定

    # 初期コミュニケーション成功率をロギングするためのリスト
    log_initial_success_rate = []

    # 進化計算
    for gen in range(num_gen): # 生まれた瞬間の成功率をとって、報酬0の場合（遺伝のみ）と学習する場合で比較
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
        log_rewards = {'sender': [], 'receiver': []}
        log_success_choice_biases = {'sender': [], 'receiver': []}
        log_fail_choice_biases = {'sender': [], 'receiver': []}

        # Running episodes
        num_episodes_in_this_gen = num_episodes_per_creature * pop_size
        for episode in range(num_episodes_in_this_gen):
            # 送信者と受信者の選択
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
                agent.store_buffer(obs, action)
            # 1エピソード終了．成功・不成功に基づいてfitnessとrewardを与え，方策を更新する．
            log_success.append(1 if env.success else 0)
            for _agent_name in env.agent_names:
                # 適応度
                fitness = success_fitness[_agent_name] if env.success else 0
                creatures[_agent_name].fitness += fitness
                # 報酬
                reward = success_reward[_agent_name] if env.success else 0
                creatures[_agent_name].agents[_agent_name].update_reward(reward)
                log_rewards[_agent_name].append(reward)
                # 遺伝子のロギング（成功時に，その行動が遺伝的にどれだけバイアスされていたか）
                if _agent_name == 'sender':
                    obs, choice = env.state, env.message
                elif _agent_name == 'receiver':
                    obs, choice = env.message, env.action
                choice_bias = creatures[_agent_name].bias_genes[_agent_name].values[obs, choice]
                if env.success:
                    log_success_choice_biases[_agent_name].append(choice_bias)
                else:
                    log_fail_choice_biases[_agent_name].append(choice_bias)
                # 方策を更新する
                creatures[_agent_name].agents[_agent_name].train()
                    

        # 1世代終了．最も適応度の高い個体の壺のボールの数と遺伝子を表示
        best_creature = max(population, key=lambda x: x.fitness)
        print(f'gen {gen} best creature (fitness {best_creature.fitness}):')
        print('sender:')
        print(str_agent(best_creature.agents['sender']))
        print('receiver:')
        print(str_agent(best_creature.agents['receiver']))
        if gen % 100 == 0 or gen == 999:
            best_sender = best_creature.agents["sender"]
            best_receiver = best_creature.agents["receiver"]
            UrnAgent.visualize_urn(sender=best_sender, receiver=best_receiver, gen=gen, title="Best Urn Contents")


        # 適応度のロギング
        log_fitness = [creature.fitness for creature in population]

        # 次世代の個体群を生成する．
        new_population = []
        for i in range(pop_size):
            # 寿命が来ていない個体はそのまま次世代に引き継ぐ
            if population[i].lifespan > 1:
                population[i].lifespan -= 1
                new_population.append(population[i])
                continue
            # トーナメント選択で親を選び，突然変異を行う．
            tournament = rng.choice(population, tornament_size, replace=False)
            if do_crossover:
                winners = sorted(tournament, key=lambda x: x.fitness, reverse=True)[:2]
                new_genes = dict()
                for role in env.agent_names:
                    new_genes[role] = winners[0].bias_genes[role].crossover(winners[1].bias_genes[role])
            else:
                winner = max(tournament, key=lambda x: x.fitness)
                new_genes = winner.bias_genes
            for role in env.agent_names:
                new_genes[role].mutate(mutation_rate)
            new_population.append(Creature(new_genes, lifespan_mean))
        population = new_population

        # 進化過程のロギング
        log_success_rate.append(np.mean(log_success))
        for _agent_name in env.agent_names:
            log_fitness_mean.append(np.mean(log_fitness))
            log_fitness_var.append(np.var(log_fitness))
            log_rewards_mean[_agent_name].append(np.mean(log_rewards[_agent_name]))
            suc_bias_mean = np.mean(log_success_choice_biases[_agent_name])
            fail_bias_mean = np.mean(log_fail_choice_biases[_agent_name])
            log_bias_suc_fail_diff_mean[_agent_name].append(suc_bias_mean - fail_bias_mean)

        print(f'gen {gen}: success_rate {log_success_rate[-1]} ' \
              f'fitness_mean {log_fitness_mean[-1]} ' \
              f'fitness_var {log_fitness_var[-1]}')
        
    from torch.utils.tensorboard import SummaryWriter
    settings = f'states{num_states}_messages{num_messages}_' \
                f'popsize{pop_size}_episodes{num_episodes_per_creature}_' \
                f'minbias{min_bias}_maxbias{max_bias}_initbias{init_bias}_' \
                f'success_fitness{success_fitness["sender"]}_{success_fitness["receiver"]}_' \
                f'success_reward{success_reward["sender"]}_{success_reward["receiver"]}_' \
                f'tornament{tornament_size}_crossover{do_crossover}_mutation{mutation_rate}_' \
                f'matching{matching_bias}_groups{num_groups}_' \
                f'trial{trial}_seed{seed}' 
    writer = SummaryWriter(log_dir=f'./tb/s2_m2_sf1_1_sr1_1_epi25_test/{settings}')
    for i, (suc, isuc, f_m, f_v, r_ms, r_mr, g_bds, gbdr) in enumerate(zip(log_success_rate,
                                                        log_initial_success_rate,
                                                        log_fitness_mean,
                                                        log_fitness_var,
                                                        log_rewards_mean['sender'],
                                                        log_rewards_mean['receiver'],
                                                        log_bias_suc_fail_diff_mean['sender'],
                                                        log_bias_suc_fail_diff_mean['receiver'])):
        writer.add_scalar("success/success_rate", suc, i)
        writer.add_scalar("success/initial_success_rate", isuc, i)
        writer.add_scalar("fitness/fitness_mean", f_m, i)
        writer.add_scalar("fitness/fitness_var", f_v, i)
        writer.add_scalar("reward/sender_reward_mean", r_ms, i)
        writer.add_scalar("reward/receiver_reward_mean", r_mr, i)
        writer.add_scalar("bias/success_fail_diff_mean_sender", g_bds, i)
        writer.add_scalar("bias/success_fail_diff_mean_receiver", gbdr, i)
    writer.close()