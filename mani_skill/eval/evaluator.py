import gym
import mani_skill.env
import csv
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from tabulate import tabulate

class Evaluator(object):
    def __init__(self, env_name, policy):
        self.env_name = env_name
        self.env = gym.make(env_name, obs_mode=policy.obs_mode)
        self.policy = policy
        self.result = {}

    def run(self, level_list, vis=False, pause_per_step=False):
        self.result = OrderedDict()
        policy = self.policy
        env = self.env

        n_episodes = len(level_list)

        cnt = defaultdict(int)
        for i_ep in tqdm(range(n_episodes)):
            policy.reset()
            level_idx = level_list[i_ep]
            obs = env.reset(level=level_idx) # OpenAI gym wrappers only take kwargs
            while True:
                if vis:
                    env.render('human')
                action = policy.act(obs)
                obs, reward, done, info = env.step(action)
                if pause_per_step:
                    aaaa = input()
                if done:
                    if 'eval_info' not in info:
                        raise Exception('No eval_info found in info.')
                    eval_info = info['eval_info']
                    for key, value in eval_info.items():
                        cnt[key] += int(value)
                    break
        for key, value in cnt.items():
            self.result[key] = value * 1.0 / n_episodes
        
        return self.result

    def export_to_csv(self, path='./eval_results.csv'):
        headers = ['env'] + list(self.result.keys())
        table = [
            [self.env_name] + list(self.result.values()),
        ]
        print(tabulate(table, headers=headers, tablefmt='psql', floatfmt='.4f'))
        with open(path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(table)
        print('The evaluation result is saved to {:s}.'.format(path))

    def __del__(self):
        self.env.close()