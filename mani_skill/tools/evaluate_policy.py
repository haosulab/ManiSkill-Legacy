'''
# Please name your solution file to user_solution.py, 
# Then add the directory containing your solution file to PYTHONPATH
# Run `PYTHONPATH=YOUR_SOLUTION_DIRECTORY:$PYTHONPATH python mani_skill/tools/evaluate_policy.py --env ENV_NAME`
'''

import argparse
import sys
from mani_skill.eval import Evaluator, RandomPolicy
# from mani_skill.eval import UserPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='ManiSkill Evaluation')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--level-range', type=str, default='100-110')  # format is like $START_LEVEL-$END_LEVEL
    parser.add_argument('--level-list',  type=str) # format is like 1,3,5,6, this will override level-range
    parser.add_argument('--result-path', type=str, default='./eval_results.csv')
    parser.add_argument('--error-path', type=str, default='./error.log')
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()

def check_env_name(env_name):
    if '-v0' not in env_name:
        raise Exception('Format of env name is wrong. Please include -v0 in env name.')

args = parse_args()
check_env_name(args.env)

def save_error_logs(msg):
    with open(args.error_path, "w") as f:
        f.write(msg)
        f.write(": ")
        f.write(str(sys.exc_info()[0]))

try:
    from user_solution import UserPolicy
except:
    save_error_logs("Error importing user policy")
    raise

def parse_levels(args):
    if args.level_list is not None:
        s = args.level_list
        levels = s.split(',')
        levels = [int(x) for x in levels]
        return levels
    else:
        s = args.level_range
        # s is like 100-200
        if '-' not in s:
            raise Exception('Incorrect level range format, it should be like 100-200.')
        a, b = s.split('-')
        a, b = int(a), int(b)
        return list(range(a,b))


def evaluate_random_policy():
    policy = RandomPolicy(args.env)
    e = Evaluator(args.env, policy)
    level_list = parse_levels(args)
    e.run(level_list, args.vis)
    e.export_to_csv(args.result_path)

def evaluate_user_policy():
    policy = UserPolicy(args.env)
    e = Evaluator(args.env, policy)
    level_list = parse_levels(args)
    e.run(level_list, args.vis)
    e.export_to_csv(args.result_path)


if __name__ == '__main__':
    # evaluate_random_policy()
    try:
        evaluate_user_policy()
    except:
        save_error_logs("Error evaluating user policy")
        raise
