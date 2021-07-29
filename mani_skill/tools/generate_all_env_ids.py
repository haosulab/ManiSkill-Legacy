import gym
import mani_skill.env

task_names = [
    'OpenCabinetDoor',
    'OpenCabinetDrawer',
    'PushChair',
    'MoveBucket',
]

def is_mani_skill(env_spec):
    for task_name in task_names:
        if task_name in env_spec.id:
            return True
    return False

all_envs = gym.envs.registry.all()
mani_skill_envs = [env_spec.id for env_spec in all_envs if is_mani_skill(env_spec)] 

with open("available_environments.txt", "w") as f:
    for env_id in mani_skill_envs:
        f.write(env_id+'\n')
