import os
import gym
import json
# import shadowhand_gym
import robosuite as suite
import torch.cuda
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
from policy.sac import SAC
from types import SimpleNamespace


AGENT = "sac_ckpt_IIWA-extended-KukaLinearLift-Position_30000"
CONFIG = "config"
ENV_ID = "KukaLinearLift"
RANDOM_ACTION = True

AGENT = os.path.join("runs", "iiwa", f"{AGENT}.zip")
CONFIG = os.path.join("runs", "iiwa", f"{CONFIG}.json")


def main(args):
    if "KukaLinearLift" in ENV_ID:
        env = GymWrapper(
            suite.make(
                ENV_ID,
                robots="IIWA_extended",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                horizon=1000,
                controller_configs=load_controller_config(default_controller="OSC_POSITION"),
                renderer="mujoco",
            )
        )
        env._max_episode_steps = 1000
    else:
        env = gym.make(ENV_ID)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_checkpoint(AGENT)

    obs = env.reset()
    done = False
    while not done:
        if RANDOM_ACTION:
            # Random action
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=True)
        obs, reward, done, info = env.step(action)
        env.render()
        print(reward)
    env.close()


if __name__ == "__main__":
    with open(CONFIG, "rb") as f:
        args = SimpleNamespace(**json.load(f))
    args.cuda = True if torch.cuda.is_available() else False

    main(args)
