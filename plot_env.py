import gym
# import shadowhand_gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config


env_id = "KukaLinearLift"
env = GymWrapper(
    suite.make(
        env_id,
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

obs = env.reset()
done = False
while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print(reward)
env.close()
