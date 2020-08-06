from gym.envs.registration import register

register(
    id='SpaceRobot-v0',
    entry_point='spacecraftRobot.envs:SpaceRobotEnv',
    max_episode_steps=75,
)