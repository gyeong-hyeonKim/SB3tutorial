import gymnasium as gym
import numpy as np
import sys
from tutorial import GoLeftEnv
from Callback import SaveModelAndDynamicOUCallback

INITIAL_NORMAL_SIGMA = 0.5
FINAL_SIGMA = 0.0
DECAY_EPISODES = 1000

from stable_baselines3.common.noise import NormalActionNoise
from record import record_video
from record import show_videos


from stable_baselines3 import PPO

def main():
    env = GoLeftEnv()

    model = PPO("MlpPolicy", env, verbose =1)
    model.learn(total_timesteps=10_000)
    video_env = GoLeftEnv(render_mode="rgb_array")
    record_video(video_env, model, video_length=500, prefix="go-left")
    show_videos("videos", prefix="go-left")

if __name__ == "__main__":
    main()