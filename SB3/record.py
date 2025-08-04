import os
import base64
from pathlib import Path
import gymnasium as gym

from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob(f"{prefix}*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(f"""<video alt="{mp4}" autoplay 
                        loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{video_b64.decode('ascii')}" type="video/mp4" />
                    </video>""")
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(env_to_record, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    vec_env = DummyVecEnv([lambda: env_to_record])
    # Start the video at step=0 and record 500 steps
    video_recorder_env = VecVideoRecorder(
        vec_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = video_recorder_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = video_recorder_env.step(action)

    # Close the video recorder
    video_recorder_env.close()