from stable_baselines3.common.callbacks import BaseCallback
import shutil
import time
import os
import numpy as np

class SaveModelAndDynamicOUCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, action_noise, initial_sigma: float, final_sigma: float, decay_episodes: int, verbose=1):
        super(SaveModelAndDynamicOUCallback, self).__init__(verbose)
        source1 = './tutorial.py'
        
        # 모델 저장 관련 설정
        self.save_freq = save_freq
        self.save_path = save_path + '/' + time.strftime("%Y-%m-%d-%H-%M")
        os.makedirs(self.save_path)
        shutil.copy2(source1, self.save_path + '/env.py')
        
        # OU Noise 관련 설정
        self.action_noise = action_noise
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_episodes = decay_episodes
        self.episode_counter = 0  # 에피소드 수를 트래킹

    def _on_step(self) -> bool:
        # 에피소드가 끝날 때마다 동작
        if self.locals['dones'][0]:
            self.episode_counter += 1
            
            # 모델 저장
            if self.episode_counter % self.save_freq == 0:
                now_time = time.strftime("%Y-%m-%d-%H-%M-%S")
                save_path_detail = self.save_path + '/' + now_time
                os.makedirs(save_path_detail)
                self.model.save(save_path_detail)
                print(f"Saved model checkpoint to {save_path_detail} after episode {self.episode_counter}")
            
            # OU Noise의 sigma 업데이트
            progress = min(1.0, self.episode_counter / self.decay_episodes)
            new_sigma = self.initial_sigma + progress * (self.final_sigma - self.initial_sigma)
            self.action_noise._sigma = new_sigma * np.ones_like(self.action_noise._sigma)
            
            if self.verbose > 0:
                print(f"Episode: {self.episode_counter}, Updated OU Noise Sigma: {new_sigma:.4f}")
        
        return True
