from stable_baselines3.common.callbacks import BaseCallback
import shutil
import time
import os

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        source1 = './tutorial.py'
        self.save_freq = save_freq
        self.save_path = save_path + '/' + time.strftime("%Y-%m-%d-%H-%M")
        os.makedirs(self.save_path)
        shutil.copy2(source1, self.save_path + '/env.py')
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # dones 배열이 True인 경우 에피소드 종료를 의미
        if self.locals['dones'][0]:
            self.episode_counter += 1
            if self.episode_counter % self.save_freq == 0:
                now_time = time.strftime("%Y-%m-%d-%H-%M-%S")
                save_path_detail = self.save_path + '/' + now_time
                os.makedirs(save_path_detail)
                self.model.save(save_path_detail)
                print(f"Saved model checkpoint to {save_path_detail} after episode {self.episode_counter}")
        return True