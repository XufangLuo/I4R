import os
import json
from torch import save as torch_save


folders = ["models", "NSSV"]


class Logger:
    def __init__(self, dir_name, dir_pre="./logs/"):
        if os.path.exists(dir_pre + dir_name):
            suffix_count = 1
            while True:
                if os.path.exists(dir_pre + dir_name + " - " + str(suffix_count)):
                    suffix_count += 1
                else:
                    os.makedirs(dir_pre + dir_name + " - " + str(suffix_count))
                    self.dir = dir_pre + dir_name + " - " + str(suffix_count)
                    break
        else:
            os.makedirs(dir_pre + dir_name)
            self.dir = dir_pre + dir_name

        for fol in folders:
            if not os.path.exists(self.dir + "/" + fol):
                os.makedirs(self.dir + "/" + fol)

    def init_console_log_file(self):
        f = open(self.dir + "/console.log", 'w')
        return f

    def init_one_sigmas_file(self, rank):
        f = open(self.dir + "/NSSV/" + str(rank) + '.txt', 'w')
        return f

    def write_sigma(self, f, count, sigma):
        f.write(json.dumps({"count": count, "sigma": sigma}))
        f.write('\n')
        f.flush()

    def write_results_log(self, f, time, num_steps, FPS, episode_reward, episode_len,
                          episode_rewards_list = None, episode_length_list = None):
        if episode_rewards_list is None:
            res = {
                "time": time,
                "num_steps": num_steps,
                "FPS": FPS,
                "avg_episode_reward": episode_reward,
                "avg_episode_len": episode_len
            }
        else:
            res = {
                "time": time,
                "num_steps": num_steps,
                "FPS": FPS,
                "avg_episode_reward": episode_reward,
                "avg_episode_len": episode_len,
                "episode_rewards_list": episode_rewards_list,
                "episode_length_list": episode_length_list
            }
        f.write(json.dumps(res))
        f.write('\n')
        f.flush()

    def save_policy_model(self, model, count):
        torch_save(model.state_dict(), self.dir + "/models/" + str(count))
