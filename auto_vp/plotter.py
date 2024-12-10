import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import torch

from datetime import datetime

class Plotter:
    def __init__(self):
        self.start(" ")

    def start(self, msg):
        self.message = msg
        self.epoch = 0
        self.loss = 0
        self.loss_tea = 0
        self.loss_stu = 0
        self.loss_distill = 0
        self.loss_train = 0
        self.loss_test = 0
        self.acc_train = 0
        self.acc_test = 0
        self.target_dict = {}

    def log(self, *args):
        if len(args) == 1:
            if isinstance(args[0], str):
                print(args[0])
        elif len(args) == 2:
            epoch, data_dict = args
            if isinstance(epoch, int) and isinstance(data_dict, dict):
                for key, value in data_dict.items():
                    # 如果 target_dict 中没有这个 key，则初始化为一个空字典
                    if key not in self.target_dict:
                        self.target_dict[key] = {}
                    # 更新或添加 target_dict 中的值
                    self.target_dict[key][epoch] = value

    def finish(self, save_path="."):
        os.makedirs(save_path, exist_ok=True)
        # 如果既不是文件夹也不是文件，使用当前逻辑
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_file_path = os.path.join(save_path, f"log_{current_time}.txt")

        # Check each key in the first dictionary entry to set up separate plots
        for key, tmp_value in self.target_dict.items():
            data_dict = tmp_value
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = sorted(data_dict.keys())
            # values = [tmp_value[epoch] for epoch in epochs]
            values = [data_dict[epoch].cpu().detach().tolist() if isinstance(data_dict[epoch], torch.Tensor)
                      else data_dict[epoch] for epoch in epochs]
            ax.plot(epochs, values, label=key)
            max_index = np.argmax(values)
            min_index = np.argmin(values)
            ax.set_xlabel(f'Epoch, max={max(values):.4f}/{epochs[max_index]}, min={min(values):.4f}/{epochs[min_index]}')
            ax.set_ylabel(key)
            ax.legend()
            ax.grid(True)
            plt.title(f'{self.message} - {key}')
            plt.savefig(os.path.join(save_path, f"{key}_{current_time}.png"))  # Save the figure
            plt.close(fig)  # Close the figure to free up memory

            # Print epochs and values
            with open(txt_file_path, 'a') as f:
                f.write(f"Map: {key}, Length: {len(values)}\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Values: {values}\n")
                f.write("\n")

plotter = Plotter()

if __name__ == '__main__':
    # Example usage:
    plotter = Plotter()
    plotter.start("ffdfgdgdfg")
    plotter.log("Starting the training process...")
    plotter.log(1, {'loss': 0.5, 'acc_train': 0.8, 'loss_test': 0.45})
    plotter.log(2, {'lfoss': 0.4, 'acc_train': 0.85, 'loss_test': 0.42})
    plotter.log(3, {'loss': 0.35, 'acc_train': 0.88, 'loss_test': 0.40})
    plotter.log(5, {'loss': 0.5, 'acc_train': 0.8, 'loss_test': 0.45})
    plotter.log(8, {'lfoss': 0.4, 'acc_train': 0.85, 'loss_test': 0.42})
    plotter.log(6, {'lfoss': 0.35, 'acc_train': 0.88, 'loss_test': 0.40})
    plotter.finish()
