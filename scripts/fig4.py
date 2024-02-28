import argparse

from lightning_pose_plots.fig4 import plot_figure4

parser = argparse.ArgumentParser(description='Figure 4')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str)

args = parser.parse_args()
data_dir = args.data_dir
dataset = args.dataset

plot_figure4(data_dir=data_dir, dataset_name=dataset)
