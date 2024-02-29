import argparse

parser = argparse.ArgumentParser(description='Figure plotting')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--figure', type=str)
parser.add_argument('--dataset', default='mirror-mouse', type=str)

args = parser.parse_args()
data_dir = args.data_dir
figure = args.figure
dataset = args.dataset

# if plotting all figures, plot all datasets as well
if figure == 'all':
    dataset_list = ['mirror-mouse', 'mirror-fish', 'crim13']
    dataset_list_ibl = ['ibl-pupil', 'ibl-paw']
else:
    dataset_list = [dataset]
    dataset_list_ibl = [dataset]

if figure == '1' or figure == 'all':
    from lightning_pose_plots.fig1 import plot_figure1
    plot_figure1(data_dir=data_dir)

if figure == '2' or figure == 'all':
    from lightning_pose_plots.fig2 import plot_figure2
    plot_figure2(data_dir=data_dir)

if figure == '3' or figure == 'all':
    from lightning_pose_plots.fig3 import plot_figure3
    for dataset_name in dataset_list:
        plot_figure3(data_dir=data_dir, dataset_name=dataset_name)

if figure == '4' or figure == 'all':
    from lightning_pose_plots.fig4 import plot_figure4
    for dataset_name in dataset_list:
        plot_figure4(data_dir=data_dir, dataset_name=dataset)

if figure == '5' or figure == 'all':
    from lightning_pose_plots.fig5 import plot_figure5
    for dataset_name in dataset_list_ibl:
        plot_figure5(data_dir=data_dir, dataset_name=dataset_name)
