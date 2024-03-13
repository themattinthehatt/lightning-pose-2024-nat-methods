import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Figure plotting')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--figure', type=str)
parser.add_argument('--dataset', default='mirror-mouse', type=str)
parser.add_argument('--format', default='png', type=str)

args = parser.parse_args()
data_dir = args.data_dir
figure = args.figure
dataset = args.dataset
format = args.format

# if plotting all figures, plot all datasets as well
if figure == 'all':
    dataset_list = ['mirror-mouse', 'mirror-fish', 'crim13']
    dataset_list_ibl = ['ibl-pupil', 'ibl-paw']
else:
    dataset_list = [dataset]
    dataset_list_ibl = [dataset]

if figure == '1' or figure == 'all':
    from lightning_pose_plots.fig1 import plot_figure1
    print('plotting figure 1...', end='', flush=True)
    plot_figure1(data_dir=data_dir, format=format)
    print('done')

if figure == '2' or figure == 'all':
    from lightning_pose_plots.fig2 import plot_figure2
    print('plotting figure 2...', end='', flush=True)
    plot_figure2(data_dir=data_dir, format=format)
    print('done')

if figure == '3' or figure == 'all':
    from lightning_pose_plots.fig3 import plot_figure3
    for dataset_name in dataset_list:
        print(
            f'plotting figure 3 for {dataset_name}; this will take some time...',
            end='', flush=True,
        )
        plot_figure3(data_dir=data_dir, dataset_name=dataset_name, format=format)
        print('done')

if figure == '4' or figure == 'all':
    from lightning_pose_plots.fig4 import plot_figure4
    for dataset_name in dataset_list:
        print(
            f'plotting figure 4 for {dataset_name}; this will take some time...',
            end='', flush=True,
        )
        plot_figure4(data_dir=data_dir, dataset_name=dataset, format=format)
        print('done')

# if figure == '5' or figure == 'all':
#     from lightning_pose_plots.fig5 import plot_figure5
#     for dataset_name in dataset_list_ibl:
#         print(f'plotting figure 5 for {dataset_name}...', end='', flush=True)
#         plot_figure5(data_dir=data_dir, dataset_name=dataset_name, format=format)
#         print('done')
