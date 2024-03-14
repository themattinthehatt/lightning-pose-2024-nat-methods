import argparse
import os
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Figure plotting')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--figure', type=str)
parser.add_argument('--dataset', default='mirror-mouse', type=str)
parser.add_argument('--format', default='pdf', type=str)

# extract command line args
args = parser.parse_args()
data_dir = args.data_dir
figure = args.figure
dataset = args.dataset
format = args.format

# define where figures are saved
save_dir = os.path.join(data_dir, 'figures')

non_ibl_datasets = ['mirror-mouse', 'mirror-fish', 'crim13']
ibl_datasets = ['ibl-pupil', 'ibl-paw']

# if plotting all figures, plot all datasets as well
if figure == 'all':
    dataset_list = non_ibl_datasets
    dataset_list_ibl = ibl_datasets
else:
    dataset_list = [dataset]
    dataset_list_ibl = [dataset]

if figure == '1' or figure == 'all':
    from lightning_pose_plots.fig1 import plot_figure1
    print('plotting figure 1...', end='', flush=True)
    plot_figure1(data_dir=data_dir, save_dir=save_dir, format=format)
    print('done')

if figure == '2' or figure == 'all':
    from lightning_pose_plots.fig2 import plot_figure2
    print('plotting figure 2...', end='', flush=True)
    plot_figure2(data_dir=data_dir, save_dir=save_dir, format=format)
    print('done')

if figure == '3' or figure == 'all':
    from lightning_pose_plots.fig3 import plot_figure3
    for dataset_name in dataset_list:
        if dataset_name not in non_ibl_datasets:
            print(f'dataset "{dataset_name}" not in {non_ibl_datasets}; skipping')
            continue
        print(
            f'plotting figure 3 for {dataset_name}; this will take some time...',
            end='', flush=True,
        )
        plot_figure3(
            data_dir=data_dir, save_dir=save_dir, dataset_name=dataset_name, format=format,
        )
        print('done')

if figure == '4' or figure == 'all':
    from lightning_pose_plots.fig4 import plot_figure4
    for dataset_name in dataset_list:
        if dataset_name not in non_ibl_datasets:
            print(f'dataset "{dataset_name}" not in {non_ibl_datasets}; skipping')
            continue
        print(
            f'plotting figure 4 for {dataset_name}; this will take some time...',
            end='', flush=True,
        )
        plot_figure4(
            data_dir=data_dir, save_dir=save_dir, dataset_name=dataset_name, format=format,
        )
        print('done')

if figure == '5' or figure == 'all':
    from lightning_pose_plots.fig5 import plot_figure5
    for dataset_name in dataset_list_ibl:
        if dataset_name not in ibl_datasets:
            print(f'dataset "{dataset_name}" not in {ibl_datasets}; skipping')
            continue
        print(f'plotting figure 5 for {dataset_name}...', end='', flush=True)
        plot_figure5(
            data_dir=data_dir, save_dir=save_dir, dataset_name=dataset_name, format=format,
        )
        print('done')
