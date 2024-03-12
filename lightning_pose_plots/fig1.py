import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from lightning_pose_plots import dataset_info as dinfo
from lightning_pose_plots.utilities import (
    cleanaxis,
    load_single_model_video_predictions_from_parquet,
)


def plot_figure1_traces(data_dir):

    dataset_info = {
        'mirror-mouse': {
            'desc': 'Mouse locomotion\n(Warren et al., 2021)\n\n17 keypoints (2 views)',
            'frame_InD': 'labeled-data_InD/180605_000/img076129.png',
            'frame_OOD': 'labeled-data_OOD/180607_004/img018677.png',
            'markersize': 6,
            'csv_InD': 'labels_InD.csv',
            'csv_OOD': 'labels_OOD.csv',
        },
        'mirror-fish': {
            'desc': 'Freely swimming mormyrid fish\n(Pedraja et al.)\n\n51 keypoints (3 views)',
            'frame_InD': 'labeled-data_InD/20201001_Hank/img122280.png',
            'frame_OOD': 'labeled-data_OOD/20210128_Raul/img031257.png',
            'markersize': 1,
            'csv_InD': 'labels_InD.csv',
            'csv_OOD': 'labels_OOD.csv',
        },
        'crim13': {
            'desc': 'Resident-intruder assay\n(Burgos-Artizzu et al., 2012)\n\n14 keypoints (2 animals)',
            'frame_InD': 'labeled-data_InD/012609_A29_Block12_BCma1_t/img00007407.png',
            'frame_OOD': 'labeled-data_OOD/030609_A25_Block11_BCfe1_t/img00009929.png',
            'markersize': 1,
            'csv_InD': 'labels_InD.csv',
            'csv_OOD': 'labels_OOD.csv',
        },
        'ibl-pupil': {
            'desc': 'Mouse pupil tracking\n(IBL 2023)\n\n4 keypoints',
            'frame_InD': 'labeled-data/_iblrig_leftCamera.raw.03cd8002-f8db-42a9-9fcc-51a114661d50_eye/img00064461.png',
            'frame_OOD': 'labeled-data/d9f0c293-df4c-410a-846d-842e47c6b502_left/img00083861.png',
            'markersize': 5,
            'csv_InD': 'CollectedData.csv',
            'csv_OOD': 'CollectedData_new.csv',
        },
        'ibl-paw': {
            'desc': 'Mouse perceptual decision-making\n(IBL 2023)\n\n2 keypoints',
            'frame_InD': 'labeled-data/6c6983ef-7383-4989-9183-32b1a300d17a_iblrig_leftCamera.paws_downsampled_20min/img00000018.png',
            'frame_OOD': 'labeled-data/30e5937e-e86a-47e6-93ae-d2ae3877ff8e_left/img00026312.png',
            'markersize': 10,
            'csv_InD': 'CollectedData.csv',
            'csv_OOD': 'CollectedData_new.csv',
        },
    }

    fig = plt.figure(figsize=(2.4 * len(dataset_info), 6))
    gs = gridspec.GridSpec(4, len(dataset_info), figure=fig, wspace=0.2, hspace=0.1,
                           height_ratios=[0.3, 0.8, 0.8, 1])

    color = [250 / 255, 229 / 255, 64 / 255]
    colors_tab10 = sns.color_palette("tab10")

    for d, (dataset_name, info) in enumerate(dataset_info.items()):

        c = 0  # axis index

        # -----------------------------
        # text
        # -----------------------------
        ax = fig.add_subplot(gs[c, d])
        ax.axis('off')
        ax.text(0.5, 0.5, info['desc'], ha='center', va='center', fontsize=8)
        c += 1

        # -----------------------------
        # frames
        # -----------------------------
        for s, split in enumerate(['InD', 'OOD']):

            ax = fig.add_subplot(gs[c, d])

            # frame
            frame_path = os.path.join(data_dir, dataset_name, info[f'frame_{split}'])
            fr_ = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            ax.imshow(fr_, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors_tab10[s])
                spine.set_linewidth(4)

            # markers
            csv_path = os.path.join(data_dir, dataset_name, info[f'csv_{split}'])
            df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            kps = np.array([d[1] for d in df.columns[::2]])
            markers = df.loc[info[f'frame_{split}']].to_numpy().reshape(-1, 2)
            plt.plot(markers[:, 0], markers[:, 1], '.', markersize=info['markersize'], color=color)

            # skeleton
            skeleton = dinfo[dataset_name]['skeleton']
            if len(skeleton) > 0:
                for s in skeleton:
                    i0 = np.where(kps == s[0])[0][0]
                    i1 = np.where(kps == s[1])[0][0]
                    ax.plot([markers[i0, 0], markers[i1, 0]], [markers[i0, 1], markers[i1, 1]],
                            '-', color=color, linewidth=1)
            c += 1

        # -----------------------------
        # sample efficiency curves
        # -----------------------------
        ax = fig.add_subplot(gs[3, d])

        data_path = os.path.join(
            data_dir, 'results_dataframes', f'{dataset_name}_sample_efficiency.pqt')
        df = pd.read_parquet(data_path)
        # take mean over labeled frames
        df = df.groupby(
            ['set', 'distribution', 'train_frames', 'rng_seed_data_pt']).mean().reset_index()
        train_frames_list = np.sort([int(t) for t in df.train_frames.unique()])

        g = sns.lineplot(
            x='train_frames',
            y='mean',
            hue='distribution',
            data=df[df.set == 'test'],
            ax=ax,
            legend=True if d == 0 else False,
        )

        if d == 0:
            ax.set_ylabel('Pixel error')
        else:
            ax.set_ylabel('')
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([int(s) for s in train_frames_list])
        ax.set_xticklabels(train_frames_list)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.set_xlabel('Training frames')
        cleanaxis(ax)

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'fig1c_sample_efficiency.pdf'), dpi=300)
    plt.close()


def plot_figure1_sample_efficiency(data_dir):

    # ----------------------------------------------------------------
    # set params
    # ----------------------------------------------------------------
    model_type = 'dlc'
    train_frames = '1'
    video_name = '180609_000_25000_27000'
    video_offset = 25000
    idxs = np.arange(500, 775)
    keypoint = 'paw1LH_top'

    framerate_og = 250
    colors = ["tab:red", "tab:blue", "tab:green", "tab:gray", "tab:brown"]
    rng_seeds = ['0', '1', '2', '3', '4']

    # ----------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------
    preds = {
        'x-coord': [],
        'y-coord': [],
        'Conf': [],
    }
    df_path = os.path.join(data_dir, 'results_dataframes',
                           f'mirror-mouse_video_preds_{model_type}.pqt')
    for rng_seed in rng_seeds:
        xs, ys, ls, marker_names = load_single_model_video_predictions_from_parquet(
            filepath=df_path,
            video_name=video_name,
            rng_seed_data_pt=rng_seed,
            train_frames=train_frames,
            model_type=model_type,
        )
        kp_idx = np.where(np.array(marker_names) == keypoint)[0]
        preds['x-coord'].append(xs[:, kp_idx])
        preds['y-coord'].append(ys[:, kp_idx])
        preds['Conf'].append(ls[:, kp_idx])

    # ----------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.1, height_ratios=[1, 1, 0.5])
    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(gs[i]))

    # loop over traces to plot (x, y, likelihood)
    for c, coord in enumerate(['x-coord', 'y-coord', 'Conf']):
        # loop over rng seeds
        for vals, color in zip(preds[coord], colors):
            axes[c].plot(
                (video_offset + idxs) / framerate_og,
                vals[idxs],
                color=color,
                linewidth=0.5,
            )
        axes[c].set_ylabel(coord)
        # cleanup
        if c == 2:
            axes[c].set_xlabel('Time (s)')
        else:
            axes[c].set_xticks([])
            axes[c].set_xlabel('')
        if c == 0:
            axes[c].set_title(keypoint)
        cleanaxis(axes[c])

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'fig1b_traces.pdf'), dpi=300)
    plt.close()


def plot_figure1(data_dir):

    # plot traces from multiple models
    plot_figure1_traces(data_dir=data_dir)

    # plot sample efficiency curves
    plot_figure1_sample_efficiency(data_dir=data_dir)
