import numpy as np
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from lightning_pose_plots import dataset_info
from lightning_pose_plots.utilities import cleanaxis


def plot_figure5(data_dir, dataset_name, format='pdf'):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    keypoint_names = dataset_info[dataset_name]['keypoints']
    fontsize_label = 10
    colors = {
        'paw_l': '#EF553B',
        'paw_r': '#00CC96',
        'pupil_bottom_r': '#AB63FA',
        'pupil_left_r': '#FFA15A',
        'pupil_right_r': '#19D3F3',
        'pupil_top_r': '#FF6692',
    }
    n_skip_scatter = 2  # scatter on frame
    n_skip_scatter2 = 10  # vert vs horiz scatter

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    single_sess_path = os.path.join(
        data_dir,
        'results_dataframes',
        'ibl-pupil_tracker_comparisons_d0c91c3c-8cbb-4929-8657-31f18bffc294.pkl'
    )
    single_sess_info = pd.read_pickle(single_sess_path)

    tracker_metrics_path = os.path.join(
        data_dir, 'results_dataframes', 'ibl-pupil_tracker_metrics.csv',
    )
    tracker_metrics_df = pd.read_csv(tracker_metrics_path)

    decoding_metrics_path = os.path.join(
        data_dir, 'results_dataframes', 'ibl-pupil_decoding_metrics.csv',
    )
    decoding_metrics_df = pd.read_csv(decoding_metrics_path)

    # ---------------------------------------------------
    # plot figure
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

    # ----------------------
    # single session example
    # ----------------------
    trackers = ['dlc', 'lp', 'lp+ks']
    for idx, tracker in enumerate(trackers):

        info = single_sess_info[tracker]

        # scatter keypoints on frame
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(info['frame'], cmap='gray', vmin=0, vmax=255)
        for kp in keypoint_names:
            ax.scatter(
                info['data']['dlc_df'][f'{kp}_x'][::n_skip_scatter],
                info['data']['dlc_df'][f'{kp}_y'][::n_skip_scatter],
                alpha=0.5, s=0.01, label=kp, c=colors[kp],
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, info['frame'].shape[1]])
        ax.set_ylim([info['frame'].shape[0], 0])
        ax.set_title(
            'LP+EKS' if tracker == 'lp+ks' else tracker.upper(),
            fontsize=fontsize_label,
        )

        # left/right vs top/bottom pupil diameters
        ax = fig.add_subplot(gs[1, idx])
        x = info['data']['diam_lr'][::n_skip_scatter2]
        y = info['data']['diam_tb'][::n_skip_scatter2]
        if tracker == 'lp+ks':
            xy = pd.DataFrame({
                'horz': x + 0.1 * np.random.randn(x.shape[0]),
                'vert': y + 0.1 * np.random.randn(y.shape[0]),
            })
        else:
            xy = pd.DataFrame({'horz': x, 'vert': y})
        sns.kdeplot(
            data=xy, x='horz', y='vert', fill=True, bw_adjust=0.75, ax=ax, color='gray')
        r, _, lo, hi = info['data']['diam_corrs']
        ax.text(
            0.05, 0.95, 'Pearson $r$ = %1.2f' % r, transform=ax.transAxes, va='top',
            fontsize=fontsize_label)
        ax.set_xlabel('Horizontal diameter', fontsize=fontsize_label)
        ax.set_ylabel('Vertical diameter', fontsize=fontsize_label)
        cleanaxis(ax)
        ax.set_aspect('equal')
        ax.set_xlim([0, 30])
        ax.set_ylim([0, 30])
        ax.plot([2, 28], [2, 28], '-k', linewidth=0.25)

        # traces/peth
        ax = fig.add_subplot(gs[2, idx])
        traces = info['correct_traces']
        # plot individual traces per trial
        traces_no_mean = traces - np.nanmedian(traces, axis=0)
        ax.plot(info['times'], traces_no_mean[:, ::2], c='k', alpha=0.1)
        # plot peth
        trace_mean = np.nanmedian(traces_no_mean, axis=1)
        ax.plot(info['times'], trace_mean, c='r', label='correct trial')
        ax.axvline(x=0, label='Reward delivery', linestyle='--', c='b')
        ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
        ax.set_xlabel('Time (s)', fontsize=fontsize_label)
        if idx == 0:
            ax.set_ylabel('Normalized diameter (pix)', fontsize=fontsize_label)
            ax.text(
                0.27, 0.98, 'Reward delivery', transform=ax.transAxes, va='top',
                fontsize=fontsize_label,
            )
        cleanaxis(ax)
        ax.set_title('Trial consistency = %1.2f' % info['peth_snr'], fontsize=fontsize_label)
        ax.set_ylim([-4, 4])

    # -------------------
    # multi session stats
    # -------------------
    for idx, y, ylabel, df in zip(
            [1, 2, 3],
            ['v_vs_h_diam', 'trial_consistency', 'R2'],
            ['Vert vs Horiz diameter $r$', 'Trial consistency', 'Decoding $R^2$'],
            [tracker_metrics_df, tracker_metrics_df, decoding_metrics_df]
    ):
        ax = fig.add_subplot(gs[idx, 3])
        ax = sns.boxplot(
            x='tracker', y=y, data=df, fliersize=0,
            medianprops=dict(color='red', alpha=1.0, linewidth=3),
            boxprops=dict(facecolor='none', edgecolor='k'),
            linewidth=2, width=0.3, ax=ax,
        )
        sns.scatterplot(x='tracker', y=y, data=df, s=3, color='k', alpha=1.0, ax=ax)
        for eid in df.eid.unique():
            vals = df[df.eid == eid][y].to_numpy()
            vals = vals[~np.isnan(vals)]
            ax.plot([0, 1, 2], vals, 'k', alpha=0.3)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
        if y == 'trial_consistency':
            ax.set_yscale('log')
        ax.set_xticklabels(['DLC', 'LP', 'LP+EKS'])
        ax.tick_params(axis='both', which='major', labelsize=fontsize_label)
        cleanaxis(ax)

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig5_{dataset_name}.{format}'),
        dpi=300,
    )
    plt.close()
