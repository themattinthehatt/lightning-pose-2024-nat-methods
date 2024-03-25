import numpy as np
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import seaborn as sns

from lightning_pose_plots import dataset_info
from lightning_pose_plots.utilities import cleanaxis


def plot_box(ax, df, y, ylabel, fontsize_label=10, logy=False):
    ax = sns.boxplot(
        x='tracker', y=y, data=df, fliersize=0,
        medianprops=dict(color='red', alpha=1.0, linewidth=3),
        boxprops=dict(facecolor='none', edgecolor='k'),
        linewidth=2, width=0.3, ax=ax,
    )
    sns.scatterplot(x='tracker', y=y, data=df, s=3, color='k', alpha=1.0, ax=ax)
    vals_all = []
    for eid in df.eid.unique():
        vals = df[df.eid == eid][y].to_numpy()
        vals = vals[~np.isnan(vals)]
        ax.plot([0, 1, 2], vals, 'k', alpha=0.3)
        vals_all.append(vals)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    if logy:
        ax.set_yscale('log')
    ax.set_xticklabels(['DLC', 'LP', 'LP+EKS'])
    ax.tick_params(axis='both', which='major', labelsize=fontsize_label)
    cleanaxis(ax)
    return np.array(vals_all)


def plot_figure6_pupil(
    save_file,
    data_dir,
    n_skip_scatter=2,  # scatter on frame
    n_skip_scatter_r=10,  # vert vs horiz scatter
):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    labels_fontsize = 10
    keypoint_names = dataset_info['ibl-pupil']['keypoints']
    colors = {
        'pupil_bottom_r': '#AB63FA',
        'pupil_left_r': '#FFA15A',
        'pupil_right_r': '#19D3F3',
        'pupil_top_r': '#FF6692',
    }

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
            fontsize=labels_fontsize,
        )

        # left/right vs top/bottom pupil diameters
        ax = fig.add_subplot(gs[1, idx])
        x = info['data']['diam_lr'][::n_skip_scatter_r]
        y = info['data']['diam_tb'][::n_skip_scatter_r]
        if tracker == 'lp+ks':
            xy = pd.DataFrame({
                'horz': x + 0.1 * np.random.randn(x.shape[0]),
                'vert': y + 0.1 * np.random.randn(y.shape[0]),
            })
        else:
            xy = pd.DataFrame({'horz': x, 'vert': y})
        sns.kdeplot(
            data=xy, x='horz', y='vert', fill=True, bw_adjust=0.75, ax=ax, color='gray',
        )
        r, _, lo, hi = info['data']['diam_corrs']
        ax.text(
            0.05, 0.95, 'Pearson $r$ = %1.2f' % r,
            transform=ax.transAxes, va='top', fontsize=labels_fontsize,
        )
        ax.set_xlabel('Horizontal diameter', fontsize=labels_fontsize)
        if idx == 0:
            ax.set_ylabel('Vertical diameter', fontsize=labels_fontsize)
        else:
            ax.set_ylabel('')
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
        ax.axvline(x=0, label='Reward delivery', linestyle='--', c='k')
        ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
        ax.set_xlabel('Time (s)', fontsize=labels_fontsize)
        if idx == 0:
            ax.set_ylabel('Normalized diameter (pix)', fontsize=labels_fontsize)
            ax.text(
                0.27, 0.98, 'Reward delivery',
                transform=ax.transAxes, va='top', fontsize=labels_fontsize,
            )
        cleanaxis(ax)
        ax.set_title('Trial consistency = %1.2f' % info['peth_snr'], fontsize=labels_fontsize)
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
        logy = True if y == 'trial_consistency' else False
        vals = plot_box(
            ax=ax, df=df, y=y, ylabel=ylabel, logy=logy, fontsize_label=labels_fontsize,
        )

        # # print stats
        # print(ylabel)
        # s01 = wilcoxon(vals[:, 1], y=vals[:, 0], alternative='greater')
        # s02 = wilcoxon(vals[:, 2], y=vals[:, 0], alternative='greater')
        # s12 = wilcoxon(vals[:, 2], y=vals[:, 1], alternative='greater')
        # means = np.nanmean(vals, axis=0)
        # sems = np.nanstd(vals, axis=0) / np.sqrt(vals.shape[0])
        # print(f'0: {round(means[0], 2)}+/-{round(sems[0], 2)}')
        # print(f'1: {round(means[1], 2)}+/-{round(sems[1], 2)}')
        # print(f'2: {round(means[2], 2)}+/-{round(sems[2], 2)}')
        # print(f'0-1: p={s01.pvalue}')
        # print(f'0-2: p={s02.pvalue}')
        # print(f'1-2: p={s12.pvalue}')
        # print()

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    plt.suptitle('ibl-pupil dataset', fontsize=labels_fontsize + 6)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=300)
    plt.close()


def plot_figure6_paw(
    save_file,
    data_dir,
    n_skip_scatter=100,  # scatter on frame
    n_skip_scatter_cca=10,  # cca scatter
):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    keypoint_names = dataset_info['ibl-paw']['keypoints']
    labels_fontsize = 10
    colors = {
        'paw_l': '#EF553B',
        'paw_r': '#00CC96',
    }

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    single_sess_path = os.path.join(
        data_dir,
        'results_dataframes',
        'ibl-paw_tracker_comparisons_1b715600-0cbc-442c-bd00-5b0ac2865de1.pkl'
    )
    single_sess_info = pd.read_pickle(single_sess_path)

    tracker_metrics_path = os.path.join(
        data_dir, 'results_dataframes', 'ibl-paw_tracker_metrics.csv',
    )
    tracker_metrics_df = pd.read_csv(tracker_metrics_path)

    decoding_metrics_path = os.path.join(
        data_dir, 'results_dataframes', 'ibl-paw_decoding_metrics.csv',
    )
    decoding_metrics_df = pd.read_csv(decoding_metrics_path)

    # ---------------------------------------------------
    # plot figure
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(12, 13))
    gs = fig.add_gridspec(
        4, 5, hspace=0.4, wspace=0.4,
        width_ratios=[1, 1, 1, 0.75, 0.75],
        height_ratios=[1.2, 1, 1, 1],
    )

    # -------------------------------------------------
    # single session example
    # -------------------------------------------------
    trackers = ['dlc', 'lp', 'lp+ks']
    for idx, tracker in enumerate(trackers):

        paw = 'right_paw'  # just plot results on right paw
        info = single_sess_info[tracker][paw]

        # scatter keypoints on frame
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, idx], hspace=0)
        for v, view in enumerate(['left', 'right']):
            ax = fig.add_subplot(gs1[v])
            if view == 'right':
                frame = np.fliplr(info['frames'][view])
            else:
                frame = info['frames'][view]
            ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
            for kp in keypoint_names:
                if view == 'right':
                    kp_ = 'paw_l' if kp == 'paw_r' else 'paw_r'
                    xs = 128 - info['data'][view]['dlc_df'][f'{kp_}_x'][::n_skip_scatter]
                else:
                    kp_ = kp
                    xs = info['data'][view]['dlc_df'][f'{kp_}_x'][::n_skip_scatter]
                ys = info['data'][view]['dlc_df'][f'{kp_}_y'][::n_skip_scatter]
                ax.scatter(xs, ys, alpha=0.5, s=0.01, label=kp, c=colors[kp])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, info['frames'][view].shape[1]])
            ax.set_ylim([info['frames'][view].shape[0], 0])
            ax.text(
                0.05, 0.05, f'{view.capitalize()} view',
                transform=ax.transAxes,
                fontsize=labels_fontsize,
                color='w',
            )
            ax.set_title(
                'LP+EKS' if tracker == 'lp+ks' else tracker.upper(), fontsize=labels_fontsize,
            )

        # cca projections
        ax = fig.add_subplot(gs[1, idx])
        x = info['data']['both']['lcam_cca0'][::n_skip_scatter_cca]
        y = info['data']['both']['rcam_cca0'][::n_skip_scatter_cca]
        if tracker == 'lp+ks':
            xy = pd.DataFrame({
                'left': x + 0.01 * np.random.randn(x.shape[0]),
                'right': y + 0.01 * np.random.randn(y.shape[0]),
            })
        else:
            xy = pd.DataFrame({'left': x, 'right': y})
        sns.kdeplot(
            data=xy, x='left', y='right', fill=True, bw_adjust=0.75, ax=ax, color='gray')
        r, _, lo, hi = info['data']['both']['proj_corrs']
        ax.text(
            0.05, 0.95, 'Pearson $r$ = %1.2f' % r,
            transform=ax.transAxes, va='top', fontsize=labels_fontsize,
        )
        ax.set_xlabel('Left video CCA proj', fontsize=labels_fontsize)
        if idx == 0:
            ax.set_ylabel('Right video CCA proj', fontsize=labels_fontsize)
        else:
            ax.set_ylabel('')
        cleanaxis(ax)
        ax.set_title('Left paw' if paw == 'left_paw' else 'Right paw')
        ax.set_aspect('equal')
        ax.set_xlim([-3.2, 3.2])
        ax.set_ylim([-3.2, 3.2])
        ax.plot([-3, 3], [-3, 3], '-k', linewidth=0.25)

        # traces/peth
        ax = fig.add_subplot(gs[2, idx])
        view = 'right'  # just plot results from a single view
        data_ = info['data'][view]
        traces = data_['correct_traces']
        traces_no_mean = traces
        trace_mean = np.nanmedian(traces_no_mean, axis=1)
        ci = 1.96 * np.nanstd(traces_no_mean, axis=1) / np.sqrt(traces_no_mean.shape[1])
        ax.fill_between(
            data_['times'], trace_mean - ci, trace_mean + ci, color='k', alpha=0.25,
        )
        # plot peth
        ax.plot(data_['times'], trace_mean, c='k', label='correct trial')
        ax.axvline(x=0, label='Movement onset', linestyle='--', c='k')
        ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
        ax.set_xlabel('Time (s)', fontsize=labels_fontsize)
        if idx == 0:
            ax.set_ylabel('Paw speed (pix/s)', fontsize=labels_fontsize)
        else:
            ax.set_ylabel('')
        cleanaxis(ax)
        if idx == 1:
            ax.text(
                0.27, 0.98, 'Movement onset',
                transform=ax.transAxes, va='top', fontsize=labels_fontsize,
            )
        ax.set_title(
            'Trial consistency = %1.2f' % info['data'][view]['peth_snr'],
            fontsize=labels_fontsize,
        )
        ax.set_ylim([-20, 250])

    # -------------------------------------------------
    # multi session stats
    # -------------------------------------------------
    for idx_p, paw in enumerate(['right', 'left']):
        for idx, y, ylabel, df in zip(
                [1, 2, 3],
                ['cca_r', 'trial_consistency', 'R2'],
                ['CCA projection corr ($r$)', 'Trial consistency', 'Decoding $R^2$'],
                [tracker_metrics_df, tracker_metrics_df, decoding_metrics_df]
        ):
            ax = fig.add_subplot(gs[idx, 3 + idx_p])
            if df is None:
                continue
            vals = plot_box(
                ax=ax, df=df[df.paw == paw], y=y, ylabel=ylabel, fontsize_label=labels_fontsize,
            )
            if idx_p > 0:
                ax.set_ylabel('')
            if idx == 1:
                ax.set_title(f'{paw.capitalize()} paw')

            # # print stats
            # print(f'{paw}: {ylabel}')
            # s01 = wilcoxon(vals[:, 1], y=vals[:, 0], alternative='greater')
            # s02 = wilcoxon(vals[:, 2], y=vals[:, 0], alternative='greater')
            # s12 = wilcoxon(vals[:, 2], y=vals[:, 1], alternative='greater')
            # means = np.nanmean(vals, axis=0)
            # sems = np.nanstd(vals, axis=0) / np.sqrt(vals.shape[0])
            # print(f'0: {round(means[0], 2)}+/-{round(sems[0], 2)}')
            # print(f'1: {round(means[1], 2)}+/-{round(sems[1], 2)}')
            # print(f'2: {round(means[2], 2)}+/-{round(sems[2], 2)}')
            # print(f'0-1: p={s01.pvalue}')
            # print(f'0-2: p={s02.pvalue}')
            # print(f'1-2: p={s12.pvalue}')
            # print()

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    plt.suptitle('ibl-paw dataset', fontsize=labels_fontsize + 6)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=300)
    plt.close()


def plot_figure6(data_dir, save_dir, dataset_name, format='pdf'):

    if dataset_name == 'ibl-pupil':
        save_file = os.path.join(save_dir, f'fig6_ibl-pupil.{format}')
        plot_figure6_pupil(
            save_file=save_file,
            data_dir=data_dir,
        )
    elif dataset_name == 'ibl-paw':
        save_file = os.path.join(save_dir, f'fig6_ibl-paw.{format}')
        plot_figure6_paw(
            save_file=save_file,
            data_dir=data_dir,
        )
    else:
        raise NotImplementedError(f'dataset {dataset_name} not in [ibl-pupil, ibl-paw]; skipping')
