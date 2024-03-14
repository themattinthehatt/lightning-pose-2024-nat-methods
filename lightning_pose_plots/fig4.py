import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from lightning_pose_plots import model_order, model_colors, dataset_info_fig4
from lightning_pose_plots.utilities import (
    cleanaxis,
    get_frames_from_idxs,
    get_trace_mask,
    load_results_dataframes,
)


def compute_percentiles(arr, std_vals, percentiles):
    num_pts = arr[0]
    vals = []
    prctiles = []
    for p in percentiles:
        v = num_pts * p / 100
        idx = np.argmin(np.abs(arr - v))
        # maybe we don't have enough data
        if idx == len(arr) - 1:
            p_ = arr[idx] / num_pts * 100
        else:
            p_ = p
        vals.append(std_vals[idx])
        prctiles.append(p_)
    return vals, prctiles


def compute_ensemble_var_for_each_pixel_error(
    df_ground_truth,
    df_labeled_preds,
    df_labeled_metrics,
    train_frames,
    models,
    rng_seeds,
    split_set='test',
    distribution='OOD',
):
    """

    Parameters
    ----------
    df_ground_truth : pd.DataFrame
        ground truth predictions
    df_labeled_preds : pd.DataFrame
        model predictions
    df_labeled_metrics : pd.DataFrame
        metrics computed on model predictions
    train_frames : array-like
        list of train_frame values to loop over
    models : array-like
        list of models to compute ensemble variance over
    rng_seeds : array-like
        list of rng seeds to compute ensemble variance over
    split_set : str
        train, val, test
    distribution : str
        InD, OOD

    Returns
    -------
    pd.DataFrame

    """
    df_w_vars = []
    for train_frame_ in train_frames:
        # compute ensemble variance
        preds = []
        for model in models:
            mask = ((df_labeled_preds.set == split_set)
                    & (df_labeled_preds.distribution == distribution)
                    & (df_labeled_preds.train_frames == train_frame_)
                    & (df_labeled_preds.model_type == model)
                    )
            df_tmp = df_labeled_preds[mask]
            for rng in rng_seeds:
                df_tmp1 = df_tmp[df_tmp.rng_seed_data_pt == rng].drop(
                    columns=['set', 'distribution', 'model_path', 'rng_seed_data_pt',
                             'train_frames', 'model_type'])
                assert np.all(df_tmp1.index == df_ground_truth.index)
                # get rid of likelihood
                arr = df_tmp1.to_numpy().reshape(df_tmp1.shape[0], -1, 3)[:, :, :2]
                preds.append(arr[..., None])
        preds = np.concatenate(preds, axis=3)
        net_vars = np.std(preds, axis=-1).mean(axis=-1)
        # record pixel errors along with ensemble variances
        for model in models:
            for rng in rng_seeds:
                mask = ((df_labeled_metrics.set == split_set)
                        & (df_labeled_metrics.distribution == distribution)
                        & (df_labeled_metrics.train_frames == train_frame_)
                        & (df_labeled_metrics.rng_seed_data_pt == rng)
                        & (df_labeled_metrics.metric == 'pixel_error')
                        & (df_labeled_metrics.model_type == model)
                        )
                dfs_all = df_labeled_metrics[mask]
                assert np.all(dfs_all.index == df_ground_truth.index)
                for i, kp in enumerate(dfs_all.columns):
                    if kp in ['model_path', 'rng_seed_data_pt', 'train_frames', 'model_type',
                              'metric', 'set', 'distribution', 'mean']:
                        continue
                    df_w_vars.append(pd.DataFrame({
                        'pixel_error': dfs_all[kp],
                        'ens-std': net_vars[:, i],
                        'keypoint': kp,
                        'rng': rng,
                        'model': model,
                        'train_frames': train_frame_,
                    }, index=df_ground_truth.index))

    df_w_vars = pd.concat(df_w_vars)

    return df_w_vars


def compute_ensemble_var_for_each_metric(
    df_video_preds,
    df_video_metrics,
    train_frames,
    models,
    rng_seeds,
    video_names,
    metric_names,
):
    """

    Parameters
    ----------
    df_video_preds : pd.DataFrame
        model predictions on video data
    df_video_metrics
        metrics computed on model predictions
    train_frames : array-like
        list of train_frame values to loop over
    models : array-like
        list of models to compute ensemble variance over
    rng_seeds : array-like
        list of rng seeds to compute ensemble variance over
    video_names : array-like
        list of video names to loop over
    metric_names : array-like
        list of metrics to compute

    Returns
    -------
    pd.DataFrame

    """

    cols_to_drop_ = ['video_name', 'model_path', 'rng_seed_data_pt', 'train_frames', 'model_type']
    df_w_vars_vids = []
    for video_name_ in video_names:
        for train_frame_ in train_frames:
            # mask predictions
            mask1 = (
                    (df_video_preds.video_name == video_name_)
                    & (df_video_preds.train_frames == train_frame_)
            )
            df_tmp1 = df_video_preds[mask1]
            # mask metrics
            mask1b = (
                    (df_video_metrics.video_name == video_name_)
                    & (df_video_metrics.train_frames == train_frame_)
            )
            df_tmp1b = df_video_metrics[mask1b]
            # compute ensemble variance over all models
            preds = []
            for model in models:
                df_tmp2 = df_tmp1[df_tmp1.model_type == model]
                for rng_seed_ in rng_seeds:
                    preds_curr = df_tmp2[df_tmp2.rng_seed_data_pt == rng_seed_].drop(
                        columns=cols_to_drop_).to_numpy()
                    preds_curr = np.delete(
                        preds_curr, list(range(2, preds_curr.shape[1], 3)), axis=1)
                    preds_curr = np.reshape(preds_curr, (preds_curr.shape[0], -1, 2))
                    preds.append(preds_curr[..., None])
            # concatenate across last dim
            preds = np.concatenate(preds, axis=3)
            # compute variance across x/y
            ens_std = np.std(preds, axis=-1).mean(axis=-1)
            ens_std_mean = np.mean(ens_std, axis=1)
            # record metrics along with ensemble variance
            for model in model_order:
                df_tmp3 = df_tmp1b[df_tmp1b.model_type == model]
                for rng_seed_ in rng_seeds:
                    df_tmp4 = df_tmp3[df_tmp3.rng_seed_data_pt == rng_seed_]
                    for metric_name in metric_names:
                        df_tmp5 = df_tmp4[df_tmp4.metric == metric_name]
                        df_w_vars_vids.append(pd.DataFrame({
                            'value': df_tmp5['mean'],
                            'metric': metric_name,
                            'ens-std': ens_std_mean,
                            'rng': rng_seed_,
                            'model': model,
                            'video_name': video_name_,
                            'train_frames': train_frame_,
                        }, index=df_tmp5['mean'].index))

    df_w_vars_vids = pd.concat(df_w_vars_vids)

    return df_w_vars_vids


def plot_figure4(
    data_dir,
    dataset_name,
    format='pdf',
    rng_seed='0',
    models_to_compare=['baseline', 'semi-super context'],
):

    labels_fontsize = 10

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    train_frames = dataset_info_fig4[dataset_name]['train_frames']
    cols_to_drop = dataset_info_fig4[dataset_name]['cols_to_drop']
    cols_to_keep = dataset_info_fig4[dataset_name]['cols_to_keep']
    vid_name = dataset_info_fig4[dataset_name]['vid_name']
    vid_name_load = dataset_info_fig4[dataset_name]['vid_name_load']
    frames_offset = dataset_info_fig4[dataset_name]['frames_offset']
    keypoint = dataset_info_fig4[dataset_name]['keypoint']
    time_window = dataset_info_fig4[dataset_name]['time_window_tr']
    time_window_frames = dataset_info_fig4[dataset_name]['time_window_fr']
    train_frames_list = dataset_info_fig4[dataset_name]['train_frames_list']
    train_frames_list_actual = dataset_info_fig4[dataset_name]['train_frames_actual']
    std_vals = dataset_info_fig4[dataset_name]['std_vals']
    yticks = dataset_info_fig4[dataset_name]['yticks']

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    vid_dir = os.path.join(data_dir, dataset_name, 'videos_OOD')
    vid_file = os.path.join(vid_dir, f'{vid_name_load}.mp4')
    cap = cv2.VideoCapture(vid_file)

    # load ground truth labels
    df_ground_truth = pd.read_csv(
        os.path.join(data_dir, dataset_name, 'labels_OOD.csv'),
        index_col=0,
        header=[1, 2],
    )
    # update relative paths in labeled data to match model results
    df_ground_truth.index = [
        p.replace('labeled-data_OOD/', 'labeled-data/') for p in df_ground_truth.index
    ]
    keypoint_names = df_ground_truth.columns.get_level_values(0)

    # load model predictions on both labeled frames and unlabeled videos
    df_labeled_preds, df_labeled_metrics, df_video_preds, df_video_metrics = \
        load_results_dataframes(
            results_df_dir=os.path.join(data_dir, 'results_dataframes'),
            dataset_name=dataset_name,
        )

    # drop keypoints
    df_ground_truth = df_ground_truth.drop(columns=cols_to_drop)
    df_labeled_preds = df_labeled_preds.drop(columns=cols_to_drop)
    df_labeled_metrics = df_labeled_metrics.drop(columns=cols_to_drop)
    df_video_preds = df_video_preds.drop(columns=cols_to_drop)
    df_video_metrics = df_video_metrics.drop(columns=cols_to_drop)
    # recompute means
    if len(cols_to_keep) > 0:
        df_video_metrics.loc[:, 'mean'] = df_video_metrics.loc[:, cols_to_keep].mean(axis=1)
        df_labeled_metrics.loc[:, 'mean'] = df_labeled_metrics.loc[:, cols_to_keep].mean(axis=1)

    metric_names = df_video_metrics.metric.unique()
    video_names = df_video_metrics[df_video_metrics.model_type == 'baseline'].video_name.unique()
    rng_seeds = df_video_metrics.rng_seed_data_pt.unique()

    # ---------------------------------------------------
    # compute pixel error as a function of ens std dev
    # ---------------------------------------------------
    df_w_vars = compute_ensemble_var_for_each_pixel_error(
        df_ground_truth=df_ground_truth,
        df_labeled_preds=df_labeled_preds,
        df_labeled_metrics=df_labeled_metrics,
        train_frames=train_frames_list,
        models=model_order,
        rng_seeds=rng_seeds,
        split_set='test',
        distribution='OOD',
    )

    # now compute pixel error for all keypoints >= a given ensemble variance
    n_points_dict = {
        t: {m: np.nan * np.zeros_like(std_vals) for m in model_order} for t in train_frames_list}
    df_line = []
    for train_frame_ in train_frames_list:
        df_w_varst = df_w_vars[df_w_vars.train_frames == train_frame_]
        for s, std in enumerate(std_vals):
            df_tmp_ = df_w_varst[df_w_varst['ens-std'] > std]
            for model in model_order:
                for rng in rng_seeds:
                    d = df_tmp_[(df_tmp_.model == model) & (df_tmp_.rng == rng)]
                    n_points = np.sum(~d['pixel_error'].isna())
                    n_points_dict[train_frame_][model][s] = n_points
                    index = []
                    for row, k in zip(d.index, d['keypoint'].to_numpy()):
                        index.append(row + f'_{model}_{s}_{train_frame_}_{k}_{rng}')
                    df_line.append(pd.DataFrame({
                        'train_frames': train_frame_,
                        'ens-std': std,
                        'model': model,
                        'mean': d.pixel_error.to_numpy(),
                        'n_points': n_points,
                    }, index=index))
    df_line = pd.concat(df_line)

    # ---------------------------------------------------
    # compute metrics as a function of ens std dev
    # ---------------------------------------------------
    df_w_vars_vids = compute_ensemble_var_for_each_metric(
        df_video_preds=df_video_preds,
        df_video_metrics=df_video_metrics,
        train_frames=train_frames_list,
        models=model_order,
        rng_seeds=rng_seeds,
        video_names=video_names,
        metric_names=metric_names,
    )

    # now compute metrics for all keypoints >= a given ensemble variance
    n_points_dict = {
        t: {m: np.nan * np.zeros_like(std_vals) for m in model_order} for t in train_frames_list}
    df_line_vids = []
    for train_frames_ in train_frames_list:
        df_w_vars_vidst = df_w_vars_vids[df_w_vars_vids.train_frames == train_frames_]
        for s, std in enumerate(std_vals):
            df_tmp1 = df_w_vars_vidst[df_w_vars_vidst['ens-std'] > std]
            for model in model_order:
                df_tmp2 = df_tmp1[df_tmp1.model == model]
                for metric_name in metric_names:
                    df_tmp3 = df_tmp2[df_tmp2.metric == metric_name]
                    for rng_seed_ in rng_seeds:
                        d = df_tmp3[df_tmp3.rng == rng_seed_]
                        n_points = np.sum(~d['value'].isna())
                        n_points_dict[train_frames_][model][s] = n_points
                        index = []
                        for row, k in zip(d.index, d['video_name'].to_numpy()):
                            index.append(
                                f'{row}_{k}_{model}_{s}_{train_frames_}_{rng_seed_}_{metric_name}')
                        df_line_vids.append(pd.DataFrame({
                            'train_frames': train_frames_,
                            'metric': metric_name,
                            'ens-std': std,
                            'model': model,
                            'value': np.nanmean(d.value.to_numpy()),
                            'n_points': n_points,
                        }, index=[index[0]]))

    df_line_vids = pd.concat(df_line_vids)

    # ---------------------------------------------------
    # plot figure
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(9, 10))
    gs = fig.add_gridspec(2, 1, top=0.93, height_ratios=[2, 2], hspace=0.2)
    gs0 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 0],
        height_ratios=[2.75, 1.25], hspace=0.3, width_ratios=[2, 1], wspace=0.3,
    )

    # ----------------------------------------------------------------
    # traces
    # ----------------------------------------------------------------
    gs00 = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs0[0, 0], height_ratios=[1, 1, 0.6, 0.6], hspace=0.1,
    )
    for c, coord in enumerate(['x', 'y', 'likelihood']):
        ax = fig.add_subplot(gs00[c])
        for model_type in models_to_compare:
            mask_trace = get_trace_mask(
                df_video_preds, video_name=vid_name,
                train_frames=train_frames, model_type=model_type, rng_seed=rng_seed,
            )
            ax.plot(
                np.arange(time_window[0], time_window[1]),
                df_video_preds[mask_trace].loc[:, (keypoint, coord)][slice(*time_window)],
                color=model_colors[model_type],
                label=model_type,
            )
        ylims = ax.get_ylim()
        clr = 0.75
        ax.fill_between(time_window_frames, ylims[0], ylims[1], color=[clr, clr, clr], zorder=0)
        if coord == 'x':
            ax.set_ylabel('x-coord', fontsize=labels_fontsize)
        elif coord == 'y':
            ax.set_ylabel('y-coord', fontsize=labels_fontsize)
        else:
            ax.set_ylabel('Confidence', fontsize=labels_fontsize)
        if c == 0:
            ax.set_title('Traces for %s' % keypoint, fontsize=labels_fontsize)
            ax.set_xticks([])
        else:
            ax.set_xticks([])
        ax.set_xlim([time_window[0] - 10, time_window[1] + 10])
        cleanaxis(ax)

    # plot multiview pca metric
    if 'pca_multiview_error' in df_video_metrics.metric.unique():
        metric_name_ = 'pca_multiview_error'
        metric_name_label_ = 'Multiview\nPCA'
    else:
        metric_name_ = 'pca_singleview_error'
        metric_name_label_ = 'Singleview\nPCA'
    c += 1
    ax = fig.add_subplot(gs00[c])
    for model_type in models_to_compare:
        mask_metric = get_trace_mask(
            df_video_metrics, video_name=vid_name, metric_name=metric_name_,
            train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
        ax.plot(
            np.arange(time_window[0], time_window[1]),
            df_video_metrics[mask_metric].loc[:, keypoint][slice(*time_window)],
            color=model_colors[model_type], label=model_type)
    ylims = ax.get_ylim()
    ax.fill_between(time_window_frames, ylims[0], ylims[1], color=[clr, clr, clr], zorder=0)
    ax.set_ylabel(metric_name_label_, fontsize=labels_fontsize)
    ax.legend(loc='lower left', framealpha=0.5)
    ax.set_xlabel('Frame number', fontsize=labels_fontsize)
    ax.set_xlim([time_window[0] - 10, time_window[1] + 10])
    cleanaxis(ax)

    # ----------------------------------------------------------------
    # frames examples
    # ----------------------------------------------------------------
    n_frames = time_window_frames[1] - time_window_frames[0]
    gs10 = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs0[1, 0], wspace=0.05)
    for i, idx_time in enumerate(np.arange(time_window_frames[0], time_window_frames[1])):
        ax = fig.add_subplot(gs10[i])
        frame = get_frames_from_idxs(cap, [idx_time + frames_offset])
        # plot frame
        if dataset_name == 'crim13':
            # zoom in on the action
            ystart = 230
            xstart = 240
            yend = 480
            xend = 640
            ax.imshow(frame[0, 0, ystart:yend, xstart:xend], cmap='gray', vmin=0, vmax=255)
        else:
            ystart = 0
            xstart = 0
            ax.imshow(frame[0, 0], cmap='gray', vmin=0, vmax=255)
        # plot predictions from 'good' model
        mask_1 = get_trace_mask(
            df_video_preds, video_name=vid_name,
            train_frames=train_frames, model_type=models_to_compare[1], rng_seed=rng_seed)
        for kp in keypoint_names:
            if kp == keypoint:
                markersize = 15
            else:
                continue
            tmp = df_video_preds[mask_1].iloc[idx_time][kp].to_numpy()
            ax.plot(
                tmp[0] - xstart, tmp[1] - ystart,
                '.', markersize=markersize, color=model_colors[models_to_compare[1]],
                markeredgecolor='w')
        # plot predictions from 'bad' model
        mask_0 = get_trace_mask(
            df_video_preds, video_name=vid_name,
            train_frames=train_frames, model_type=models_to_compare[0], rng_seed=rng_seed)
        tmp = df_video_preds[mask_0].iloc[idx_time][keypoint].to_numpy()
        ax.plot(
            tmp[0] - xstart, tmp[1] - ystart,
            'X', markersize=6, color=model_colors[models_to_compare[0]],
            markeredgecolor='w')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Frame %i' % idx_time, fontsize=labels_fontsize)

    # ----------------------------------------------------------------
    # pixel error vs ensemble variance
    # ----------------------------------------------------------------
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs0[:, 1], height_ratios=[1, 1], hspace=0.4)
    axes = [fig.add_subplot(gs01[0]), fig.add_subplot(gs01[1])]

    ymin = 100
    ymax = 0

    # plot errors
    for ax, tr, tr_actual in zip(axes, train_frames_list, train_frames_list_actual):
        g = sns.lineplot(
            x='ens-std',
            y='mean',
            hue='model',
            hue_order=model_order, palette=[model_colors[m] for m in model_order],
            data=df_line[df_line.train_frames == tr],
            ax=ax,
            errorbar='se',
            legend=model_order if tr == '75' else None,
        )
        ax.set_title(f'{tr_actual} train frames', fontsize=labels_fontsize)
        g.set(yscale='log')
        ax.set_ylabel('Pixel error', fontsize=labels_fontsize)
        g.set(yticks=yticks, yticklabels=yticks)
        if tr == '75':
            ax.set_xlabel(None)
        else:
            ax.set_xlabel('Ensemble std dev', fontsize=labels_fontsize)
        ymin = min(ymin, ax.get_ylim()[0])
        ymax = min(ymax, ax.get_ylim()[1])
        cleanaxis(ax)

    # plot annotations
    for ax, tr in zip(axes, train_frames_list):
        if tr == '1' or tr == '800':
            percentiles = [100, 50, 5]
        else:
            percentiles = [100, 50, 20]
        vals, prctiles = compute_percentiles(
            arr=n_points_dict[tr]['dlc'],
            std_vals=std_vals,
            percentiles=percentiles,
        )
        ax.set_ylim([ymin, ymax])
        for p, v in zip(prctiles, vals):
            ax.axvline(v, ymax=0.95, linestyle='--', linewidth=0.5, color='k', zorder=-1)
            ax.text(
                v / np.diff(ax.get_xlim()), 0.95, str(round(p)) + '%',
                transform=ax.transAxes,
                ha='left',
            )

    # ----------------------------------------------------------------
    # video metrics vs ensemble variance
    # ----------------------------------------------------------------
    gs1 = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[1, 0], height_ratios=[0.01, 1, 1], hspace=0.5)

    ax = fig.add_subplot(gs1[0])
    total_frames = int(n_points_dict['75']['dlc'][0] * len(rng_seeds))
    ax.set_title(
        f'Unlabeled data metrics ({total_frames} total frames)',
        fontsize=labels_fontsize + 2,
    )
    ax.set_frame_on(False)
    ax.axis('off')

    plots = {'temporal_norm': 'Temporal difference loss (pix)'}
    if 'pca_multiview_error' in df_line_vids.metric.unique():
        plots['pca_multiview_error'] = 'Multi-view PCA loss (pix)'
    plots['pca_singleview_error'] = 'Pose PCA loss (pix)'

    gs11 = gridspec.GridSpecFromSubplotSpec(
        1, len(plots), subplot_spec=gs1[1], wspace=0.3, hspace=1)
    gs12 = gridspec.GridSpecFromSubplotSpec(
        1, len(plots), subplot_spec=gs1[2], wspace=0.3, hspace=1)
    for i, (metric_name, ax_title) in enumerate(plots.items()):
        axs = []
        xlims = []
        ylims = []
        for j, train_frame in enumerate(train_frames_list):
            if j == 0:
                ax = fig.add_subplot(gs11[0, i])
            else:
                ax = fig.add_subplot(gs12[0, i])
            axs.append(ax)

            # plot lines
            g = sns.lineplot(
                x='ens-std',
                y='value',
                hue='model',
                hue_order=model_order, palette=[model_colors[m] for m in model_order],
                data=df_line_vids[
                    (df_line_vids.train_frames == train_frame)
                    & (df_line_vids.metric == metric_name)
                    ],
                ax=ax,
                errorbar='se',
                legend=None,
            )
            g.set(yscale='log')

            # plot percentiles
            if train_frame == '1' or train_frame == '800':
                percentiles = [100, 50, 5]
            else:
                percentiles = [100, 50, 20]
            vals, prctiles = compute_percentiles(
                arr=n_points_dict[train_frame]['dlc'],
                std_vals=std_vals,
                percentiles=percentiles,
            )
            for p, v in zip(prctiles, vals):
                ax.axvline(v, ymax=0.95, linestyle='--', linewidth=0.5, color='k', zorder=-1)
                ax.text(
                    v / np.diff(ax.get_xlim()), 0.95, str(round(p)) + '%',
                    transform=ax.transAxes,
                    ha='left',
                )

            # -----------------------
            # formatting
            # -----------------------
            if train_frame == '75':
                ax_text = '75 train frames'
            else:
                ax_text = f'{train_frames_list_actual[-1]} train frames'
            if j == 1:
                ax.set_xlabel('Ensemble std dev', fontsize=labels_fontsize)
            else:
                ax.set_xticks([])
                ax.set_xlabel(None)
                ax.set_title('%s' % ax_title, fontsize=labels_fontsize)
            if i == 0:
                ax.set_ylabel('Loss value', fontsize=labels_fontsize)
                ax.text(
                    -0.2, 1.2 if j == 1 else 1.25, ax_text, fontweight='semibold',
                    fontsize=labels_fontsize, transform=ax.transAxes)
            else:
                ax.set_ylabel(None)
            cleanaxis(ax)
            xlims += ax.get_xlim()
            ylims += ax.get_ylim()

        # set axes to be the same
        mn_x = np.min(xlims)
        mx_x = np.max(xlims)
        mn_y = np.min(ylims)
        mx_y = np.max(ylims)
        for ax in axs:
            ax.set_xlim(mn_x, mx_x)
            ax.set_ylim(mn_y, mx_y)

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    plt.suptitle(f'{dataset_name} dataset', fontsize=labels_fontsize + 2)
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'fig4_{dataset_name}.{format}'), dpi=300)
    plt.close()
