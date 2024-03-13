import cv2
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
import numpy as np
import os
import pandas as pd
from sklearn import metrics
import seaborn as sns

from lightning_pose_plots import (
    colors_tab10,
    dataset_info_fig3_examples,
    dataset_info_fig3_metrics,
)
from lightning_pose_plots.utilities import (
    cleanaxis,
    get_frames_from_idxs,
    get_trace_mask,
    load_results_dataframes,
)


def compute_outlier_metrics(
    df_video_preds,
    df_video_metrics,
    dataset_name,
    views_list,
    bodyparts_list,
    max_frames,
    views_list_x=None,
    views_list_y=None,
    metric_thresh=20,
    confidence_thresh=0.9,
    model_type='dlc',
):

    metric_names = df_video_metrics.metric.unique()
    video_names = df_video_metrics[df_video_metrics.model_type == model_type].video_name.unique()
    rng_seeds = df_video_metrics.rng_seed_data_pt.unique()

    df_errors = []
    frames = {m: {tr: np.array([]) for tr in ['75', max_frames]} for m in metric_names}
    n_total_kps = {m: {tr: 0 for tr in ['75', max_frames]} for m in metric_names}
    for rng_seed_ in rng_seeds:
        df_metrics_0 = df_video_metrics[df_video_metrics.rng_seed_data_pt == rng_seed_]
        df_preds_0 = df_video_preds[df_video_preds.rng_seed_data_pt == rng_seed_]
        for train_frame_ in ['75', max_frames]:
            df_metrics_1 = df_metrics_0[df_metrics_0.train_frames == train_frame_]
            df_preds_1 = df_preds_0[df_preds_0.train_frames == train_frame_]
            for model_type_ in ['dlc']:
                df_metrics_2 = df_metrics_1[df_metrics_1.model_type == model_type_]
                df_preds_2 = df_preds_1[df_preds_1.model_type == model_type_]
                for metric_name_ in metric_names:
                    for bp in bodyparts_list:
                        true_vals = []
                        pred_vals = []
                        weights = []
                        for video_name_ in video_names:
                            df_metrics_3 = df_metrics_2[df_metrics_2.video_name == video_name_]
                            df_preds_3 = df_preds_2[df_preds_2.video_name == video_name_]
                            # compute predicted errors
                            if dataset_name == 'crim13':
                                df_test = df_metrics_3[df_metrics_3.metric == metric_name_].loc[
                                          :, bp]
                            else:
                                df_test = df_metrics_3[df_metrics_3.metric == metric_name_].loc[
                                          :, [f'{bp}_{view}' for view in views_list]]
                            mat_test = df_test.to_numpy()
                            if len(mat_test.shape) == 1:
                                y_score = np.copy(mat_test)
                            else:
                                if metric_name_ == 'confidence':
                                    y_score = mat_test.min(axis=1)
                                else:
                                    y_score = mat_test.max(axis=1)
                            # collect unique names for each keypoint-frame
                            idxs_all = np.array(
                                [f'{video_name_}_{bp}_{rng_seed_}_{i}' for i in df_preds_3.index]
                            )
                            if metric_name_ == 'confidence':
                                idxs_selected = idxs_all[y_score < confidence_thresh]
                            else:
                                idxs_selected = idxs_all[y_score > metric_thresh]
                            frames[metric_name_][train_frame_] = np.concatenate([
                                frames[metric_name_][train_frame_], idxs_selected])
                            n_total_kps[metric_name_][train_frame_] += len(idxs_all)
                            # compute "true" errors
                            if dataset_name == 'mirror-mouse':
                                bp_xs = df_preds_3.loc[
                                        :, [f'{bp}_{view}' for view in views_list]
                                        ].to_numpy()[:, ::3]
                                diffs = np.abs(np.diff(bp_xs, axis=1))
                                y_true = (diffs > metric_thresh)[:, 0]
                            elif dataset_name == 'mirror-fish':
                                bp_xs = df_preds_3.loc[
                                        :, [f'{bp}_{view}' for view in views_list_x]
                                        ].to_numpy()[:, ::3]
                                diffs_x = np.abs(np.diff(bp_xs, axis=1))
                                bp_ys = df_preds_3.loc[
                                        :, [f'{bp}_{view}' for view in views_list_y]
                                        ].to_numpy()[:, 1::3]
                                diffs_y = np.abs(np.diff(bp_ys, axis=1))
                                y_true = (diffs_x > metric_thresh)[:, 0] | (diffs_y > metric_thresh)[:, 0]
                            else:
                                y_true = np.array([0])
                            if np.sum(y_true) == 0:
                                # print('no outliers!')
                                continue
                            else:
                                n_true = np.sum(y_true)
                                weight_pos = 1.0 / n_true
                                weight_neg = 1.0 / (y_true.shape[0] - n_true)
                                sample_weight = np.zeros(y_true.shape[0])
                                sample_weight[y_true] = weight_pos
                                sample_weight[~y_true] = weight_neg
                            # collect
                            true_vals.append(y_true)
                            pred_vals.append(y_score)
                            weights.append(sample_weight)
                        if len(true_vals) == 0:
                            continue
                        # compute metrics over all videos
                        y_true = np.concatenate(true_vals)
                        y_score = np.concatenate(pred_vals)
                        sample_weights = np.concatenate(weights)
                        if metric_name_ == 'confidence':
                            pos_label = 0
                        else:
                            pos_label = 1
                        mask = ~np.isnan(y_score)
                        # calculate auc
                        y_true_m = y_true[mask]
                        y_score_m = y_score[mask]
                        if len(y_true_m) == 0:
                            auc_val = np.nan
                            ap = np.nan
                            auprc = np.nan
                        else:
                            fpr, tpr, thresholds = metrics.roc_curve(
                                y_true[mask], y_score[mask],
                                pos_label=pos_label,
                                sample_weight=sample_weights[mask],
                            )
                            auc_val = metrics.auc(fpr, tpr)
                            # calculate AP score
                            ap = metrics.average_precision_score(
                                y_true[mask], y_score[mask], sample_weight=sample_weights[mask],
                                pos_label=pos_label,
                            )
                            p, r, thresholds = metrics.precision_recall_curve(
                                y_true[mask], y_score[mask], sample_weight=sample_weights[mask],
                                pos_label=pos_label,
                            )
                            auprc = metrics.auc(r, p)
                        df_errors.append(pd.DataFrame({
                            'rng_seed': rng_seed_,
                            'train_frames': train_frame_,
                            'model_type': model_type_,
                            'bodypart': bp,
                            'metric': metric_name_,
                            'auc': auc_val,
                            'ap': ap,
                            'auprc': auprc,
                            'n_errors': np.sum(y_true),
                            'n_frames': y_true.shape[0],
                        }, index=[0]))

    if len(df_errors) > 0:
        df_errors = pd.concat(df_errors)

    return df_errors, frames, n_total_kps


def plot_figure3_example_frame_sequence(data_dir, dataset_name, version=0, format='pdf'):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    rng_seed = '0'
    model_type = 'dlc'

    vid_name_ = dataset_info_fig3_examples[f'{dataset_name}-{version}']['vid_name_tr']
    vid_name_load = dataset_info_fig3_examples[f'{dataset_name}-{version}']['vid_name_load']
    frames_offset = dataset_info_fig3_examples[f'{dataset_name}-{version}']['frames_offset']
    keypoint_ = dataset_info_fig3_examples[f'{dataset_name}-{version}']['keypoint_tr']
    time_window_beg = dataset_info_fig3_examples[f'{dataset_name}-{version}']['time_window_beg']
    n_frames = dataset_info_fig3_examples[f'{dataset_name}-{version}']['n_frames']
    train_frames = dataset_info_fig3_examples[f'{dataset_name}-{version}']['train_frames']

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    # load model predictions on unlabeled videos
    _, _, df_video_preds, df_video_metrics = load_results_dataframes(
        results_df_dir=os.path.join(data_dir, 'results_dataframes'),
        dataset_name=dataset_name,
    )

    vid_file = os.path.join(data_dir, dataset_name, 'videos_OOD', '%s.mp4' % vid_name_load)
    cap = cv2.VideoCapture(vid_file)

    mask_preds = get_trace_mask(
        df_video_preds, video_name=vid_name_,
        train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
    mask_metrics = get_trace_mask(
        df_video_metrics, video_name=vid_name_,
        train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)

    # ---------------------------------------------------
    # plot
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(12, 6))
    gs = fig.add_gridspec(1, n_frames, wspace=0)

    for i, idx_time in enumerate(np.arange(time_window_beg, time_window_beg + n_frames)):
        ax = fig.add_subplot(gs[i])
        ax.axis('off')
        frame = get_frames_from_idxs(cap, [idx_time + frames_offset])
        # plot frame
        ax.imshow(frame[0, 0], cmap='gray', vmin=0, vmax=255)
        # plot marker
        tmp_preds = df_video_preds[mask_preds].iloc[idx_time][keypoint_].to_numpy()
        ax.plot(tmp_preds[0], tmp_preds[1], 'x', markersize=8, markeredgewidth=3, color='m')
        # put metric info in title
        df_ = df_video_metrics[mask_metrics]
        c = df_[df_.metric == 'confidence'].loc[frames_offset + idx_time][keypoint_]
        t = df_[df_.metric == 'temporal_norm'].loc[frames_offset + idx_time][keypoint_]
        if dataset_name == 'crim13':
            p = df_[df_.metric == 'pca_singleview_error'].loc[frames_offset + idx_time][keypoint_]
            title_str = 'Confidence: %1.2f\nTemporal diff: %1.2f\nPose PCA: %1.2f' % (c, t, p)
        else:
            p = df_[df_.metric == 'pca_multiview_error'].loc[frames_offset + idx_time][keypoint_]
            if dataset_name == 'mirror-mouse':
                x0 = df_video_preds[mask_preds].iloc[idx_time][keypoint_]['x']
                x1 = df_video_preds[mask_preds].iloc[idx_time][keypoint_.replace('bot', 'top')]['x']
                d = np.abs(x0 - x1)
                title_str = 'Top/bot horizontal\ndisplacement: %1.2f\nConfidence: %1.2f\nTemporal diff: %1.2f\nMulti-view PCA: %1.2f' % (d, c, t, p)
            else:
                title_str = 'Confidence: %1.2f\nTemporal norm: %1.2f\nMultiview PCA: %1.2f' % (c, t, p)
        plt.title(title_str, fontsize=10, ha='left', loc='left')
        ax.text(0.05, 0.05, f'Frame {idx_time}', transform=ax.transAxes, color='w')
        ax.set_xticklabels('')
        ax.set_yticklabels('')

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig3a_{dataset_name}_frame_sequence_v{version}.{format}'),
        dpi=300,
    )
    plt.close()


def plot_figure3_example_traces(data_dir, dataset_name, version=0, format='pdf'):

    cmap_red = cm.get_cmap('tab10', 2)
    cmap_red.colors[0, :] = [1, 1, 1, 0]
    cmap_red.colors[1, :] = [*colors_tab10[3], 0.5]

    cmap_blue = cm.get_cmap('tab10', 2)
    cmap_blue.colors[0, :] = [1, 1, 1, 0]
    cmap_blue.colors[1, :] = [*colors_tab10[0], 0.5]

    labels_fontsize = 10

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    rng_seed = '0'
    model_type = 'dlc'

    # get dataset-specific vid/keypoint info
    vid_name_tr = dataset_info_fig3_examples[f'{dataset_name}-{version}']['vid_name_tr']
    keypoint_tr = dataset_info_fig3_examples[f'{dataset_name}-{version}']['keypoint_tr']
    time_window_tr = dataset_info_fig3_examples[f'{dataset_name}-{version}']['time_window_tr']
    train_frames = dataset_info_fig3_examples[f'{dataset_name}-{version}']['train_frames']
    thresh_ = dataset_info_fig3_examples[f'{dataset_name}-{version}']['metric_thresh']

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    # load model predictions on unlabeled videos
    _, _, df_video_preds, df_video_metrics = load_results_dataframes(
        results_df_dir=os.path.join(data_dir, 'results_dataframes'),
        dataset_name=dataset_name,
    )

    # ----------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(5, 6))
    gs00 = fig.add_gridspec(
        5, 1, top=0.93, height_ratios=[0.75, 0.75, 0.5, 0.5, 0.5], hspace=0.1)

    flags_conf = None
    flags_metric = None

    mask_trace = get_trace_mask(
        df_video_preds, video_name=vid_name_tr,
        train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
    df_traces = df_video_preds[mask_trace]
    axes = [None for _ in range(6)]

    metrics_dict = {
        'x': 'x-coord', 'y': 'y-coord', 'likelihood': 'Confidence',
        'temporal_norm': 'Temporal\ndifference\nloss (pix)',
    }
    if dataset_name == 'crim13':
        metrics_dict['pca_singleview_error'] = 'Pose PCA\nloss (pix)'
    else:
        metrics_dict['pca_multiview_error'] = 'Multiview\nPCA\nloss (pix)'

    for c, (metric, ax_title) in enumerate(metrics_dict.items()):
        axes[c] = fig.add_subplot(gs00[c])
        if metric in ['x', 'y', 'likelihood']:
            # plot traces
            axes[c].plot(
                np.arange(time_window_tr[0], time_window_tr[1]),
                df_traces.loc[:, (keypoint_tr, metric)][slice(*time_window_tr)],
                color='k',
            )
            # plot flagged frames, ONLY for likelihood; combinations for x/y plotted below
            if metric == 'likelihood':
                thresh = 0.9
                flags_conf = (df_traces.loc[:, (keypoint_tr, metric)][slice(*time_window_tr)].to_numpy() < thresh).astype('int')
                ylims = axes[c].get_ylim()
                axes[c].imshow(
                    flags_conf[None, :],
                    extent=(time_window_tr[0] - 0.5, time_window_tr[1] - 0.5, ylims[0], ylims[1]),
                    aspect='auto',
                    origin='lower', cmap=cmap_blue, alpha=1.0, zorder=0, interpolation='nearest',
                )
                axes[c].plot(
                    [time_window_tr[0], time_window_tr[1]], [thresh, thresh], '--k', linewidth=0.5,
                )
        else:
            mask_metric = get_trace_mask(
                df_video_metrics, video_name=vid_name_tr, metric_name=metric,
                train_frames=train_frames, model_type=model_type,
                rng_seed=rng_seed,
            )
            df_metric = df_video_metrics[mask_metric]
            # plot traces
            axes[c].plot(
                np.arange(time_window_tr[0], time_window_tr[1]),
                df_metric.loc[:, keypoint_tr][slice(*time_window_tr)],
                color='k')
            # plot flagged frames
            thresh = thresh_
            flags_metric_curr = (df_metric.loc[:, keypoint_tr][slice(*time_window_tr)].to_numpy() > thresh).astype('int')
            ylims = axes[c].get_ylim()
            axes[c].imshow(
                flags_metric_curr[None, :],
                extent=(time_window_tr[0] - 0.5, time_window_tr[1] - 0.5, ylims[0], ylims[1]),
                aspect='auto',
                origin='lower', cmap=cmap_blue if metric == 'temporal_norm' else cmap_red,
                alpha=1.0, zorder=0, interpolation='nearest',
            )
            axes[c].plot(
                [time_window_tr[0], time_window_tr[1]], [thresh, thresh], '--k', linewidth=0.5,
            )
            if metric == 'temporal_norm':
                flags_conf |= flags_metric_curr
            else:
                if flags_metric is None:
                    flags_metric = flags_metric_curr
                else:
                    flags_metric |= flags_metric_curr

        axes[c].set_ylabel(ax_title, fontsize=labels_fontsize)
        if c == 0:
            axes[c].set_title(f'Traces for {keypoint_tr}', fontsize=labels_fontsize + 2)
        if c < len(metrics_dict) - 1:
            axes[c].set_xticks([])
        else:
            axes[c].set_xlabel('Frame number', fontsize=labels_fontsize)
        axes[c].set_xlim([time_window_tr[0] - 10, time_window_tr[1] + 10])
        cleanaxis(axes[c])

    # plot backgrounds
    for c, metric in enumerate(['x', 'y']):
        ylims = axes[c].get_ylim()
        axes[c].imshow(
            flags_conf[None, :],
            extent=(time_window_tr[0] - 0.5, time_window_tr[1] - 0.5, ylims[0], ylims[1]),
            aspect='auto',
            origin='lower', cmap=cmap_blue, zorder=0, interpolation='nearest',
        )
        axes[c].imshow(
            flags_metric[None, :],
            extent=(time_window_tr[0] - 0.5, time_window_tr[1] - 0.5, ylims[0], ylims[1]),
            aspect='auto',
            origin='lower', cmap=cmap_red, zorder=0, interpolation='nearest',
        )

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig3b_{dataset_name}_example_traces_v{version}.{format}'),
        dpi=300,
    )
    plt.close()


def plot_figure3_venn_diagrams(data_dir, dataset_name, format='pdf'):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    cols_to_keep = dataset_info_fig3_metrics[dataset_name]['cols_to_keep']
    cols_to_drop = dataset_info_fig3_metrics[dataset_name]['cols_to_drop']
    bodyparts_list = dataset_info_fig3_metrics[dataset_name]['bodyparts_list']
    views_list = dataset_info_fig3_metrics[dataset_name]['views_list']
    views_list_x = dataset_info_fig3_metrics[dataset_name]['views_list_x']
    views_list_y = dataset_info_fig3_metrics[dataset_name]['views_list_y']
    metric_thresh = dataset_info_fig3_metrics[dataset_name]['metric_thresh']
    max_frames = dataset_info_fig3_metrics[dataset_name]['max_frames']
    total_frames = dataset_info_fig3_metrics[dataset_name]['total_frames']

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    # load model predictions on both labeled frames and unlabeled videos
    _, _, df_video_preds, df_video_metrics = load_results_dataframes(
        results_df_dir=os.path.join(data_dir, 'results_dataframes'),
        dataset_name=dataset_name,
    )

    # drop body parts
    df_video_preds = df_video_preds.drop(columns=cols_to_drop)
    df_video_metrics = df_video_metrics.drop(columns=cols_to_drop)
    # recompute means
    if len(cols_to_keep) > 0:
        df_video_metrics.loc[:, 'mean'] = df_video_metrics.loc[:, cols_to_keep].mean(axis=1)

    # compute error metrics
    df_errors, frames, n_total_kps = compute_outlier_metrics(
        df_video_preds=df_video_preds,
        df_video_metrics=df_video_metrics,
        dataset_name=dataset_name,
        views_list=views_list,
        bodyparts_list=bodyparts_list,
        max_frames=max_frames,
        views_list_x=views_list_x,
        views_list_y=views_list_y,
        metric_thresh=metric_thresh,
        confidence_thresh=0.9,
        model_type='dlc',
    )

    # ---------------------------------------------------
    # plot
    # ---------------------------------------------------
    metric_names = df_video_metrics.metric.unique()

    fig = plt.figure(constrained_layout=True, figsize=(12, 3.5))
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[1, 10])

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 0], wspace=0.1)

    for t, tr in enumerate(['75', max_frames]):

        # title
        a = np.concatenate([
            frames['confidence'][tr],
            frames['temporal_norm'][tr],
            frames['pca_multiview_error'][tr] if 'pca_multiview_error' in metric_names else np.array([]),
            frames['pca_singleview_error'][tr],
        ])
        outliers = int(round(len(np.unique(a)) / 1000))
        total = int(round(n_total_kps["confidence"][tr] / 1000))
        ax = fig.add_subplot(gs0[0, t])
        ax.set_axis_off()
        tr_ = '75' if tr == '75' else total_frames
        ax.text(
            0.5, 0.5,
            f'{tr_} train frames\nOutliers: {outliers}k / {total}k keypoints',
            ha='center',
            fontsize=12,
        )

        # pca
        ax = fig.add_subplot(gs1[0, 2 * t])
        if 'pca_multiview_error' in metric_names:
            data = [
                set(frames['pca_singleview_error'][tr]),
                set(frames['pca_multiview_error'][tr]),
            ]
            labels = (
                'Pose PCA',
                'Multi-view PCA',
            )
            v = venn2(data, set_labels=labels, ax=ax)
            c = venn2_circles(data, ax=ax)
            c[0].set_lw(0.5)
            c[1].set_lw(0.5)
        else:
            ax.set_axis_off()

        # all 3
        ax = fig.add_subplot(gs1[0, 2 * t + 1])
        data = [
            set(frames['confidence'][tr]),
            set(frames['temporal_norm'][tr]),
            set(frames['pca_multiview_error'][tr]) if 'pca_multiview_error' in metric_names
            else set(frames['pca_singleview_error'][tr]),
        ]
        labels = (
            'Conf',
            'Temporal',
            'Multi-view PCA' if 'pca_multiview_error' in metric_names else 'Pose PCA',
        )
        v = venn3(data, set_labels=labels, ax=ax)
        c = venn3_circles(data, ax=ax)
        c[0].set_lw(0.5)
        c[1].set_lw(0.5)
        c[2].set_lw(0.5)

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig3c_{dataset_name}_venn_diagrams.{format}'),
        dpi=300,
    )
    plt.close()


def plot_figure3_outlier_detector_performance(data_dir, dataset_name, format='pdf'):

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    cols_to_keep = dataset_info_fig3_metrics[dataset_name]['cols_to_keep']
    cols_to_drop = dataset_info_fig3_metrics[dataset_name]['cols_to_drop']
    bodyparts_list = dataset_info_fig3_metrics[dataset_name]['bodyparts_list']
    views_list = dataset_info_fig3_metrics[dataset_name]['views_list']
    views_list_x = dataset_info_fig3_metrics[dataset_name]['views_list_x']
    views_list_y = dataset_info_fig3_metrics[dataset_name]['views_list_y']
    metric_thresh = dataset_info_fig3_metrics[dataset_name]['metric_thresh']
    max_frames = dataset_info_fig3_metrics[dataset_name]['max_frames']

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
    # load model predictions on both labeled frames and unlabeled videos
    _, _, df_video_preds, df_video_metrics = load_results_dataframes(
        results_df_dir=os.path.join(data_dir, 'results_dataframes'),
        dataset_name=dataset_name,
    )

    # drop body parts
    df_video_preds = df_video_preds.drop(columns=cols_to_drop)
    df_video_metrics = df_video_metrics.drop(columns=cols_to_drop)
    # recompute means
    if len(cols_to_keep) > 0:
        df_video_metrics.loc[:, 'mean'] = df_video_metrics.loc[:, cols_to_keep].mean(axis=1)

    # compute error metrics
    df_errors, _, _ = compute_outlier_metrics(
        df_video_preds=df_video_preds,
        df_video_metrics=df_video_metrics,
        dataset_name=dataset_name,
        views_list=views_list,
        bodyparts_list=bodyparts_list,
        max_frames=max_frames,
        views_list_x=views_list_x,
        views_list_y=views_list_y,
        metric_thresh=metric_thresh,
        confidence_thresh=0.9,
        model_type='dlc',
    )

    # ---------------------------------------------------
    # plot
    # ---------------------------------------------------
    n_bodyparts = len(bodyparts_list)
    fig = plt.figure(constrained_layout=True, figsize=(4, 2 * n_bodyparts))
    gs = fig.add_gridspec(n_bodyparts, 2, hspace=0.15)
    order = [
        'confidence',
        'temporal_norm',
        'pca_singleview_error',
        'pca_multiview_error'
    ]
    for i, tf in enumerate(['75', max_frames]):
        for a, bp in enumerate(bodyparts_list):
            ax = fig.add_subplot(gs[a, i])
            mask = ((df_errors.bodypart == bp)
                    & (df_errors.train_frames == tf)
                    )
            sns.boxplot(
                data=df_errors[mask], x='metric', y='auc',
                order=order, ax=ax,
                medianprops=dict(color='k', alpha=1.0, linewidth=1),
                boxprops=dict(facecolor='none', edgecolor='k'),
                linewidth=1, width=0.4,
            )
            if a + 1 == len(order):
                xticklabels = [
                    'Conf',
                    'Temporal diff',
                    'Pose PCA',
                    'Multi-view PCA',
                ]
                xlabel = 'Metric'
            else:
                xticklabels = []
                xlabel = ''
            ax.set_xticklabels(xticklabels, rotation=45, ha='right')
            ax.set_xlabel(xlabel)
            if i == 0:
                ax.set_ylabel(f'{bp}\nAUROC')
            else:
                ax.set_ylabel('')
            if a == 0:
                ax.set_title(f'{tf} train frames', fontsize=10)
            cleanaxis(ax)

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig3d_{dataset_name}_outlier_detector_performance.{format}'),
        dpi=300,
    )
    plt.close()


def plot_figure3(data_dir, dataset_name, format='pdf'):

    # crim13 has two versions of frames and traces to make up for lack of AUROC plot
    versions = [0, 1] if dataset_name == 'crim13' else [0]
    for version in versions:

        # plot sequences of frames
        plot_figure3_example_frame_sequence(
            data_dir=data_dir, dataset_name=dataset_name, version=version, format=format)

        # plot traces of predictions and error metrics with colored backgrounds
        plot_figure3_example_traces(
            data_dir=data_dir, dataset_name=dataset_name, version=version, format=format)

    # plot venn diagrams of outlier detector overlaps
    plot_figure3_venn_diagrams(data_dir=data_dir, dataset_name=dataset_name, format=format)

    # plot metric performance as outlier detector, but only for multiview datasets
    if dataset_name in ['mirror-mouse', 'mirror-fish']:
        plot_figure3_outlier_detector_performance(
            data_dir=data_dir, dataset_name=dataset_name, format=format)
