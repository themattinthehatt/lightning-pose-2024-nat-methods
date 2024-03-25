import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

from lightning_pose_plots import dataset_info_fig5
from lightning_pose_plots.utilities import compute_ensemble_variance, compute_percentiles


def compute_ensemble_var_for_each_pixel_error(
    df_ground_truth,
    df_labeled_preds,
    df_labeled_metrics,
    post_processors_dict,
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
    post_processors_dict : dict
        'processor': list of seeds
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
        # compute ensemble variance on raw outputs
        net_vars = compute_ensemble_variance(
            df_ground_truth=df_ground_truth,
            df_preds=df_labeled_preds,
            train_frames=train_frame_,
            models=models,
            rng_seeds=rng_seeds,
            split_set=split_set,
            distribution=distribution,
        )

        dfs_all = {}
        for model, rng_seeds_ in post_processors_dict.items():
            for rng_seed in rng_seeds_:
                mask = ((df_labeled_metrics.set == split_set)
                        & (df_labeled_metrics.distribution == 'OOD')
                        & (df_labeled_metrics.rng_seed_ensembling == model)
                        & (df_labeled_metrics.rng_seed_data_pt == rng_seed)
                        & (df_labeled_metrics.train_frames == train_frame_)
                        & (df_labeled_metrics.metric == 'pixel_error')
                        )
                idx = f'{model}-{rng_seed}'
                dfs_all[idx] = df_labeled_metrics[mask]
                dfs_all[idx].sort_index(inplace=True)
                assert np.all(dfs_all[idx].index == df_ground_truth.index)

        for i, kp in enumerate(dfs_all['raw-0'].columns):
            if kp in ['model_path', 'rng_seed_data_pt', 'train_frames', 'model_type', 'metric',
                      'set', 'distribution', 'rng_seed_ensembling', 'mean']:
                continue
            for m in list(dfs_all.keys()):
                df_w_vars.append(pd.DataFrame({
                    'pixel_error': dfs_all[m][kp],
                    'ens-std': net_vars[:, i],
                    'keypoint': kp,
                    'model': m,
                    'train_frames': train_frame_,
                }, index=df_ground_truth.index))

    df_w_vars = pd.concat(df_w_vars)
    return df_w_vars


def compute_pixel_error_using_ensemble_std_dev_threshold(
    df,
    train_frames,
    std_vals,
):
    models = df.model.unique()
    n_points_dict = {
        t: {m: np.nan * np.zeros_like(std_vals) for m in models} for t in train_frames
    }
    df_line = []
    for train_frame_ in train_frames:
        dft = df[df.train_frames == train_frame_]
        for s, std in enumerate(std_vals):
            df_tmp_ = dft[dft['ens-std'] > std]
            for i in models:
                d = df_tmp_[df_tmp_.model == i]
                n_points = np.sum(~d['pixel_error'].isna())
                n_points_dict[train_frame_][i][s] = n_points
                index = []
                for row, k in zip(d.index, d['keypoint'].to_numpy()):
                    index.append(row + f'_{i}_{s}_{train_frames}_{k}')
                df_line.append(pd.DataFrame({
                    'train_frames': train_frame_,
                    'ens-std': std,
                    'model': '-'.join(i.split('-')[:-1]),
                    'rng_seed': i.split('-')[-1],
                    'mean': d.pixel_error.to_numpy(),
                    'n_points': n_points,
                }, index=index))
    df_line = pd.concat(df_line)
    return df_line, n_points_dict


def plot_pixel_error_vs_ensemble_std_dev(
    df_line,
    colors,
    axes,
    train_frames_list,
    train_frames_list_actual,
    n_points_dict,
    std_vals,
    percentiles,
):
    for idx, (ax, tr, tr_real) in enumerate(
            zip(axes, train_frames_list, train_frames_list_actual)
    ):
        g = sns.lineplot(
            x='ens-std',
            y='mean',
            hue='model',
            palette=colors,
            data=df_line[df_line.train_frames == tr],
            ax=ax,
            errorbar='se',
            legend=False,  # True if idx == 1 else False,
        )
        g.set(yscale='log')
        g.set(yticks=[4.5, 10, 20, 30, 40, 50])

        vals, prctiles = compute_percentiles(
            arr=n_points_dict[tr]['raw-0'],
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
        ax.set_title(f'{tr_real} train frames', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Pixel error', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Ensemble std dev', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def plot_traces(
    df_smoothers,
    df_raw,
    keypoint,
    coord,
    time_window,
    fig,
    subplot_spec,
    colors,
):
    n_rows = len(df_smoothers)

    gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 2, subplot_spec=subplot_spec,
        hspace=0.2, width_ratios=[1, 1], wspace=0.05,
    )
    for ax_idx, plot_diffs in enumerate([False, True]):
        cols0 = (f'{keypoint}_{coord}')
        # plot smoothers
        for idx, (smoother_name, df_) in enumerate(df_smoothers.items()):
            ax = fig.add_subplot(gs[idx, ax_idx])
            # plot individual models
            for i, (_, df) in enumerate(df_raw.items()):
                m = df.loc[:, cols0].to_numpy()
                if (smoother_name == 'arima' or smoother_name == 'median-filt') \
                        and (i != 4):
                    continue
                data = m[slice(*time_window)]
                data = np.diff(data) if plot_diffs else data
                ax.plot(data, color=[0.5, 0.5, 0.5], linewidth=0.5, alpha=0.5)
                if smoother_name.find('eks') > -1 \
                        and len(smoother_name.split('-')) == 3:
                    # we're using a specific subsample of ensemble members for eks
                    idx_member = int(smoother_name.split('-')[-1])
                    if idx_member == i + 1:
                        # exit loop here
                        break

            if idx == 0:
                ylims = ax.get_ylim()
            if plot_diffs:
                ylims = [-50, 50]
            # plot smoothed model
            model_name = df_.columns.get_level_values('scorer')[0]
            cols1 = (model_name, keypoint, coord)
            data = df_.loc[:, cols1].to_numpy()[slice(*time_window)]
            data = np.diff(data) if plot_diffs else data
            ax.plot(
                data,
                color=colors[idx],
                linewidth=1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(ylims)
            # fix axes
            ax.axis('off')
            if ax_idx == 0:
                ax.text(
                    0.0, 0.95, smoother_name, fontsize=8,
                    ha='left', va='center', rotation=0, transform=ax.transAxes,
                )
            if idx == 0:
                metric = 'velocity' if plot_diffs else 'position'
                ax.set_title(f'{cols0} {metric}', fontsize=8)


def plot_figure5_all_panels(
    save_file,
    df_ground_truth,
    df_labeled_preds,
    df_labeled_post,
    df_labeled_eks,
    trace_data,
    train_frames_list,
    train_frames_list_actual,
    std_vals,
    traces_keypoint,
    traces_coord,
    traces_time_window,
):

    # ---------------------------------------------------
    # compute metrics as a function of ensemble var
    # ---------------------------------------------------
    post_processors_post_dict = {
        'median-filt': ['0', '1', '2', '3', '4'],
        'arima': ['0', '1', '2', '3', '4'],
        'ens-mean': ['all'],
        'ens-median': ['all'],
        'eks-temporal': ['all'],
        'eks-pca': ['all'],
        'raw': ['0', '1', '2', '3', '4'],
    }
    df_w_vars_post = compute_ensemble_var_for_each_pixel_error(
        df_ground_truth=df_ground_truth,
        df_labeled_preds=df_labeled_preds,
        df_labeled_metrics=df_labeled_post,
        post_processors_dict=post_processors_post_dict,
        train_frames=train_frames_list,
        models=['dlc'],
        rng_seeds=['0', '1', '2', '3', '4'],
        split_set='test',
        distribution='OOD',
    )
    df_line_post, n_points_dict_post = compute_pixel_error_using_ensemble_std_dev_threshold(
        df=df_w_vars_post,
        train_frames=train_frames_list,
        std_vals=std_vals,
    )

    # plot eks w/ various ensemble numbers
    post_processors_eks_dict = {
        'eks-temporal-2': ['all'],
        'eks-temporal-3': ['all'],
        'eks-temporal-4': ['all'],
        'eks-temporal-5': ['all'],
        'eks-temporal-6': ['all'],
        'eks-temporal-8': ['all'],
        'raw': ['0', '1', '2', '3', '4'],
    }
    df_w_vars_eks = compute_ensemble_var_for_each_pixel_error(
        df_ground_truth=df_ground_truth,
        df_labeled_preds=df_labeled_preds,
        df_labeled_metrics=df_labeled_eks,
        post_processors_dict=post_processors_eks_dict,
        train_frames=train_frames_list,
        models=['dlc'],
        rng_seeds=['0', '1', '2', '3', '4'],
        split_set='test',
        distribution='OOD',
    )
    df_line_eks, n_points_dict_eks = compute_pixel_error_using_ensemble_std_dev_threshold(
        df=df_w_vars_eks,
        train_frames=train_frames_list,
        std_vals=std_vals,
    )

    # ---------------------------------------------------
    # plot figure
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], hspace=0.4, wspace=0.2)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.3)
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.3)

    # ---------------------------------------------------
    # pixel error - post-processors
    # ---------------------------------------------------
    base = sns.color_palette("hls", 7)
    colors = [base[0], base[2], base[3], base[4], base[5], base[6], [0.5, 0.5, 0.5]]
    axes = [
        fig.add_subplot(gs00[0]),
        fig.add_subplot(gs00[1]),
    ]
    plot_pixel_error_vs_ensemble_std_dev(
        df_line=df_line_post,
        colors=colors,
        axes=axes,
        train_frames_list=train_frames_list,
        train_frames_list_actual=train_frames_list_actual,
        n_points_dict=n_points_dict_post,
        std_vals=std_vals,
        percentiles=[100, 50, 5],
    )

    # ---------------------------------------------------
    # traces - post-processors
    # ---------------------------------------------------
    smoothers = [
        'median-filt',
        'arima',
        'ens-mean',
        'ens-median',
        'eks-temporal-5',
        'eks-pca',
    ]
    raw_models = [f'raw-{i}' for i in range(5)]
    plot_traces(
        df_smoothers={s: trace_data[s] for s in smoothers},
        df_raw={s: trace_data[s] for s in raw_models},
        keypoint=traces_keypoint,
        coord=traces_coord,
        time_window=traces_time_window,
        fig=fig,
        subplot_spec=gs[1, 0],
        colors=colors[:-1],
    )

    # ---------------------------------------------------
    # pixel error - ensemble members
    # ---------------------------------------------------
    base = sns.color_palette("hls", 7)[5]
    colors_ = sns.dark_palette(base, len(post_processors_eks_dict), reverse=True)
    colors = [
        colors_[0], colors_[1], colors_[2], colors_[3], colors_[4], colors_[5], [0.5, 0.5, 0.5]
    ]
    axes = [
        fig.add_subplot(gs01[0]),
        fig.add_subplot(gs01[1]),
    ]
    plot_pixel_error_vs_ensemble_std_dev(
        df_line=df_line_eks,
        colors=colors,
        axes=axes,
        train_frames_list=train_frames_list,
        train_frames_list_actual=train_frames_list_actual,
        n_points_dict=n_points_dict_eks,
        std_vals=std_vals,
        percentiles=[100, 50, 5],
    )

    # ---------------------------------------------------
    # traces - ensemble members
    # ---------------------------------------------------
    smoothers = [
        'eks-temporal-2',
        'eks-temporal-3',
        'eks-temporal-4',
        'eks-temporal-5',
        'eks-temporal-6',
        'eks-temporal-8',
    ]
    raw_models = [f'raw-{i}' for i in range(8)]
    plot_traces(
        df_smoothers={s: trace_data[s] for s in smoothers},
        df_raw={s: trace_data[s] for s in raw_models},
        keypoint=traces_keypoint,
        coord=traces_coord,
        time_window=traces_time_window,
        fig=fig,
        subplot_spec=gs[1, 1],
        colors=colors[:-1],
    )

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=300)
    plt.close()


def plot_figure5(data_dir, save_dir, format='pdf'):

    # ----------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------
    dataset_name = dataset_info_fig5['dataset_name']

    # load ground truth labels
    df_ground_truth = pd.read_csv(
        os.path.join(data_dir, dataset_name, 'labels_OOD.csv'),
        index_col=0,
        header=[1, 2],
    )
    df_ground_truth.sort_index(inplace=True)
    # update relative paths in labeled data to match model results
    df_ground_truth.index = [
        p.replace('labeled-data_OOD/', 'labeled-data/') for p in df_ground_truth.index
    ]

    # load model predictions from dlc models
    df_labeled_preds = pd.read_parquet(os.path.join(
        data_dir, 'results_dataframes', 'mirror-mouse_labeled_preds_dlc_post-processors.pqt',
    ))

    # load metrics for labeled frames for post-processors
    df_labeled_post = pd.read_parquet(os.path.join(
        data_dir, 'results_dataframes', 'mirror-mouse_labeled_metrics_dlc_post-processors.pqt',
    ))

    # load metrics for labeled frames for eks w/ varying ensemble members
    df_labeled_eks = pd.read_parquet(os.path.join(
        data_dir, 'results_dataframes',
        'mirror-mouse_labeled_metrics_dlc_post-processors-ensemble-size.pqt',
    ))

    # load model predictions for videos
    trace_file = os.path.join(
        data_dir, 'results_dataframes', 'mirror-mouse_video_preds_dlc_post-processors.pkl'
    )
    trace_data = pickle.load(open(trace_file, 'rb'))

    # drop keypoints
    cols_to_drop = dataset_info_fig5['cols_to_drop']
    cols_to_keep = dataset_info_fig5['cols_to_keep']
    if len(cols_to_drop) > 0:
        df_ground_truth = df_ground_truth.drop(columns=cols_to_drop)
        df_labeled_preds = df_labeled_preds.drop(columns=cols_to_drop)
        df_labeled_post = df_labeled_post.drop(columns=cols_to_drop)
        df_labeled_eks = df_labeled_eks.drop(columns=cols_to_drop)
        # recompute means
        if len(cols_to_keep) > 0:
            df_labeled_post.loc[:, 'mean'] = df_labeled_post.loc[:, cols_to_keep].mean(axis=1)
            df_labeled_eks.loc[:, 'mean'] = df_labeled_eks.loc[:, cols_to_keep].mean(axis=1)

    # ----------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------
    save_file = os.path.join(save_dir, f'fig5_{dataset_name}.{format}')
    plot_figure5_all_panels(
        save_file=save_file,
        df_ground_truth=df_ground_truth,
        df_labeled_preds=df_labeled_preds,
        df_labeled_post=df_labeled_post,
        df_labeled_eks=df_labeled_eks,
        trace_data=trace_data,
        train_frames_list=dataset_info_fig5['train_frames_list'],
        train_frames_list_actual=dataset_info_fig5['train_frames_actual'],
        std_vals=dataset_info_fig5['std_vals'],
        traces_keypoint=dataset_info_fig5['keypoint'],
        traces_coord=dataset_info_fig5['coord'],
        traces_time_window=dataset_info_fig5['time_window'],
    )
