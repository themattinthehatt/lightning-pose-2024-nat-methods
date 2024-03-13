import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA

from lightning_pose_plots.utilities import cleanaxis, format_data_for_pca, load_results_dataframes


def pearsonr_ci(x, y, alpha=0.05):
    """Calculate Pearson correlation along with the confidence interval using scipy and numpy.

    from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default

    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals

    """
    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size-3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def get_scatter_mask(
    df, metric_name, train_frames, model_type, split_set=None, distribution=None, rng_seed=None,
):
    """Helper function to subselect data from labeled data dataframe."""
    mask = ((df.metric == metric_name)
            & (df.train_frames == train_frames)
            & (df.model_type == model_type)
           )
    if split_set is not None:
        mask = mask & (df.set == split_set)
    if distribution is not None:
        mask = mask & (df.distribution == distribution)
    if rng_seed is not None:
        mask = mask & (df.rng_seed_data_pt == rng_seed)
    return mask


def plot_scatters(
    df, metric_names, train_frames, split_set, distribution, model_types, keypoint, ax,
    add_diagonal=False, add_trendline=False, markersize=None, alpha=0.25,
    trendline_kwargs={},
):
    """Plot scatters using matplotlib, with option to add trendline."""

    mask_0 = get_scatter_mask(
        df=df, metric_name=metric_names[0], train_frames=train_frames, split_set=split_set,
        distribution=distribution, model_type=model_types[0])
    mask_1 = get_scatter_mask(
        df=df, metric_name=metric_names[1], train_frames=train_frames, split_set=split_set,
        distribution=distribution, model_type=model_types[1])
    df_xs = df[mask_0][keypoint]
    df_ys = df[mask_1][keypoint]
    assert np.all(df_xs.index == df_ys.index)
    xs = df_xs.to_numpy()
    ys = df_ys.to_numpy()
    xs = np.log10(xs)
    ys = np.log10(ys)
    rng_seed = df[mask_0].rng_seed_data_pt.to_numpy()
    mn = np.nanmin([np.nanmin(xs), np.nanmin(ys)])
    mx = np.nanmax([np.nanmax(xs), np.nanmax(ys)])
    for j, r in enumerate(np.unique(rng_seed)):
        ax.scatter(
            xs[rng_seed == r], ys[rng_seed == r], marker='.', color='k',
            s=markersize, alpha=alpha, label='RNG seed %s' % r)

    # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
    label_format = '{:,.1f}'
    # set x-axis ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels([label_format.format(10 ** x) for x in ticks_loc])
    # set y-axis ticks
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels([label_format.format(10 ** y) for y in ticks_loc])

    ret_vals = None
    if add_diagonal:
        span = mx - mn
        if not np.isnan(span):
            ax.plot([mn, mx], [mn, mx], 'k')

    if add_trendline:
        nan_idxs = np.isnan(xs) | np.isnan(ys)
        xs_nonan = xs[~nan_idxs]
        ys_nonan = ys[~nan_idxs]
        zs = np.polyfit(xs_nonan, ys_nonan, 1)
        p = np.poly1d(zs)
        r_val, p_val, lo, hi = pearsonr_ci(xs_nonan, ys_nonan)
        xs_sorted = np.sort(xs_nonan)
        ax.plot(xs_sorted, p(xs_sorted), **trendline_kwargs)
        ret_vals = r_val, p_val, lo, hi

    return ret_vals


def plot_figure2_scatters(data_dir, format='pdf'):

    dataset_name = 'mirror-mouse'
    results_df_dir = os.path.join(data_dir, 'results_dataframes')

    # ---------------------------------------------------
    # define analysis parameters
    # ---------------------------------------------------
    # define which keypoints to analyze
    cols_to_drop = [
        'obs_top', 'obsHigh_bot', 'obsLow_bot',
    ]
    cols_to_keep = (
        'paw1LH_top', 'paw2LF_top', 'paw3RF_top', 'paw4RH_top', 'tailBase_top',
        'tailMid_top', 'nose_top', 'paw1LH_bot', 'paw2LF_bot',
        'paw3RF_bot', 'paw4RH_bot', 'tailBase_bot', 'tailMid_bot', 'nose_bot',
    )

    split_set = 'test'
    train_frames = '75'
    model = 'baseline'
    labels_fontsize = 10

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------
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

    # load model predictions on both labeled frames and unlabeled videos
    _, df_labeled_metrics, _, _ = load_results_dataframes(
        results_df_dir=results_df_dir,
        dataset_name=dataset_name,
    )
    # add temporal norm
    df_labeled_metrics = pd.concat([
        df_labeled_metrics,
        pd.read_parquet(
            os.path.join(results_df_dir, f'{dataset_name}_labeled_metrics_temporal_lp.pqt'))
    ])

    # drop keypoints
    df_ground_truth = df_ground_truth.drop(columns=cols_to_drop)
    df_labeled_metrics = df_labeled_metrics.drop(columns=cols_to_drop)
    # recompute means
    if len(cols_to_keep) > 0:
        df_labeled_metrics.loc[:, 'mean'] = df_labeled_metrics.loc[:, cols_to_keep].mean(axis=1)

    # only keep frames with all keypoints labeled
    # when a body part’s label is missing we cannot compute the pixel error but can still compute
    # PCA losses over the predictions.
    # when the predictions are wrong, the PCA loss can be very high without being reflected in the
    # pixel error.
    # we thus compute the correlations only for frames with fully visible frames.
    index_to_keep = df_ground_truth[df_ground_truth.isna().sum(axis=1) == 0].index
    df_labeled_metrics_clean = df_labeled_metrics[df_labeled_metrics.index.isin(index_to_keep)]

    # ---------------------------------------------------
    # plot
    # ---------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(9, 3))
    gs = fig.add_gridspec(1, 1, top=0.75, hspace=0.3)

    plots = {
        'temporal_norm': 'Temporal difference\nloss (pixels)',
        'pca_multiview_error': 'Multi-view PCA\nloss (pixels)',
        'pca_singleview_error': 'Pose PCA\nloss (pixels)'
    }

    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, len(plots), subplot_spec=gs[0, 0], wspace=0.4, hspace=0.1)

    for i, (metric_name, ax_title) in enumerate(plots.items()):
        ax = fig.add_subplot(gs2[0, i])
        r_val, p_val, lo, hi = plot_scatters(
            df=df_labeled_metrics_clean,
            metric_names=['pixel_error', metric_name],
            train_frames=train_frames, split_set=split_set, distribution='OOD',
            model_types=[model, model],
            keypoint='mean',
            ax=ax, markersize=5, alpha=0.25,
            add_trendline=True,
            trendline_kwargs={'linestyle': '-', 'color': 'r', 'linewidth': 2}
        )
        ax.set_title(
            'r=%1.2f\n95%% CI = [%1.2f, %1.2f]' % (r_val, lo, hi),
            fontsize=labels_fontsize,
        )
        ax.set_xlabel('Pixel error', fontsize=labels_fontsize)
        ax.set_ylabel('%s' % ax_title, fontsize=labels_fontsize)
        cleanaxis(ax)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    # ----------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------
    plt.suptitle(f'{dataset_name} dataset', fontsize=14)
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir, f'fig2_loss_vs_pix_error_scatters.{format}'),
        dpi=300,
    )
    plt.close()


def plot_figure2_pca(data_dir, format='pdf'):

    # loop over pca loss types
    for loss_type in ['pca_singleview', 'pca_multiview']:

        # ---------------------------------------------------
        # define analysis parameters
        # ---------------------------------------------------
        if loss_type == 'pca_singleview':
            dataset_names = [
                'mirror-mouse',
                'mirror-fish',
                'crim13',
                'ibl-pupil',
            ]
            loss_name = 'Pose PCA'
            xlabel = 'Fraction of PCs Kept'
        else:
            dataset_names = [
                'mirror-mouse',
                'mirror-fish',
            ]
            loss_name = 'Multi-view PCA'
            xlabel = 'Number of PCs Kept'

        # ---------------------------------------------------
        # load data, compute variance explained
        # ---------------------------------------------------
        var_explained = {}
        for dataset_name in dataset_names:
            # load data
            labels_df_path = os.path.join(data_dir, dataset_name, 'labels_InD.csv')
            if not os.path.exists(labels_df_path):
                labels_df_path = os.path.join(data_dir, dataset_name, 'CollectedData.csv')
            labels_array = pd.read_csv(labels_df_path, header=[0, 1, 2], index_col=0).to_numpy()
            labels_array = format_data_for_pca(labels_array, loss_type, dataset_name)
            # remove NaNs
            isnan = np.isnan(np.sum(labels_array, axis=1))
            labels_array = labels_array[~isnan]
            # fit pca
            pca = PCA()
            pca.fit(labels_array)
            # compute cumulative var explained
            cum_sum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
            var_explained[dataset_name] = cum_sum_explained_variance

        # ---------------------------------------------------
        # plot
        # ---------------------------------------------------
        plt.figure(figsize=(4.5, 4.5))

        fraction_to_keep = 0.99

        for dataset_name in dataset_names:
            num_total_comps = var_explained[dataset_name].shape[0]
            if loss_type == 'pca_singleview':
                xrange = np.arange(1, num_total_comps + 1) / num_total_comps
                label = f'{dataset_name} ({num_total_comps}D)'
            else:
                xrange = np.arange(1, num_total_comps + 1)
                label = f'{dataset_name} ({num_total_comps // 2} views)'
            plt.plot(
                xrange, var_explained[dataset_name],
                label=label,
                marker='o', linewidth=2, markersize=4
            )

        if loss_type == 'pca_singleview':
            xmin, xmax = 0, 1
            textx, texty = -0.04, fraction_to_keep - 0.06
        else:
            xmin, xmax = 0.5, 6
            textx, texty = 0.5, fraction_to_keep - 0.04

        plt.hlines(
            y=fraction_to_keep,
            xmin=xmin,
            xmax=xmax,
            color='black', linestyles='dashed', linewidth=2
        )
        plt.text(textx, texty, '0.99', fontsize=14)

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('Cumulative Explained Variance', fontsize=16)
        plt.title(f'{loss_name}', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()

        # ----------------------------------------------------------------
        # cleanup
        # ----------------------------------------------------------------
        fig_dir = os.path.join(data_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(
            os.path.join(fig_dir, f'fig2_{loss_type}_cumulative_variance.{format}'),
            dpi=300,
        )
        plt.close()


def plot_figure2(data_dir, format='pdf'):

    # plot metric vs pixel error scatters
    plot_figure2_scatters(data_dir=data_dir, format=format)

    # plot cumulative var pca curves
    plot_figure2_pca(data_dir=data_dir, format=format)
