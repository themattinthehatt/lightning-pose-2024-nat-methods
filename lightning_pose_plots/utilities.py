"""Utility functions for plots"""

# import cv2
import numpy as np
import os
import pandas as pd


def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices'
            )
            break
    return frames


def get_trace_mask(df, video_name, train_frames, model_type, rng_seed, metric_name=None):
    mask = ((df.train_frames == train_frames)
            & (df.rng_seed_data_pt == rng_seed)
            & (df.model_type == model_type)
            & (df.video_name == video_name))
    if metric_name is not None:
        mask = mask & (df.metric == metric_name)
    return mask


def load_results_dataframes(results_df_dir, dataset_name):

    # load lp results from dataframes
    df_labeled_preds = pd.read_parquet(
        os.path.join(results_df_dir, f'{dataset_name}_labeled_preds_lp.pqt'))
    df_labeled_metrics = pd.read_parquet(
        os.path.join(results_df_dir, f'{dataset_name}_labeled_metrics_lp.pqt'))
    df_video_preds = pd.read_parquet(
        os.path.join(results_df_dir, f'{dataset_name}_video_preds_lp.pqt'))
    df_video_metrics = pd.read_parquet(
        os.path.join(results_df_dir, f'{dataset_name}_video_metrics_lp.pqt'))

    # load dlc results from dataframes
    df_labeled_preds = pd.concat([
        df_labeled_preds,
        pd.read_parquet(
            os.path.join(results_df_dir, f'{dataset_name}_labeled_preds_dlc.pqt'))
    ])
    df_labeled_metrics = pd.concat([
        df_labeled_metrics,
        pd.read_parquet(os.path.join(results_df_dir, f'{dataset_name}_labeled_metrics_dlc.pqt'))
    ])
    df_video_preds = pd.concat([
        df_video_preds,
        pd.read_parquet(
            os.path.join(results_df_dir, f'{dataset_name}_video_preds_dlc.pqt'))
    ])
    df_video_metrics = pd.concat([
        df_video_metrics,
        pd.read_parquet(
            os.path.join(results_df_dir, f'{dataset_name}_video_metrics_dlc.pqt'))
    ])

    return df_labeled_preds, df_labeled_metrics, df_video_preds, df_video_metrics


def load_single_model_video_predictions_from_parquet(
    filepath,
    video_name,
    rng_seed_data_pt,
    train_frames,
    model_type,
):
    """Load markers from csv file assuming DLC format.

    Parameters
    ----------
    filepath : str
        absolute path of pqt file
    video_name : str
    rng_seed_data_pt : str
    train_frames : str
    model_type : str

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """

    # define first three rows as headers
    # drop first column ('scorer' at level 0) which just contains frame indices
    df_ = pd.read_parquet(filepath)
    df = df_[
        (df_.video_name == video_name)
        & (df_.rng_seed_data_pt == rng_seed_data_pt)
        & (df_.train_frames == train_frames)
        & (df_.model_type == model_type)
    ]
    df = df.drop(
        columns=['video_name', 'model_path', 'rng_seed_data_pt', 'train_frames', 'model_type']
    )
    # collect marker names from multiindex header
    marker_names = [c[0] for c in df.columns[::3]]
    markers = df.values
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


def cleanaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False)
    ax.tick_params(right=False)
