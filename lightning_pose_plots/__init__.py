__version__ = "1.0.0"

import numpy as np
import seaborn as sns


colors_tab10 = sns.color_palette("tab10")
colors = [colors_tab10[4], colors_tab10[3], colors_tab10[1], colors_tab10[2], colors_tab10[0]]
model_order = ['dlc', 'baseline', 'context', 'semi-super', 'semi-super context']
model_colors = {model: colors[i] for i, model in enumerate(model_order)}

dataset_info = {
    'mirror-mouse': {
        'skeleton': [],
        'columns_for_singleview_pca': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        'mirrored_column_matches': [[0, 1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13, 14]],
    },
    'mirror-fish': {
        'skeleton': [
            ['chin_base_main', 'head_main'],
            ['chin1_4_main', 'chin_base_main'],
            ['chin_half_main', 'chin1_4_main'],
            ['chin3_4_main', 'chin_half_main'],
            ['chin_tip_main', 'chin3_4_main'],
            ['mid_main', 'head_main'],
            ['tail_neck_main', 'dorsal_main'],
            ['tail_neck_main', 'anal_main'],
            ['tail_neck_main', 'mid_main'],
            ['tail_neck_main', 'fork_main'],
            ['fork_main', 'caudal_d_main'],
            ['fork_main', 'caudal_v_main'],
            ['pectoral_L_main', 'pectoral_L_base_main'],
            ['pectoral_L_base_main', 'head_main'],
            ['pectoral_R_main', 'pectoral_R_base_main'],
            ['pectoral_R_base_main', 'head_main'],
            ['chin_base_top', 'head_top'],
            ['chin1_4_top', 'chin_base_top'],
            ['chin_half_top', 'chin1_4_top'],
            ['chin3_4_top', 'chin_half_top'],
            ['chin_tip_top', 'chin3_4_top'],
            ['mid_top', 'head_top'],
            ['tail_neck_top', 'dorsal_top'],
            ['tail_neck_top', 'anal_top'],
            ['tail_neck_top', 'mid_top'],
            ['tail_neck_top', 'fork_top'],
            ['fork_top', 'caudal_d_top'],
            ['fork_top', 'caudal_v_top'],
            ['pectoral_L_top', 'pectoral_L_base_top'],
            ['pectoral_L_base_top', 'head_top'],
            ['pectoral_R_top', 'pectoral_R_base_top'],
            ['pectoral_R_base_top', 'head_top'],
            ['chin_base_right', 'head_right'],
            ['chin1_4_right', 'chin_base_right'],
            ['chin_half_right', 'chin1_4_right'],
            ['chin3_4_right', 'chin_half_right'],
            ['chin_tip_right', 'chin3_4_right'],
            ['mid_right', 'head_right'],
            ['tail_neck_right', 'dorsal_right'],
            ['tail_neck_right', 'anal_right'],
            ['tail_neck_right', 'mid_right'],
            ['tail_neck_right', 'fork_right'],
            ['fork_right', 'caudal_d_right'],
            ['fork_right', 'caudal_v_right'],
            ['pectoral_L_right', 'pectoral_L_base_right'],
            ['pectoral_L_base_right', 'head_right'],
            ['pectoral_R_right', 'pectoral_R_base_right'],
            ['pectoral_R_base_right', 'head_right'],
        ],
        'columns_for_singleview_pca': [4, 5, 6, 7, 8, 9, 14, 15, 16, 22, 23, 24, 38, 39, 40, 41, 42, 43, 48, 50],
        'mirrored_column_matches': [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        ],
    },
    'crim13': {
        'skeleton': [
            ['black_mouse_nose', 'black_mouse_right_ear'],
            ['black_mouse_nose', 'black_mouse_left_ear'],
            ['black_mouse_top_of_neck', 'black_mouse_right_ear'],
            ['black_mouse_top_of_neck', 'black_mouse_left_ear'],
            ['black_mouse_top_of_neck', 'black_mouse_right_rear_knee'],
            ['black_mouse_top_of_neck', 'black_mouse_left_rear_knee'],
            ['black_mouse_right_rear_knee', 'black_mouse_base_of_tail'],
            ['black_mouse_left_rear_knee', 'black_mouse_base_of_tail'],
            ['white_mouse_nose', 'white_mouse_right_ear'],
            ['white_mouse_nose', 'white_mouse_left_ear'],
            ['white_mouse_top_of_neck', 'white_mouse_right_ear'],
            ['white_mouse_top_of_neck', 'white_mouse_left_ear'],
            ['white_mouse_top_of_neck', 'white_mouse_right_rear_knee'],
            ['white_mouse_top_of_neck', 'white_mouse_left_rear_knee'],
            ['white_mouse_right_rear_knee', 'white_mouse_base_of_tail'],
            ['white_mouse_left_rear_knee', 'white_mouse_base_of_tail'],
        ],
        'columns_for_singleview_pca': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'mirrored_column_matches': None,
    },
    'ibl-pupil': {
        'skeleton': [
            ['pupil_top_r', 'pupil_right_r'],
            ['pupil_right_r', 'pupil_bottom_r'],
            ['pupil_bottom_r', 'pupil_left_r'],
            ['pupil_left_r', 'pupil_top_r'],
        ],
        'columns_for_singleview_pca': [0, 1, 2, 3],
        'mirrored_column_matches': None,
    },
    'ibl-paw': {
        'skeleton': [],
        'columns_for_singleview_pca': None,
        'mirrored_column_matches': None,
    },
}


dataset_info_fig1_traces = {
    'model_type': 'dlc',
    'train_frames': '1',
    'video_name': '180609_000_25000_27000',
    'video_offset': 25000,
    'idxs': np.arange(500, 775),
    'keypoint': 'paw1LH_top',
}


dataset_info_fig1_sample_efficiency = {
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


dataset_info_fig3_examples = {
    'mirror-mouse-0': {
        'vid_name_tr': '180613_000_25000_27000',
        'vid_name_load': '180613_000',
        'frames_offset': 25000,
        'keypoint_tr': 'paw1LH_bot',
        'time_window_tr': (200, 425),
        'time_window_beg': 290,
        'n_frames': 5,
        'train_frames': '1',
        'metric_thresh': 20,
    },
    'mirror-fish-0': {
        'vid_name_tr': '20210202_Sean',
        'vid_name_load': '20210202_Sean',
        'frames_offset': 0,
        'keypoint_tr': 'caudal_d_right',
        'time_window_tr': (0, 2000),
        'time_window_beg': 1232,
        'n_frames': 5,
        'train_frames': '1',
        'metric_thresh': 20,
    },
    'crim13-0': {
        'vid_name_tr': '031309_A29_Block14_BCfe1_t_1000_3000',
        'vid_name_load': '031309_A29_Block14_BCfe1_t',
        'frames_offset': 1000,
        'keypoint_tr': 'black_mouse_base_of_tail',
        'time_window_tr': (950, 1150),
        'time_window_beg': 1007,
        'n_frames': 4,
        'train_frames': '800',
        'metric_thresh': 50,
    },
    'crim13-1': {
        'vid_name_tr': '110508_A24_Block4_castBCma1_t_1000_3000',
        'vid_name_load': '110508_A24_Block4_castBCma1_t',
        'frames_offset': 1000,
        'keypoint_tr': 'black_mouse_left_rear_knee',
        'time_window_tr': (1800, 2000),
        'time_window_beg': 1863,
        'n_frames': 4,
        'train_frames': '800',
        'metric_thresh': 50,
    },
}


dataset_info_fig3_metrics = {
    'mirror-mouse': {
        'cols_to_keep': (
            'paw1LH_top', 'paw2LF_top', 'paw3RF_top', 'paw4RH_top',
            'paw1LH_bot', 'paw2LF_bot', 'paw3RF_bot', 'paw4RH_bot',
        ),
        'cols_to_drop': [
            'obs_top', 'obsHigh_bot', 'obsLow_bot',
            'tailBase_top', 'tailMid_top', 'tailBase_bot', 'tailMid_bot',
            'nose_top', 'nose_bot',
        ],
        'bodyparts_list': ['paw1LH', 'paw2LF', 'paw3RF', 'paw4RH'],
        'views_list': ['top', 'bot'],
        'views_list_x': [],  # for mirror-fish
        'views_list_y': [],  # for mirror-fish
        'metric_thresh': 20,
        'max_frames': '1',
        'total_frames': 631,
    },
    'mirror-fish': {
        'cols_to_keep': (
            'head_main', 'mid_main', 'tail_neck_main',
            'head_top', 'mid_top', 'tail_neck_top',
            'head_right', 'mid_right', 'tail_neck_right',
        ),
        'cols_to_drop': [
            'chin_base_main', 'chin1_4_main', 'chin3_4_main', 'chin_half_main', 'chin_tip_main',
            'caudal_v_main', 'caudal_d_main', 'dorsal_main', 'anal_main', 'fork_main',
            'pectoral_L_base_main', 'pectoral_L_main', 'pectoral_R_base_main', 'pectoral_R_main',
            'chin_base_top', 'chin1_4_top', 'chin3_4_top', 'chin_half_top', 'chin_tip_top',
            'caudal_v_top', 'caudal_d_top', 'dorsal_top', 'anal_top', 'fork_top',
            'pectoral_L_base_top', 'pectoral_L_top', 'pectoral_R_base_top', 'pectoral_R_top',
            'chin_base_right', 'chin1_4_right', 'chin3_4_right', 'chin_half_right', 'chin_tip_right',
            'caudal_v_right', 'caudal_d_right', 'dorsal_right', 'anal_right', 'fork_right',
            'pectoral_L_base_right', 'pectoral_L_right', 'pectoral_R_base_right', 'pectoral_R_right',
        ],
        'bodyparts_list': ['head', 'mid', 'tail_neck'],
        'views_list_x': ['main', 'top'],    # views where we look at x-diffs to det outliers
        'views_list_y': ['main', 'right'],  # views where we look at y-diffs to det outliers
        'views_list': ['main', 'top', 'right'],
        'metric_thresh': 20,
        'max_frames': '1',
        'total_frames': 354,
    },
    'crim13': {
        'cols_to_keep': (),
        'cols_to_drop': [],
        'bodyparts_list': [
            'black_mouse_nose',
            'black_mouse_right_ear',
            'black_mouse_left_ear',
            'black_mouse_top_of_neck',
            'black_mouse_right_rear_knee',
            'black_mouse_left_rear_knee',
            'black_mouse_base_of_tail',
            'white_mouse_nose',
            'white_mouse_right_ear',
            'white_mouse_left_ear',
            'white_mouse_top_of_neck',
            'white_mouse_right_rear_knee',
            'white_mouse_left_rear_knee',
            'white_mouse_base_of_tail',
        ],
        'views_list': [],
        'views_list_x': [],
        'views_list_y': [],
        'metric_thresh': 20,
        'max_frames': '800',
        'total_frames': 800,
    },
}


# dataset_info_fig4 = {
#     'mirror-mouse': {
#         'cols_to_drop':
#         'cols_to_keep':
#         'vid_name':
#         'vid_name_load':
#         'keypoint':
#         'time_window':
#         'time_window_beg':
#         'n_frames':
#         'train_frames_list':
#         'train_frames_actual':
#         'std_vals':
#         'yticks':
#     },
#     'mirror-fish': {
#         'cols_to_drop':
#         'cols_to_keep':
#         'vid_name':
#         'vid_name_load':
#         'keypoint':
#         'time_window':
#         'time_window_beg':
#         'n_frames':
#         'train_frames_list':
#         'train_frames_actual':
#         'std_vals':
#         'yticks':
#     },
#     'crim13': {
#         'cols_to_drop':
#         'cols_to_keep':
#         'vid_name':
#         'vid_name_load':
#         'keypoint':
#         'time_window':
#         'time_window_beg':
#         'n_frames':
#         'train_frames_list':
#         'train_frames_actual':
#         'std_vals':
#         'yticks':
#     },
# }
