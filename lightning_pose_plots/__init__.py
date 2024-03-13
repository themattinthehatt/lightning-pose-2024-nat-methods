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
        'keypoints': ['pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r'],
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
        'keypoints': ['paw_l', 'paw_r'],
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


dataset_info_fig4 = {
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
}


ibl_session_ids_pupil = [
    'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
    '5285c561-80da-4563-8694-739da92e5dd0',
    '8928f98a-b411-497e-aa4b-aa752434686d',
    '3d59aa1a-b4ba-47fe-b9cf-741b5fdb0c7b',
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
    '5d6aa933-4b00-4e99-ae2d-5003657592e9',
    'a92c4b1d-46bd-457e-a1f4-414265f0e2d4',
    '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00',
    '9468fa93-21ae-4984-955c-e8402e280c83',
    '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
    'b658bc7d-07cd-4203-8a25-7b16b549851b',
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
    '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
    '5bcafa14-71cb-42fa-8265-ce5cda1b89e0',
    '6364ff7f-6471-415a-ab9e-632a12052690',
    'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
    'd0c91c3c-8cbb-4929-8657-31f18bffc294',  # example session in Fig. 5
    '30e5937e-e86a-47e6-93ae-d2ae3877ff8e',
    '931a70ae-90ee-448e-bedb-9d41f3eda647',
    '78b4fff5-c5ec-44d9-b5f9-d59493063f00',
    'c6db3304-c906-400c-aa0f-45dd3945b2ea',
    '9fe512b8-92a8-4642-83b6-01158ab66c3c',
    'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
    'aa20388b-9ea3-4506-92f1-3c2be84b85db',
    '0b7ee1b6-42db-46cd-a465-08f531366187',
    'd2f5a130-b981-4546-8858-c94ae1da75ff',
    '768a371d-7e88-47f8-bf21-4a6a6570dd6e',
    '948fd27b-507b-41b3-bdf8-f9f5f0af8e0b',
    '752456f3-9f47-4fbf-bd44-9d131c0f41aa',
    '81a1dca0-cc90-47c5-afe3-c277319c47c8',
    'a4747ac8-6a75-444f-b99b-696fff0243fd',
    '6668c4a0-70a4-4012-a7da-709660971d7a',
    '1ca83b26-30fc-4350-a616-c38b7d00d240',
    '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
    'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
    '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
    '3f6e25ae-c007-4dc3-aa77-450fd5705046',
    '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
    '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
    'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
    '821f1883-27f3-411d-afd3-fb8241bbc39a',
    '75b6b132-d998-4fba-8482-961418ac957d',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',
    '034e726f-b35f-41e0-8d6c-a22cc32391fb',
    '56b57c38-2699-4091-90a8-aba35103155e',
    'ee40aece-cffd-4edb-a4b6-155f158c666a',
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
    'dda5fc59-f09a-4256-9fb5-66c67667a466',
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
    '51e53aff-1d5d-4182-a684-aba783d50ae5',
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
    '3638d102-e8b6-4230-8742-e548cd87a949',
    '88224abb-5746-431f-9c17-17d7ef806e6a',
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
    '3f859b5c-e73a-4044-b49e-34bb81e96715',
    'b22f694e-4a34-4142-ab9d-2556c3487086',
    '746d1902-fa59-4cab-b0aa-013be36060d5',
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    'ecb5520d-1358-434c-95ec-93687ecd1396',
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
]


ibl_session_ids_paw = [
    '032ffcdf-7692-40b3-b9ff-8def1fc18b2e',
    '6ed57216-498d-48a6-b48b-a243a34710ea',
    '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
    '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',
    'a405053a-eb13-4aa4-850c-5a337e5dc7fd',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # example session in Fig. 5
    '158d5d35-a2ab-4a76-87b0-51048c5d5283',
    '7622da34-51b6-4661-98ae-a57d40806008',
    'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
    'f9860a11-24d3-452e-ab95-39e199f20a93',
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
    '2c44a360-5a56-4971-8009-f469fb59de98',
    'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
    '27ef44c0-acb2-4220-b776-477d0d5abd35',
    'dc21e80d-97d7-44ca-a729-a8e3f9b14305',
    '58b1e920-cfc8-467e-b28b-7654a55d0977',
    'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c',
    '6c6b0d06-6039-4525-a74b-58cfaa1d3a60',
    'ae8787b1-4229-4d56-b0c2-566b61a25b77',
    '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',
    'a7763417-e0d6-4f2a-aa55-e382fd9b5fb8',
    'ee212778-3903-4f5b-ac4b-a72f22debf03',
    '64e3fb86-928c-4079-865c-b364205b502e',
    '034e726f-b35f-41e0-8d6c-a22cc32391fb',
    '6364ff7f-6471-415a-ab9e-632a12052690',
    '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',
    '15f742e1-1043-45c9-9504-f1e8a53c1744',
    'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
    'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
    '90e524a2-aa63-47ce-b5b8-1b1941a1223a',
    '746d1902-fa59-4cab-b0aa-013be36060d5',
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    '7416f387-b302-4ca3-8daf-03b585a1b7ec',
    '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',
    'd901aff5-2250-467a-b4a1-0cb9729df9e2',
    '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',
    'd2832a38-27f6-452d-91d6-af72d794136c',
    'ee40aece-cffd-4edb-a4b6-155f158c666a',
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
    'dda5fc59-f09a-4256-9fb5-66c67667a466',
    'ecb5520d-1358-434c-95ec-93687ecd1396',
    '4b00df29-3769-43be-bb40-128b1cba6d35',
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
]
