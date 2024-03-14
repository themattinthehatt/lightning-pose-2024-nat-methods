import argparse
import os
import io
import requests
import zipfile as zf


parser = argparse.ArgumentParser(description='Download Lightning Pose data')
parser.add_argument('--data_dir', type=str)

args = parser.parse_args()
output_dir = args.data_dir

os.makedirs(output_dir, exist_ok=True)

# list out datasets and their locations here
datasets_url_dict = {
    'mirror-mouse': 'https://figshare.com/ndownloader/files/44031561',
    'mirror-fish': 'https://figshare.com/ndownloader/files/44031591',
    'crim13': 'https://figshare.com/ndownloader/files/44031672',
    'ibl-pupil': 'https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/Tags/2023_Q1_Biderman_Whiteway_et_al/_ibl_videoTracking.trainingDataPupil.27dcdbb6-3646-4a50-886d-03190db68af3.zip',  # noqa
    'ibl-paw': 'https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/Tags/2023_Q1_Biderman_Whiteway_et_al/_ibl_videoTracking.trainingDataPaw.7e79e865-f2fc-4709-b203-77dbdac6461f.zip',  # noqa
    # 'results_dataframes': 'todo',
}

print('Downloading data - this will take 10-15 minutes depending on your download speed')

for dataset, url in datasets_url_dict.items():

    # check if data exists
    output_dir_dataset = os.path.join(output_dir, dataset)
    if os.path.exists(output_dir_dataset):
        print(f'data already exists at {output_dir_dataset}; skipping')
        continue

    print(f'fetching {dataset} from url...', end='', flush=True)
    r = requests.get(url, stream=True)
    z = zf.ZipFile(io.BytesIO(r.content))
    print('done')

    print(f'extracting data to {output_dir_dataset}...', end='', flush=True)
    for file in z.namelist():
        z.extract(file, output_dir)
    print('done')
