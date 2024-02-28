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
}

print('Downloading data - this may take 10-15 minutes depending on your download speed')

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

    # extract data
    print(f'extracting data to {output_dir_dataset}...', end='', flush=True)
    for file in z.namelist():
        z.extract(file, output_dir)
    print('done')
