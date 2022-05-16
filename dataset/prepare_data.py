# Data must be prepared and augmented before it can be processed

"""
1. Load extracted datasets
2. Convert MP3 to WAV using SoX (single channel [mono], 16kHz de-facto standard for processing)
3. Augment the converted dataset
    - Remove empty audio files
    - Remove prolonged silence
    - Shuffle audio files
"""
import csv
import json
import logging
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import sox
from sox import Transformer
from tqdm import tqdm
from typing import List

data_dir = '../mozilla/en'
files_to_process = ['test.tsv', 'dev.tsv', 'train.tsv']
manifest_dir = '/'
num_threads = multiprocessing.cpu_count()
sample_rate = 16000
n_channels = 1
bit_depth = 16


def create_manifest(data: List[tuple], output_name: str, manifest_path: str):
    output_file = Path(manifest_path) / output_name
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with output_file.open(mode='w') as f:
        for wav_path, duration, text in tqdm(data, total=len(data)):
            if wav_path != '':
                # skip invalid input files that could not be converted
                f.write(
                    json.dumps({'audio_filepath': os.path.abspath(wav_path), "duration": duration, 'text': text})
                    + '\n'
                )


def process_files(csv_file, data_root, num_workers):
    wavs_dir = os.path.join(data_root, 'wavs/')
    os.makedirs(wavs_dir, exist_ok=True)
    audio_clips_path = os.path.dirname(csv_file) + '/clips/'

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.lower().strip()
        audio_path = os.path.join(audio_clips_path, file_path)
        if os.path.getsize(audio_path) == 0:
            logging.warning(f'Skipping empty audio file {audio_path}')
            return '', '', ''

        output_wav_path = os.path.join(wavs_dir, file_name + '.wav')

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.convert(samplerate=sample_rate, n_channels=n_channels, bitdepth=bit_depth)
            tfm.silence()
            # tfm.trim(TRIM_START, TRIM_END)
            # tfm.fade(fade_in_len=0.3, fade_out_len=0.3)
            tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

        duration = sox.file_info.duration(output_wav_path)
        return output_wav_path, duration, text

    logging.info('Running conversion of {}'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        data = []
        for row in reader:
            file_name = row['path']
            # add the mp3 extension if the tsv entry does not have it
            if not file_name.endswith('.mp3'):
                file_name += '.mp3'
            data.append((file_name, row['sentence']))
        with ThreadPool(num_workers) as pool:
            data = list(tqdm(pool.imap(process, data), total=len(data)))
    return data


def main():
    data_root = data_dir
    os.makedirs(data_root, exist_ok=True)

    for csv_file in files_to_process:
        data = process_files(
            csv_file=os.path.join(data_dir, csv_file),
            data_root=os.path.join(data_root, os.path.splitext(csv_file)[0]),
            num_workers=num_threads,
        )
        logging.info('Creating manifests...')
        create_manifest(
            data=data,
            output_name=f'commonvoice_{os.path.splitext(csv_file)[0]}_manifest.json',
            manifest_path=manifest_dir,
        )


if __name__ == "__main__":
    main()
