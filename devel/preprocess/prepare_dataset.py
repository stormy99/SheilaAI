# Pre-process and prepare the dataset prior to training/creating the model.
# Extract features (labels, MFCs, MFCCs and pass numerical mappings - not characters)

import json
import librosa
import os

DATA_PATH = "../../dataset/google"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050  # SR 22.05kHz, 2secs worth of sound, def librosa config


# go through audio files, extract MFCCs, store in json, used in learning
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # create a data dictionary
    data = {
        "mapping": [],  # should pass numbers not words to neutral-network
        "labels": [],  # expected target outputs from mappings
        "MFCCs": [],  # MFC coefficients
        "files": []  # directories
    }

    # loop through all subdirectories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure operation occurs at subdirectory level (i.e. not root)
        if dirpath is not dataset_path:

            # update mappings
            label = dirpath.split("/")[-1]  # dataset/category -> [dataset, category]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # loop through all filenames and extract MFCCs
            for f in filenames:

                # get filepath
                file_path = os.path.join(dirpath, f)

                # load audiofile
                signal, sr = librosa.load(file_path)

                # check if audiofile >= 2 second
                if len(signal) >= SAMPLE_RATE:
                    # enforce 2 second duration signal
                    signal = signal[:SAMPLE_RATE]

                    # extract MFCCs
                    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                                 n_fft=n_fft, hop_length=hop_length)

                    # store data
                    data["MFCCs"].append(mfccs.T.tolist())  # transpose array to python list
                    data["labels"].append(i-1)  # subtract one from iteration of current subdirectory
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # store in JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    print("Preparing dataset..")
    prepare_dataset(DATA_PATH, JSON_PATH)
    print("Finished prepping dataset.")
    exit(0)
