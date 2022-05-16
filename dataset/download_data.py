# Downloads an open-source dataset from Mozilla CommonVoice https://commonvoice.mozilla.org/en/datasets

import argparse
import logging
import os
import wget

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Download and process dataset for training purposes.')
    parser.add_argument("--target_path", default='./mozilla', type=str, help="Directory to store dataset.")
    parser.add_argument("--ver", default='cv-corpus-1.1-2019-02-25', type=str, help="Specifies which version to select")
    parser.add_argument("--lang", default='en', type=str, help="Specifies the English variant of the dataset")
    args = parser.parse_args()
    COMMON_VOICE_URL = f"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/{}.tar.gz".format(args.ver, args.lang)

    target_path = args.target_path
    os.makedirs(target_path, exist_ok=True)
    download = wget.download(COMMON_VOICE_URL, target_path)
