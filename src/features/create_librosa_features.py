# Create a data set using features extracted by librosa.
import multiprocessing
from os import path
import pickle

import common
from dbispipeline.utils import prefix_path
import librosa
import numpy as np
import pandas as pd

DATA_PATH = prefix_path("audio_data", common.DEFAULT_PATH)


def extract_features(song_path):
    # Load song.
    song, sr = librosa.load(song_path)

    # Extract BPM.
    bpm, _ = librosa.beat.beat_track(y=song, sr=sr)

    # Extract zero-crossing rate.
    zcr = sum(librosa.zero_crossings(y=song)) / len(song)

    # Extract spectral centroid.
    spec_centroid = np.mean(
        librosa.feature.spectral_centroid(y=song, sr=sr)[0])
    spec_centroid_stddev = np.std(
        librosa.feature.spectral_centroid(y=song, sr=sr)[0])

    # Extract spectral rolloff.
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=song, sr=sr)[0])
    spec_rolloff_stddev = np.std(
        librosa.feature.spectral_rolloff(y=song, sr=sr)[0])

    # Extract spectral flatness.
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=song)[0])
    spec_flat_stddev = np.std(librosa.feature.spectral_flatness(y=song)[0])

    # Extract spectral contrast.
    spec_contrast = np.mean(
        librosa.feature.spectral_contrast(y=song, sr=sr)[0])
    spec_contrast_stddev = np.std(
        librosa.feature.spectral_contrast(y=song, sr=sr)[0])

    # Extract MFCCs.
    mfccs = librosa.feature.mfcc(y=song, sr=sr)
    mfcc = [np.mean(c) for c in mfccs]

    # Done.
    features = [
        bpm,
        zcr,
        spec_centroid,
        spec_centroid_stddev,
        spec_rolloff,
        spec_rolloff_stddev,
        spec_flat,
        spec_flat_stddev,
        spec_contrast,
        spec_contrast_stddev,
    ]
    for c in mfcc:
        features.append(c)
    columns = [
        "bpm",
        "zcr",
        "spectral_centroid",
        "spectral_centroid_stddev",
        "spectral_rolloff",
        "spectral_rolloff_std",
        "spectral_flatness",
        "spectral_flatness_std",
        "spectral_contrast",
        "spectral_contrast_std",
    ]
    for i in range(len(mfcc)):
        columns.append(f"mfcc{i + 1}")

    return pd.DataFrame([features], columns=columns)


def process_line(line):
    i, line = line[0], line[1]
    print(f"{i}")

    # Get audio path and tags for the current song.
    fields = line.split("\t")
    mp3_path = path.join(DATA_PATH, fields[3])
    tags = [t.replace("\n", "") for t in fields[5:]]

    # Extract audio features for the given song.
    df_features = extract_features(mp3_path)

    # Construct result data frame.
    df_tags = pd.DataFrame([[tags]], columns=["tags"])
    df_res = df_features.join(df_tags)

    # Done.
    return df_res


def generate_data_set(set_path, save_path):
    # Process every song in the given set.
    with open(set_path, "r") as f:
        lines = enumerate(f.readlines()[1:])

    pool = multiprocessing.pool.Pool()
    frames = pool.map(process_line, lines)

    # Combine data frames.
    res = pd.concat(frames)

    # Save.
    pickle.dump(res, open(save_path, "wb"))


if __name__ == "__main__":
    generate_data_set(
        prefix_path("autotagging_moodtheme-train.tsv", common.DEFAULT_PATH),
        prefix_path("autotagging_moodtheme-train-librosa.pickle",
                    common.DEFAULT_PATH),
    )
    generate_data_set(
        prefix_path("autotagging_moodtheme-test.tsv", common.DEFAULT_PATH),
        prefix_path("autotagging_moodtheme-test-librosa.pickle",
                    common.DEFAULT_PATH),
    )
