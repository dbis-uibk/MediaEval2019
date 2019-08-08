import os
import glob
import json
import pickle
import pandas as pd
from pandas.io.json import json_normalize

columns_to_drop = [
    'lowlevel.barkbands.max',
    'lowlevel.barkbands.mean', 
    'lowlevel.barkbands.median',
    'lowlevel.barkbands.min', 
    'lowlevel.barkbands.var',
    'lowlevel.erbbands.max', 
    'lowlevel.erbbands.mean',
    'lowlevel.erbbands.median', 
    'lowlevel.erbbands.min',
    'lowlevel.erbbands.var', 
    'lowlevel.gfcc.cov', 
    'lowlevel.gfcc.icov', 
    'lowlevel.gfcc.mean',
    'lowlevel.hfc.max', 
    'lowlevel.hfc.mean',
    'lowlevel.hfc.median', 
    'lowlevel.hfc.min', 
    'lowlevel.hfc.var',
    'lowlevel.melbands.max', 
    'lowlevel.melbands.mean',
    'lowlevel.melbands.median', 
    'lowlevel.melbands.min',
    'lowlevel.melbands.var', 
    'lowlevel.mfcc.cov', 
    'lowlevel.mfcc.icov', 
    'lowlevel.spectral_contrast_coeffs.median',
    'lowlevel.spectral_contrast_coeffs.var',
    'lowlevel.spectral_contrast_valleys.median',
    'lowlevel.spectral_contrast_valleys.var',
    'rhythm.beats_loudness_band_ratio.median',
    'rhythm.beats_loudness_band_ratio.var', 
    'rhythm.beats_position',
    'tonal.chords_histogram',
    'tonal.hpcp.max', 
    'tonal.hpcp.mean',
    'tonal.hpcp.median', 
    'tonal.hpcp.min', 
    'tonal.hpcp.var',
    'tonal.thpcp',
    'highlevel.danceability.probability',
    'highlevel.danceability.value', 
    'highlevel.gender.probability',
    'highlevel.gender.value',
    'highlevel.genre_dortmund.probability',
    'highlevel.genre_dortmund.value',
    'highlevel.genre_electronic.probability',
    'highlevel.genre_electronic.value',
    'highlevel.genre_rosamerica.probability',
    'highlevel.genre_rosamerica.value',
    'highlevel.genre_tzanetakis.probability',
    'highlevel.genre_tzanetakis.value',
    'highlevel.ismir04_rhythm.probability',
    'highlevel.ismir04_rhythm.value',
    'highlevel.mood_acoustic.probability',
    'highlevel.mood_acoustic.value',
    'highlevel.mood_aggressive.probability',
    'highlevel.mood_aggressive.value',
    'highlevel.mood_electronic.probability',
    'highlevel.mood_electronic.value',
    'highlevel.mood_happy.probability', 
    'highlevel.mood_happy.value',
    'highlevel.mood_party.probability', 
    'highlevel.mood_party.value',
    'highlevel.mood_relaxed.probability',
    'highlevel.mood_relaxed.value', 
    'highlevel.mood_sad.probability',
    'highlevel.mood_sad.value', 
    'highlevel.moods_mirex.probability', 
    'highlevel.moods_mirex.value',
    'highlevel.timbre.probability', 
    'highlevel.timbre.value',
    'highlevel.tonal_atonal.probability',
    'highlevel.tonal_atonal.value',
    'highlevel.voice_instrumental.probability',
    'highlevel.voice_instrumental.value',
]

def expand_list_column(df, column):
    # Convert columnto series and rename them.
    col = df[column].apply(pd.Series)
    col = col.rename(columns=lambda x: f"{column}_{x}")

    # Append new columns to the dataframe.
    df = pd.concat([df[:], col[:]], axis=1)

    # Drop the old column.
    df = df.drop([column], axis=1)

    return df

def create_dataset(chart_file, feature_directory, target_file):
    # Read dataset.
    lines = None
    with open(chart_file, "r") as f:
        lines = f.readlines()[1:]

    # Load the features for every song in the chart data
    # into a dataframe.
    track_frames = []
    for line in lines:
        fields = line.split("\t")

        song_id = fields[0]
        tags = [t.replace("\n", "") for t in fields[5:]]
        base_path = os.path.join(feature_directory, fields[3]).replace(".mp3", "")

        lowlevel_files = glob.glob(f"{base_path}.json")
        highlevel_files = glob.glob(f"{base_path}.json.highlevel.json")

        # Load features and pack them into dataframes
        df_meta = pd.DataFrame([[song_id, tags]], columns=["#ID", "#tags"])
        df_low = json_normalize(json.load(open(lowlevel_files[0], "r")))
        df_high = json_normalize(json.load(open(highlevel_files[0], "r")))

        df = df_meta.join(df_low)
        df = df.join(df_high, rsuffix="_high")

        track_frames.append(df)

    # Join into one dataframe.
    df = pd.concat(track_frames)

    # Replace selected list columns with expanded versions.
    df = expand_list_column(df, "lowlevel.mfcc.mean")
    df = expand_list_column(df, "lowlevel.spectral_contrast_coeffs.mean")
    df = expand_list_column(df, "lowlevel.spectral_contrast_coeffs.max")
    df = expand_list_column(df, "lowlevel.spectral_contrast_coeffs.min")
    df = expand_list_column(df, "lowlevel.spectral_contrast_valleys.mean")
    df = expand_list_column(df, "lowlevel.spectral_contrast_valleys.max")
    df = expand_list_column(df, "lowlevel.spectral_contrast_valleys.min")
    df = expand_list_column(df, "rhythm.beats_loudness_band_ratio.mean")
    df = expand_list_column(df, "rhythm.beats_loudness_band_ratio.max")
    df = expand_list_column(df, "rhythm.beats_loudness_band_ratio.min")

    # Drop other unneeded columns.
    df = df.drop(list(filter(lambda x: x.startswith("metadata"), df.columns.values)), axis=1) # Metadata

    df = df.drop(list(filter(lambda x: x.endswith("dmean"), df.columns.values)), axis=1)
    df = df.drop(list(filter(lambda x: x.endswith("dmean2"), df.columns.values)), axis=1)
    df = df.drop(list(filter(lambda x: x.endswith("dvar"), df.columns.values)), axis=1)
    df = df.drop(list(filter(lambda x: x.endswith("dvar2"), df.columns.values)), axis=1)

    df = df.drop(columns_to_drop, axis=1)

    # One-hot encode string features.
    string_columns = ['tonal.chords_key', 'tonal.chords_scale', 'tonal.key_scale', 'tonal.key_key']
    for c in string_columns:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=False)
        df = pd.concat([df.drop(c, axis=1), dummies], axis=1)

    # Save dataframe.
    pickle.dump(df, open(target_file, "wb"))

if __name__ == "__main__":
    create_dataset("/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train.tsv", "/storage/nas3/datasets/music/mediaeval2019/acousticbrainz_data", "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-train.pickle")
    create_dataset("/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test.tsv", "/storage/nas3/datasets/music/mediaeval2019/acousticbrainz_data", "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-test.pickle")
    create_dataset("/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-validation.tsv", "/storage/nas3/datasets/music/mediaeval2019/acousticbrainz_data", "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-validation.pickle")
