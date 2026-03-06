# src/elp_rumble/data_creation/neg_audio_clips.py
"""
Usage Instructions:
-------------------
It is recommended to first run pos_audio_clips.py in both test and train modes,
as the number of positive clips generated will be used to determine the number
of negative clips that should be created in order to ensure a balanced dataset.

Ensure your virtual environment is activated before running.
"""

import os
import random
import wave
from pathlib import Path
import numpy as np
from .utils import (
    apply_low_pass_filter,
    count_wavs,
    down_sample,
    find_wav_files,
    get_wav_params,
    save_audio_to_wav,
    train_test_split,
    validate_dir,
)
from elp_rumble.config.paths import (
    NEG_SOURCE_INPUT_DIR,
    TRAIN_VAL_NEG_CLIPS_DIR,
    HOLDOUT_TEST_NEG_CLIPS_DIR,
    POS_TRAIN_VAL_CLIPS_DIR,
    POS_HOLDOUT_TEST_CLIPS_DIR,
)

def main():
    validate_dir(NEG_SOURCE_INPUT_DIR, "Negative source input directory")

    os.makedirs(TRAIN_VAL_NEG_CLIPS_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_TEST_NEG_CLIPS_DIR, exist_ok=True)

    neg_wav_files = []
    find_wav_files(NEG_SOURCE_INPUT_DIR, neg_wav_files)
    print(f"{len(neg_wav_files)} negative wav files found in {NEG_SOURCE_INPUT_DIR}")

    if not neg_wav_files:
        raise ValueError(
            f"No .wav files found in negative source directory: {NEG_SOURCE_INPUT_DIR}. "
            "Provide valid local raw data and rerun."
        )

    # Shuffle wav files for randomness
    random.seed(42)
    random.shuffle(neg_wav_files)

    print(f"Checking number of existing positive clips to ensure balanced dataset when generating negative clips.")

    pos_train_clips_count = count_wavs(POS_TRAIN_VAL_CLIPS_DIR)
    pos_test_clips_count = count_wavs(POS_HOLDOUT_TEST_CLIPS_DIR)
    print(f"pos_train_clips_count: {pos_train_clips_count}")
    print(f"pos_test_clips_count: {pos_test_clips_count}")

    # Fallbacks should mirror requested train/test intent when positives are absent.
    DEFAULT_TRAIN_RATIO = 0.8
    DEFAULT_MAX_TRAIN = 10698
    DEFAULT_MAX_TEST  = 4784

    have_train = pos_train_clips_count > 0
    have_test  = pos_test_clips_count > 0

    if have_train and have_test:
        total_pos = pos_train_clips_count + pos_test_clips_count
        train_ratio = pos_train_clips_count / total_pos
        print(f"Train ratio: {train_ratio:.4f}")
    else:
        train_ratio = DEFAULT_TRAIN_RATIO
        print(f"Positive clips missing, using default train ratio: {train_ratio}")

    max_train_clips = pos_train_clips_count if have_train else DEFAULT_MAX_TRAIN
    max_test_clips  = pos_test_clips_count  if have_test  else DEFAULT_MAX_TEST

    if not have_train:
        print(f"No positive train clips found, default neg train clips: {max_train_clips}")
    if not have_test:
        print(f"No positive test clips found, default neg test clips: {max_test_clips}")

    if len(neg_wav_files) < 2:
        raise ValueError("Need at least 2 negative wav files to create train/test split.")

    # Train test split of input .wavs
    test_ratio = 1.0 - train_ratio
    neg_train_wavs, neg_test_wavs = train_test_split(
        neg_wav_files,
        test_size=test_ratio,
        random_state=42,
    )
    num_neg_train_input_wavs = len(neg_train_wavs)
    num_neg_test_input_wavs = len(neg_test_wavs)
    print(f"Using {num_neg_train_input_wavs} neg input wavs for creating training clips and {num_neg_test_input_wavs} neg input wavs for creating testing clips.")
    print(f"Train test split performed.")
    print(f"Training neg input wav files: \n{neg_train_wavs}")
    print(f"Testing neg input wav files: \n{neg_test_wavs}")

    # Process clips
    sample_length = 5 # seconds
    target_sr = 4000 # hz
    expected_final_frames = sample_length * target_sr  # Expected length after downsampling

    for split_type in ["train", "test"]:

        print(f"Generating neg {split_type}ing clips.")

        if split_type == "train":
            max_clips = max_train_clips
            input_wav_files = neg_train_wavs
            output_dir = TRAIN_VAL_NEG_CLIPS_DIR
        else:
            max_clips = max_test_clips
            input_wav_files = neg_test_wavs
            output_dir = HOLDOUT_TEST_NEG_CLIPS_DIR

        counter = 0

        for file in input_wav_files:
            file_stem = Path(file).stem
            params = get_wav_params(file)
            max_frames = params.nframes
            sample_width = params.sampwidth

            sample_frames_to_grab = sample_length * params.framerate

            if max_frames >= sample_frames_to_grab and counter < max_clips:
                starting_pos = 0

                while starting_pos + sample_frames_to_grab <= max_frames:
                    
                    with wave.open(file, 'rb') as wav_file:
                        wav_file.setpos(starting_pos)
                        frames = wav_file.readframes(sample_frames_to_grab)

                        dtype = None
                        if sample_width == 1:
                            dtype = np.int8
                        elif sample_width == 2:
                            dtype = np.int16
                        elif sample_width == 4:
                            dtype = np.int32
                        else:
                            exit()

                        data = np.frombuffer(frames, dtype=dtype)

                        if params.nchannels > 1:
                            data = data.reshape((-1, params.nchannels))
                    
                    data = apply_low_pass_filter(data, params.framerate, cutoff_hz=200)
                    data = down_sample(data, params.framerate, target_sr, expected_final_frames)

                    if len(data) == expected_final_frames:
                        save_audio_to_wav(
                            os.path.join(output_dir, f"{file_stem}_neg_{round(starting_pos)}_{counter}.wav"),
                            data,
                            target_sr,
                        )
                        counter += 1
                    else:
                        print("Not saving, insufficient frames")

                    if counter == max_clips:
                        print(f"{max_clips} neg {split_type}ing clips generated.")
                        break

                    starting_pos += int(sample_length * params.framerate)

if __name__ == "__main__":
    main()