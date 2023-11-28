import pandas as pd
import os
import helper

raw_data_path = "../../surfdrive/data/merging_eye_tracking/raw"
participant_raw_data_path = os.path.join(raw_data_path, "Data_Part%i.txt")

resampled_raw_data_path = "../../surfdrive/data/merging_eye_tracking/resampled_raw"
if not os.path.exists(resampled_raw_data_path):
    os.makedirs(resampled_raw_data_path)

# names of the columns in the data file exported from the edf files in DataViewer
columns_edf = ["TRIAL_INDEX", "VIDEO_NAME", "LEFT_GAZE_X", "LEFT_GAZE_Y", "RIGHT_GAZE_X", "RIGHT_GAZE_Y", "TIMESTAMP",
               "VARIABLE_Key_Pressed", "VARIABLE_RT"]
gaze_cols = ["LEFT_GAZE_X", "LEFT_GAZE_Y", "RIGHT_GAZE_X", "RIGHT_GAZE_Y"]

for participant_id in range(3, 27):
    participant_df = pd.read_csv(participant_raw_data_path % participant_id, sep="\t", low_memory=False,
                                 usecols=columns_edf)
    participant_df["participant"] = participant_id
    participant_df["trial"] = participant_df.TRIAL_INDEX

    # Extract video number from name
    participant_df["video"] = participant_df.VIDEO_NAME.str.extract('(\d+)').astype("Int64")
    participant_df = participant_df.dropna(subset=["video"])
    participant_df[gaze_cols] = participant_df[gaze_cols].apply(pd.to_numeric, errors='coerce')

    # update variables
    participant_df["key_pressed"] = participant_df["VARIABLE_Key_Pressed"]

    # rescale time
    participant_df.loc[:, "t"] = participant_df.groupby("TRIAL_INDEX")["TIMESTAMP"].transform(
        lambda t: (t - t.min())) / 1000
    participant_df["RT"] = participant_df["VARIABLE_RT"] / 1000

    # get average gaze location
    participant_df["eye_x"] = (participant_df["LEFT_GAZE_X"] + participant_df["RIGHT_GAZE_X"]) / 2
    participant_df["eye_y"] = (participant_df["LEFT_GAZE_Y"] + participant_df["RIGHT_GAZE_Y"]) / 2

    # drop old columns
    participant_df = participant_df.drop(columns=columns_edf)

    # resample eye trajectories
    participant_df_resampled = (participant_df.groupby(["trial"]).apply(helper.resample_trajectory)
                                .reset_index(drop=False).drop(['level_1'], axis=1))
    participant_df = participant_df_resampled.join(
        participant_df.groupby("trial").first().drop(columns=["t", "eye_x", "eye_y"]), on="trial")
    participant_df.to_csv(os.path.join(resampled_raw_data_path, "part_%i.csv" % participant_id), index=False)