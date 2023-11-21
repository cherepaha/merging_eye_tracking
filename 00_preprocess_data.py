import pandas as pd
import os
import helper

def get_trial_metrics(trial):
    RT = trial.RT.iloc[0]
    dwell_mirror = len(trial[(trial.is_looking_at_mirror) & (trial.t<RT)])/len(trial[trial.t<RT])
    looked_at_mirror_early = len(trial[(trial.is_looking_at_mirror) & (trial.t<0.3)]) > 0
    return pd.Series({"dwell_mirror": dwell_mirror,
                      "looked_at_mirror_early": looked_at_mirror_early})

# download the data from https://osf.io/43bng/, update the data path to the directory with the downloaded data
raw_data_path = "../../surfdrive/data/merging_eye_tracking/raw"
participant_raw_data_path = os.path.join(raw_data_path, "Part%i/Output/Results_xy.txt")
processed_data_path = "data/processed"

mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max, front_x_min, front_x_max, front_y_min, front_y_max = helper.get_constants()

video_conditions = pd.DataFrame.from_dict(data={1: [4, 20, 4], 2: [4, 20, 6], 3: [4, 30, 4],
                                                4: [4, 30, 6], 5: [4, 40, 4], 6: [4, 40, 6],
                                                7: [6, 20, 4], 8: [6, 20, 6], 9: [6, 30, 4],
                                                10: [6, 30, 6], 11: [6, 40, 4], 12: [6, 40, 6]},
                                          orient="index", columns=["time_budget", "d", "tta"])
video_conditions.index.name = "video"

participant_dfs = []

for participant_id in range(1, 27):
    participant_df = pd.read_csv(participant_raw_data_path % participant_id, sep="\t", low_memory=False)
    # Discard eye movements within one frame, only use the first record per frame
    participant_df = participant_df.groupby(["TRIAL_INDEX", "VIDEO_FRAME_INDEX"]).first().reset_index()
    participant_df["participant"] = participant_id

    # participant 17's data misses video number and frame index so we exclude their data
    if not participant_id == 17:
        participant_dfs.append(participant_df)

data = pd.concat(participant_dfs)

data["VIDEO_FRAME_INDEX"] = pd.to_numeric(data["VIDEO_FRAME_INDEX"], errors='coerce')
data = data.dropna(subset=["VIDEO_FRAME_INDEX"])
data.loc[:, "VIDEO_FRAME_INDEX"] = data["VIDEO_FRAME_INDEX"].astype("Int64")
data["AVERAGE_GAZE_X"] = pd.to_numeric(data["AVERAGE_GAZE_X"], errors='coerce')
data["AVERAGE_GAZE_Y"] = pd.to_numeric(data["AVERAGE_GAZE_Y"], errors='coerce')

# exclude 5 trials with no indicated decision
print("Number of trials with no indicated decision:")
print(len(data[(data["VARIABLE_Key_Pressed"] == ".")].groupby(["participant", "TRIAL_INDEX"]).first()))
data = data[~(data["VARIABLE_Key_Pressed"] == ".")]

data = data.rename(columns={"TRIAL_INDEX": "trial",
                            "VARIABLE_RT": "RT",
                            "AVERAGE_GAZE_X": "eye_x",
                            "AVERAGE_GAZE_Y": "eye_y"})

# in the EyeLink coordinate frame, the origin is in the top left corner; for intuitive plotting we place the origin in the bottom left corner
data["eye_y"] = (1080 - data["eye_y"])
# with the framerate of 20 frames/s (each frame presented for 50ms), time can be calculated from the frame number
data["t"] = data["VIDEO_FRAME_INDEX"] * 0.05
data["video"] = data["VIDEO_NAME"].str.extract('(\d+)').astype("Int64")

# Categorize each data point according to which area of interest (AOI) was gazed at that moment
data["AOI"] = "other"
data.loc[(data.eye_x > mirror_x_min) & (data.eye_x < mirror_x_max) & (data.eye_y > mirror_y_min) & (
            data.eye_y < mirror_y_max), "AOI"] = "mirror"
data.loc[(data.eye_x > front_x_min) & (data.eye_x < front_x_max) & (data.eye_y > front_y_min) & (
            data.eye_y < front_y_max), "AOI"] = "front"

data["is_looking_at_mirror"] = (data["AOI"] == "mirror")
data["is_looking_in_front"] = (data["AOI"] == "front")
data["is_looking_elsewhere"] = (data["AOI"] == "other")

data["is_gap_accepted"] = (data["VARIABLE_Key_Pressed"] == "Lshift")
data["decision"] = "Reject"
data.loc[data.is_gap_accepted, "decision"] = "Accept"

# convert RT from ms to s
data["RT"] /= 1000

# add information on tta/d/time budget conditions
data = data.join(video_conditions, on="video")
data = data.sort_values(["participant", "trial", "t"], ascending=True)

# saving the resulting data
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)

columns_to_save = ["participant", "trial", "video", "tta", "d", "time_budget", "decision", "RT", "is_gap_accepted",
                   "t", "eye_x", "eye_y", "AOI", "is_looking_at_mirror", "is_looking_in_front", "is_looking_elsewhere"]
data[columns_to_save].to_csv(os.path.join(processed_data_path, "processed_eye_data.csv"), index=False)

# get and save trial-level metrics (relative dwell time on the mirror and presence of early fixations on the mirror)
metrics = (data.groupby(["participant", "trial"])
           .apply(get_trial_metrics)
           .join(data.groupby(["participant", "trial"]).first()[["tta", "d", "time_budget", "decision", "RT", "is_gap_accepted"]])
           .reset_index())

metrics.to_csv(os.path.join(processed_data_path, "metrics.csv"), index=False)