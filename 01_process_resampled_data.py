import pandas as pd
import os
import helper

def get_trial_metrics(trial):
    RT = trial.RT.iloc[0]
    dwell_mirror = len(trial[(trial.is_looking_at_mirror) & (trial.t<RT)])/len(trial[trial.t<RT])
    looked_at_mirror_early = len(trial[(trial.is_looking_at_mirror) & (trial.t<0.3)]) > 0
    return pd.Series({"dwell_mirror": dwell_mirror,
                      "looked_at_mirror_early": looked_at_mirror_early})

resampled_raw_data_path = "../../surfdrive/data/merging_eye_tracking/resampled_raw"
participant_raw_data_path = os.path.join(resampled_raw_data_path, "part_%i.csv")

processed_data_path = "../../surfdrive/data/merging_eye_tracking/processed"
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)

mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max, front_x_min, front_x_max, front_y_min, front_y_max = helper.get_constants()

video_conditions = pd.DataFrame.from_dict(data={1: [4, 20, 4], 2: [4, 20, 6], 3: [4, 30, 4],
                                                4: [4, 30, 6], 5: [4, 40, 4], 6: [4, 40, 6],
                                                7: [6, 20, 4], 8: [6, 20, 6], 9: [6, 30, 4],
                                                10: [6, 30, 6], 11: [6, 40, 4], 12: [6, 40, 6]},
                                          orient="index", columns=["time_budget", "d", "tta"])
video_conditions.index.name = "video"

data = pd.concat([pd.read_csv(participant_raw_data_path % participant_id, sep=",") for participant_id in range(3, 27)])

# exclude 6 trials with no indicated decision
print("Number of trials with no indicated decision:")
print(len(data[(data["key_pressed"] == ".")].groupby(["participant", "trial"]).first()))
data = data[~(data["key_pressed"] == ".")]

# in the EyeLink coordinate frame, the origin is in the top left corner; for intuitive plotting we place the origin in the bottom left corner
data["eye_y"] = (1080 - data["eye_y"])

# Categorize each data point according to which area of interest (AOI) was gazed at that moment
data["AOI"] = "other"
data.loc[(data.eye_x > mirror_x_min) & (data.eye_x < mirror_x_max) & (data.eye_y > mirror_y_min) & (
            data.eye_y < mirror_y_max), "AOI"] = "mirror"
data.loc[(data.eye_x > front_x_min) & (data.eye_x < front_x_max) & (data.eye_y > front_y_min) & (
            data.eye_y < front_y_max), "AOI"] = "front"

data["is_looking_at_mirror"] = (data["AOI"] == "mirror")
data["is_looking_in_front"] = (data["AOI"] == "front")
data["is_looking_elsewhere"] = (data["AOI"] == "other")

data["is_gap_accepted"] = (data["key_pressed"] == "Lshift")
data["decision"] = "Reject"
data.loc[data.is_gap_accepted, "decision"] = "Accept"

# add information on tta/d/time budget conditions
data = data.join(video_conditions, on="video")
data = data.sort_values(by=["participant", "trial", "t"], ascending=True)

# saving the resulting data
columns_to_save = ["participant", "trial", "video", "tta", "d", "time_budget", "decision", "RT", "is_gap_accepted",
                   "t", "eye_x", "eye_y", "AOI", "is_looking_at_mirror", "is_looking_in_front", "is_looking_elsewhere"]
data[columns_to_save].to_csv(os.path.join(processed_data_path, "processed_eye_data.csv"), index=False)

# get and save trial-level metrics (relative dwell time on the mirror and presence of early fixations on the mirror)
metrics = (data.groupby(["participant", "trial"])
           .apply(get_trial_metrics)
           .join(data.groupby(["participant", "trial"]).first()[["tta", "d", "time_budget", "decision", "RT", "is_gap_accepted"]])
           .reset_index())

metrics.to_csv(os.path.join(processed_data_path, "metrics.csv"), index=False)