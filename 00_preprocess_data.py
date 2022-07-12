import pandas as pd

measures = pd.read_csv("measures_raw.csv")
measures = measures.rename(columns={"Participant_no": "subj_id",
                                    "Time_gap_onramp": "tta_or_condition",
                                    "Time_gap_upcoming_vehicle": "tta_condition",
                                    "Distance_gap_upcoming_vehicle": "d_condition",
                                    "Is_gap_accepted": "is_gap_accepted",
                                    "Response_time": "RT",
                                    "Dwell_to_mirror": "dwell_time"})
measures["is_gap_accepted"] = measures["is_gap_accepted"].astype("bool")

measures["RT"] /= 1000
measures["decision"] = "Wait"
measures.loc[measures.is_gap_accepted, ["decision"]] = "Merge"

measures.to_csv("measures.csv", sep=",", index=False)
