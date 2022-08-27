import pyddm
import numpy as np
import models
import loss_functions
import pandas as pd
import helper
import os

def get_gaze_sample_synthetic(simulation_params):
    # In a typical trial, participants looked
    # 1) at the on-ramp for a short time - 300 ms
    # 2) at the mirror for 700 ms
    # 3) back at the on-ramp for the rest of the trial
    # TODO: replace this fake gaze sample with the actual average from the experiment
    return np.concatenate([np.zeros(int(0.3/simulation_params["dt"])), np.ones(int(0.7/simulation_params["dt"])),
                           np.zeros(int((simulation_params["duration"]-1.0)/simulation_params["dt"])+1)])

def fit_model_by_condition(subj_idx=0, loss="vincent"):
    # HACK: for now the model is fitted conditional on one "average" gaze sample - we assume that the same
    #  parameter values can predict response of the model to individual gaze samples.
    simulation_params = {"dt": 0.01, "duration": 4.0}
    gaze_sample = helper.get_mean_gaze_rate(simulation_params)

    # model = models.ModelDynamicDriftCollapsingBounds()
    # model = models.ModelGazeDependent(gaze_sample)
    model = models.ModelGazeDependentBoundGeneralizedGap(gaze_sample)

    exp_data = pd.read_csv("measures.csv")
    exp_data = exp_data[exp_data.RT < 4.0]
    subjects = exp_data.subj_id.unique()

    if subj_idx == "all":
        subj_id = "all"
        subj_data = exp_data
        loss = loss_functions.LossWLSVincent if loss == "vincent" else pyddm.LossRobustBIC
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = loss_functions.LossWLS

    output_directory = "model_fit_results/gaze_dependent_bound_generalized_gap_model"

    file_name = "subj_%s_parameters_fitted.csv" % (str(subj_id))
    if not os.path.isfile(os.path.join(output_directory, file_name)):
        helper.write_to_csv(output_directory, file_name, ["subj_id", "loss"] + model.param_names, write_mode="w")

    print(subj_id)

    training_data = subj_data
    print("len(training_data): " + str(len(training_data)))

    fitted_model = helper.fit_model(model.model, training_data, loss)
    helper.write_to_csv(output_directory, file_name,
                        [subj_id, fitted_model.get_fit_result().value()]
                        + [float(param) for param in fitted_model.get_model_parameters()])

    return fitted_model


fitted_model = fit_model_by_condition(subj_idx="all", loss="robustBIC")


