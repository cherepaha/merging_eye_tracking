import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import pyddm
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_constants():
    mirror_x_min = 175
    mirror_x_max = 535
    mirror_y_min = 90
    mirror_y_max = 360
    front_x_min = 545
    front_x_max = 1380
    front_y_min = 410
    front_y_max = 680

    return mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max, front_x_min, front_x_max, front_y_min, front_y_max

def resample_trajectory(trajectory, frequency=100):
    time_step = 1.0 / frequency
    t_regular = np.arange(trajectory.t.min(), trajectory.t.max(), time_step)
    eye_x_interp = np.interp(t_regular, trajectory.t.values, trajectory.eye_x.values)
    eye_y_interp = np.interp(t_regular, trajectory.t.values, trajectory.eye_y.values)
    traj_interp = pd.DataFrame([t_regular, eye_x_interp, eye_y_interp]).transpose()
    traj_interp.columns = ['t', 'eye_x', 'eye_y']

    return traj_interp


def get_psf_ci(data):
    # psf: psychometric function
    # ci: dataframe with confidence intervals for probability per condition
    d_conditions = np.sort(data.d.unique())

    psf = np.array([len(data[data.is_gap_accepted & (data.d == d)])
                    / len(data[data.d == d])
                    if len(data[(data.d == d)]) > 0 else np.NaN
                    for d in d_conditions])

    ci = pd.DataFrame(psf, columns=["p_go"], index=d_conditions)

    n = [len(data[(data.d == d)]) for d in d_conditions]
    ci["ci_l"] = ci["p_go"] - np.sqrt(psf * (1 - psf) / n)
    ci["ci_r"] = ci["p_go"] + np.sqrt(psf * (1 - psf) / n)

    return ci.reset_index().rename(columns={"index": "d"})


def get_mean_sem(data, var="RT", groupby_var="tta", n_cutoff=2, ci_95=False):
    mean = data.groupby(groupby_var)[var].mean()
    sem = data.groupby(groupby_var)[var].apply(lambda x: scipy.stats.sem(x, axis=None, ddof=0))
    if ci_95:
        sem *= 1.96

    n = data.groupby(groupby_var).size()
    data_mean_sem = pd.DataFrame({"mean": mean, "sem": sem, "n": n}, index=mean.index)
    data_mean_sem = data_mean_sem[data_mean_sem.n > n_cutoff]

    return data_mean_sem

