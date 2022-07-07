import numpy as np
import pandas as pd
from getpass import getuser
from scipy.signal import savgol_filter

process_data_path = r"C:\Users\{}\Documents\VT4_Velux\data\process_data.csv".format(getuser())

class DynamicsEstimator:
    def __init__(self, pitch=0.01, data_path=process_data_path):
        self.df = pd.read_csv(data_path, index_col=0)
        self.df = self.df[["sessionIDTag", "angle", "torque"]]
        self.df["id"] = self.df["sessionIDTag"]
        self.pitch = pitch
        self.df["depth"] = (pitch * self.df["angle"])/1000
        self.df = self.df.drop(columns=["angle", "sessionIDTag"])
        self.df_original = self.df

        for name, group in self.df.groupby("id"):
            if group["torque"].max() < 1.5 or group["torque"].max() > 5:
                self.df = self.df[self.df.id != name]
            elif 0.5 < group["torque"].min():
                self.df = self.df[self.df.id != name]
        self.df = self.df.reset_index(drop=True)

        self.df_sorted = pd.DataFrame()
        for name, group in self.df.groupby("id"):
            if group["depth"].max() <= 0.025:
                group["screw_length"] = [0.02]*len(group)
            elif 0.025 < group["depth"].max() <= 0.035:
                group["screw_length"] = [0.03]*len(group)
            elif 0.035 < group["depth"].max():
                group["screw_length"] = [0.05]*len(group)
            self.df_sorted = pd.concat([self.df_sorted, group])

        self.df_smoothed = pd.DataFrame()
        for name, group in self.df_sorted.groupby("id"):
            group = group.reset_index(drop=True)
            group["torque"] = savgol_filter(group["torque"], 15, 2)
            group["depth"] = (group["depth"]-np.min(group["depth"]))/(np.max(group["depth"])-np.min(group["depth"]))*group["screw_length"][0]
            self.df_smoothed = pd.concat([self.df_smoothed, group])

    def determine_friction(self, position, data):
        for row in data.itertuples():
            if row.depth >= position and row.Index > 0:
                index = row.Index
                depth, torque = [data.depth[index-1], data.depth[index]], [data.torque[index-1], data.torque[index]]
                friction = np.interp(position, depth, torque)
                break
        return friction

    def get_random_sample(self):
        groups = self.df_smoothed.groupby("id")
        sample = groups.get_group(np.random.choice(self.df_smoothed.id.unique())).reset_index(drop=True)
        return sample

"""
import matplotlib.pyplot as plt

dyn = DynamicsEstimator()

plt.figure(figsize=(5, 3), dpi=300)

for _ in range(5):
    sample = dyn.get_random_sample()
    plt.plot(sample["depth"], sample["torque"])
plt.grid()
plt.show()
"""