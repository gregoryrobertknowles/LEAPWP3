from .abstractprocessor import AbstractProcessor
from .oeparser import SensorDataset
import pandas as pd


class OEAccPreprocessor(AbstractProcessor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {"imu": {}, "ppg": {}}

    def load(self):
        ds = SensorDataset(self.filepath)
        self.data["imu"]["acc"] = ds.imu.acc
        self.data["ppg"] = ds.ppg
        return self.data


# add in more classes for other sensors when needed


if __name__ == "__main__":
    # Example usage
    oepp = OEAccPreprocessor(r"syncingdata\rightOE22Aug1057.oe")
    data = oepp.load()
    print(data["imu"]["acc"].head())
