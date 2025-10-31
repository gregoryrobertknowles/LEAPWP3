from .abstractprocessor import AbstractProcessor
from .oeparser import SensorDataset
import pandas as pd


class OEPreprocessor(AbstractProcessor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}

    def load(self):
        ds = SensorDataset(self.filepath)
        self.data = {
            "imu": {
                "acc": ds.imu.acc,
                "gyro": ds.imu.gyro,
            },
            "ppg": {
                "red": ds.ppg.red,
                "ir": ds.ppg.ir,
                "green": ds.ppg.green,
            },
        }

        return self.data


# add in more classes for other sensors when needed


if __name__ == "__main__":
    # Example usage
    oepp = OEPreprocessor(r"syncingdata\rightOE22Aug1057.oe")
    data = oepp.load()
    print(data["imu"]["acc"].head())
