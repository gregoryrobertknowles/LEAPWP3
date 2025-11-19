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
            "barometer": {
                "temperature": ds.barometer.temperature,
                "pressure": ds.barometer.pressure,
            },
            "bone_acc": ds.bone_acc,
        }

        return self.data


# add in more classes for other sensors when needed


if __name__ == "__main__":
    # Example usage
    oepp = OEPreprocessor(r"data/pilotp05/Pilot05L.oe")
    data = oepp.load()
    print(data["imu"]["acc"].head())
  
