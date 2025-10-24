from .abstractprocessor import AbstractProcessor
import pyedflib
import numpy as np
import pandas as pd
from datetime import timedelta


class EDFPreprocessorAbstract(AbstractProcessor):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.edf_file = pyedflib.EdfReader(self.filepath)
        self.signal_labels = self.edf_file.getSignalLabels()
        self.start_datetime = self.edf_file.getStartdatetime()

    def read_edf(self, type):
        idx = self.signal_labels.index(type)
        edf_data = self.edf_file.readSignal(idx)
        fs = self.edf_file.getSampleFrequency(idx)  # idx of ECG
        # close the EDF file
        self.edf_file._close()
        df = pd.DataFrame({type: edf_data})
        timestamps = [
            self.start_datetime + timedelta(seconds=i / fs)
            for i in range(len(edf_data))
        ]
        df["datetime"] = pd.to_datetime(timestamps)
        return df


class EDFPreprocessorECG(EDFPreprocessorAbstract):
    def load(self):
        df = self.read_edf("ECG")
        self.data = df
        return self.data


class EDFPreprocessorAcc(EDFPreprocessorAbstract):
    def load(self):
        """
        Accelerometer_X
        Accelerometer_Y
        Accelerometer_Z
        """
        # this seems to work
        df_x = self.read_edf("Accelerometer_X")

        # these however return zeroes
        df_y = self.read_edf("Accelerometer_Y")
        df_z = self.read_edf("Accelerometer_Z")
        # clue: read -1, less than 336800 requested!!!

        df = pd.concat(
            [
                df_x["Accelerometer_X"],
                df_y["Accelerometer_Y"],
                df_z["Accelerometer_Z"],
                df_x["datetime"],
            ],
            axis=1,
        )
        df.columns = ["x_g", "y_g", "z_g", "datetime"]
        self.data = df
        return self.data


if __name__ == "__main__":
    # Example usage
    edf_file_path = r"28Jul Pilot\14-30-49.EDF"
    trial_labels = "trial_times_updated.csv"
    preprocessor = EDFPreprocessorAcc(edf_file_path)
    preprocessor.load()
    preprocessor.preview()

    # save to CSV
    preprocessor.data.to_csv("ecg_data_acc.csv", index=False)
