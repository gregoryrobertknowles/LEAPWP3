from .abstractprocessor import AbstractProcessor
import pandas as pd


class TemperaturePreProcessor(AbstractProcessor):
    def load(self):
        with open(self.filepath, "r") as f:
            lines = f.readlines()
        skip_lines = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Date/Time,Unit,Value"):
                print("Found header line, skipping to next line")
                skip_lines = i + 1
                break
        if skip_lines == 0:
            raise ValueError("No header line found in the file.")

        df = pd.read_csv(self.filepath, skiprows=skip_lines, header=None)

        df.columns = ["Date/Time", "Unit", "Value"]
        df["datetime"] = pd.to_datetime(df["Date/Time"], format="%d/%m/%y %H:%M:%S")
        df = df.drop(columns=["Date/Time"])

        self.data = df
        return self.data


if __name__ == "__main__":
    path = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\28Jul Pilot\temperature30000000820F3941_072825.csv"
    trial_labels = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\trial_times_updated.csv"
    tproc = TemperaturePreProcessor(path)
    tproc.load()
    print(tproc.data.head())
    # tproc.data.to_csv(
    #    r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\28Jul Pilot\temperature_data.csv",
    #    index=False,
    # )
