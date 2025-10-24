from .abstractprocessor import AbstractProcessor
import pandas as pd
import pytz


class RespRateProcessor(AbstractProcessor):
    # inherits def init

    def load(self):
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        skip_lines = 0
        for i, line in enumerate(lines):
            if line.strip() == "# EndOfHeader":
                skip_lines = i + 1
                break
        df = pd.read_csv(
            self.filepath, sep="\t", skiprows=skip_lines, header=None, usecols=[0, 1]
        )
        df.columns = ["timestamp_ms", "value"]
        # check timestamps for errors
        self.validate_timestamps(df)

        df["datetime"] = (
            pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            .dt.tz_convert("Europe/London")  # Convert UTC → local timezone
            .dt.tz_localize(None)  # Drop tz info for easier comparison
        )
        self.data = df
        return self.data


if __name__ == "__main__":

    filepath = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\28Jul Pilot\data_0080E127E7BD_2025-07-28_14-43-27.txt"
    # triallabels = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\trial_times_updated.csv"
    rrproc = RespRateProcessor(filepath)
    rrproc.load()
    print(rrproc.data)

    rrproc.data.to_csv(
        r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\28Jul Pilot\respiration_data.csv",
        index=False,
    )
