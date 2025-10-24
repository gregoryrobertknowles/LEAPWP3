from abc import ABC, abstractmethod
import pandas as pd


class AbstractProcessor(ABC):
    def __init__(self, filepath):
        self.filepath = filepath

        self.trial_labels = None

        self.data = None

    @abstractmethod
    def load(self):
        """load raw data into self.data"""
        pass

    def clip(
        self,
    ):  # this needs some work as the trials are getting labelled but out of order. I suspect that is because when we add the missing 'stop' items, it changes the order of the trials.
        """clip data between times"""
        if self.data is not None:
            intervals = []
            starts = self.trial_labels[
                self.trial_labels["Label"].str.startswith("Start")
            ]
            for _, start_row in starts.iterrows():
                label_name = start_row["Label"].replace("Start ", "").strip()
                stop_row = self.trial_labels[
                    self.trial_labels["Label"] == f"Stop {label_name}"
                ]

                if not stop_row.empty:
                    start_time = start_row["Time"]
                    stop_time = stop_row.iloc[0]["Time"]
                    intervals.append((start_time, stop_time, label_name))

            label_to_code = {
                label: i + 1
                for i, label in enumerate(sorted({lbl for _, _, lbl in intervals}))
            }

            self.data["label_code"] = 0
            for start, stop, label in intervals:
                mask = (self.data["datetime"] >= start) & (
                    self.data["datetime"] <= stop
                )
                self.data.loc[mask, "label_code"] = label_to_code[label]

            return self.data

    def validate_timestamps(self, df):
        """Validate timestamps in the DataFrame."""
        errors = []
        if not df["timestamp_ms"].is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing.")
        if df["timestamp_ms"].duplicated().any():
            errors.append("Timestamps contain duplicates.")
        if errors:
            raise ValueError(" ".join(errors))

    def preview(self):
        return print(self.data.head())
