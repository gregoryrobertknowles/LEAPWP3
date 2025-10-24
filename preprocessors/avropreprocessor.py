from avro.datafile import DataFileReader
from avro.io import DatumReader
import json
from .abstractprocessor import AbstractProcessor
import pandas as pd
import pytz


# arguably all these could be in the top level abstract?
class AvroProcessor(AbstractProcessor):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.avrodata = self.readavro(self.filepath)

    def readavro(self, filepath):
        reader = DataFileReader(open(self.filepath, "rb"), DatumReader())
        schema = json.loads(reader.meta.get("avro.schema").decode("utf-8"))
        return next(reader)

    # this can be removed as moved to the abstract class
    def printdata(self):
        return print(self.data.head())


class AvroAcc(AvroProcessor):
    def load(self):
        df = pd.DataFrame()
        acc = self.avrodata["rawData"]["accelerometer"]
        timestamp = [
            round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"]))
            for i in range(len(acc["x"]))
        ]
        # convert ADV counts in g
        delta_physical = (
            acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
        )
        delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
        x_g = [val * delta_physical / delta_digital for val in acc["x"]]
        y_g = [val * delta_physical / delta_digital for val in acc["y"]]
        z_g = [val * delta_physical / delta_digital for val in acc["z"]]
        df = pd.DataFrame({"x_g": x_g, "y_g": y_g, "z_g": z_g})

        df["unixtime_us"] = timestamp
        df["datetime"] = (
            pd.to_datetime(df["unixtime_us"], unit="us", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        self.data = df
        return self.data


class AvroGyro(AvroProcessor):
    """
    So the code provided by empatica doesn't actually produce gyro - i think that
    is probably on the Embrace plus plan - so need to confirm.

    But the class seems to work.
    """

    def load(self):
        df = pd.DataFrame()
        gyro = self.avrodata["rawData"]["gyroscope"]
        timestamp = [
            round(gyro["timestampStart"] + i * (1e6 / gyro["samplingFrequency"]))
            for i in range(len(gyro["x"]))
        ]
        df = pd.DataFrame(
            {"gyro_x": gyro["x"], "gyro_y": gyro["y"], "gyro_z": gyro["z"]}
        )
        df["unixtime_us"] = timestamp
        df["datetime"] = (
            pd.to_datetime(df["unixtime_us"], unit="us", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        self.data = df
        return self.data


class AvroEDA(AvroProcessor):
    def load(self):
        df = pd.DataFrame()
        eda = self.avrodata["rawData"]["eda"]
        timestamp = [
            round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"]))
            for i in range(len(eda["values"]))
        ]

        df["EDAvalues"] = eda["values"]

        df["unixtime_us"] = timestamp
        df["datetime"] = (
            pd.to_datetime(df["unixtime_us"], unit="us", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        self.data = df
        return self.data


class AvroTemperature(AvroProcessor):
    def load(self):
        df = pd.DataFrame()
        tmp = self.avrodata["rawData"]["temperature"]
        timestamp = [
            round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"]))
            for i in range(len(tmp["values"]))
        ]
        df["values"] = tmp["values"]
        df["unixtime_us"] = timestamp
        df["datetime"] = (
            pd.to_datetime(df["unixtime_us"], unit="us", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        self.data = df
        return self.data


class AvroBVP(AvroProcessor):
    def load(self):
        df = pd.DataFrame()

        bvp = self.avrodata["rawData"]["bvp"]
        timestamp = [
            round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"]))
            for i in range(len(bvp["values"]))
        ]
        df["unixtime_us"] = timestamp
        df["datetime"] = (
            pd.to_datetime(df["unixtime_us"], unit="us", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        self.data = df
        return self.data


class AvroSystolic(AvroProcessor):
    def load(self):
        df = pd.DataFrame()
        sps = self.avrodata["rawData"]["systolicPeaks"]
        # note that systolic peaks is an array of timestamps where those peaks occur - therefore not a separate value
        # however these will need to be process, we will have a unixtime col, and a datetimecol.
        df["unixtime_ns"] = sps["peaksTimeNanos"]
        df["datetime"] = (
            pd.to_datetime(df["unixtime_ns"], unit="ns", utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )

        # I haven't converted to datetime objects

        self.data = df
        return self.data


# lets quickly test this before doing the subsequent classes, and then the rest should follow pretty easily

if __name__ == "__main__":

    # path = (
    #    r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\28Jul Pilot/1-1-JUL28TEST_1753710800.avro"
    # )
    path = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\1-1-TESTING01_1753354415.avro"
    triallabels = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\testing\trial_times_updated.csv"

    acc = AvroAcc(path, triallabels)
    eda = AvroEDA(path, triallabels)
    tmp = AvroTemperature(path, triallabels)
    bvp = AvroBVP(path, triallabels)
    sys = AvroSystolic(path, triallabels)

    acc.load()
    eda.load()
    tmp.load()
    bvp.load()
    sys.load()

    acc.printdata()
    eda.printdata()
    tmp.printdata()
    bvp.printdata()
    sys.printdata()

    # avprocacc.data.to_csv("testavroacc.csv")
