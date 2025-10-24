import os

from preprocessors.edfpreprocessor import EDFPreprocessorAcc as edfaccpp
from preprocessors.edfpreprocessor import EDFPreprocessorECG as edfecgpp
from preprocessors.avropreprocessor import AvroAcc as avroaccpp
from preprocessors.oepreprocessor import OEAccPreprocessor as oeaccpp
from preprocessors.concat_avro import AvroMerger

from sync import synchotron

from plotters.plotters import TimeSeriesPlot

if __name__ == "__main__":

    data_dir_path = r"C:\Users\HCI\Documents\GRK2025\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\data_preprocessing\data"
    participant_id = "P05"
    path = os.path.join(data_dir_path, participant_id)

    OELpath = os.path.join(path, "Pilot05L.oe")
    OERpath = os.path.join(path, "Pilot05R.oe")

    avro_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".avro"):
                avro_files.append(os.path.join(root, file))

    # avromerger = AvroMerger(avro_files)
    # merged_avro_output_path = os.path.join(path, f"{participant_id}_merged.avro")
    # avromerger.merge_avro_files(merged_avro_output_path)
    # AVROpath = merged_avro_output_path

    EDFpath = os.path.join(path, "P05.edf")

    # continue for others but lets just plot for now

    oedataL = oeaccpp(OELpath).load()
    # oedataR = oeaccpp(OERpath).load()
    edfaccdata = edfaccpp(EDFpath).load()
    edfecgdata = edfecgpp(EDFpath).load()
    # avrodata = avroaccpp(AVROpath).load()
    print("Data loaded")

    TimeSeriesPlot(edfecgdata["datetime"], edfecgdata["ECG"]).plot()
    # TimeSeriesPlot(
    #   oedataL["datetime"], oedataL["ppg"]["red"], title="OE-L PPG Red Channel"
    # ).plot()
