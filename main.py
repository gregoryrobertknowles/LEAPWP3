import os

from preprocessors.edfpreprocessor import EDFPreprocessorAcc as edfaccpp
from preprocessors.edfpreprocessor import EDFPreprocessorECG as edfecgpp
from preprocessors.avropreprocessor import AvroAcc as avroaccpp
from preprocessors.oepreprocessor import OEPreprocessor as oepp
from preprocessors.concat_avro import AvroMerger

from sync import synchotron

import matplotlib.pyplot as plt
import numpy as np

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

    oedataL = oepp(OELpath).load()
    # oedataR = oeaccpp(OERpath).load()
    # edfaccdata = edfaccpp(EDFpath).load()
    # edfecgdata = edfecgpp(EDFpath).load()
    # avrodata = avroaccpp(AVROpath).load()
    print("Data loaded")

    # plot OE L PPG Red on its own for now, there is no datetime index so just use sample number
    oedataL_ppg_red = oedataL["ppg"]["red"]

    print(type(oedataL_ppg_red))
    print(oedataL_ppg_red.head())

    oedataL_ppg_red = oedataL["ppg"]["red"]
    if oedataL_ppg_red is None:
        raise RuntimeError(
            "OE L PPG Red data is None; check loader output and file paths"
        )

    # convert whatever the loader returned to a 1D numpy array
    if hasattr(oedataL_ppg_red, "values"):
        data_arr = np.asarray(oedataL_ppg_red.values)
    else:
        data_arr = np.asarray(oedataL_ppg_red)

    x = np.arange(data_arr.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, data_arr, label="OE L PPG Red", color="red")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{participant_id} OE L PPG Red Signal")
    ax.legend()
    ax.grid(True)
    plt.show()
    plt.show()
