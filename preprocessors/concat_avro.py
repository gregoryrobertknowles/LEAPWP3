import os
import json
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from avro.schema import parse as parse_schema


class AvroMerger:
    def __init__(self, avro_file_paths):
        self.avro_file_paths = avro_file_paths
        self.schema = None

    def load_raw_avro(self, filepath):
        with DataFileReader(open(filepath, "rb"), DatumReader()) as reader:
            schema_json = reader.meta.get("avro.schema").decode("utf-8")
            schema = parse_schema(schema_json)
            data = next(reader)
        return data, schema

    def merge_raw_avro(self, avro_data_list):
        base = json.loads(json.dumps(avro_data_list[0]))

        for sensor_key, sensor_data in base["rawData"].items():
            if sensor_key == "systolicPeaks":
                merged_peaks = []
                for data in avro_data_list:
                    merged_peaks.extend(
                        data["rawData"]["systolicPeaks"]["peaksTimeNanos"]
                    )
                sensor_data["peaksTimeNanos"] = merged_peaks
            else:
                merged_arrays = {}
                for k, v in sensor_data.items():
                    if isinstance(v, list):
                        merged_arrays[k] = []
                timestamp_starts = []

                for data in avro_data_list:
                    current = data["rawData"][sensor_key]
                    if "timestampStart" in current:
                        timestamp_starts.append(current["timestampStart"])
                    for k, v in current.items():
                        if isinstance(v, list):
                            merged_arrays[k].extend(v)

                if timestamp_starts:
                    sensor_data["timestampStart"] = min(timestamp_starts)

                for k, arr in merged_arrays.items():
                    sensor_data[k] = arr

        return base

    def merge_avro_files(self, output_file):
        raw_data_list = []
        for f in self.avro_file_paths:
            data, schema_f = self.load_raw_avro(f)
            raw_data_list.append(data)
            if self.schema is None:
                self.schema = schema_f
            else:
                if schema_f.to_json() != self.schema.to_json():
                    raise ValueError(f"Schema mismatch in file {f}")

        merged_data = self.merge_raw_avro(raw_data_list)

        with open(output_file, "wb") as out_f:
            with DataFileWriter(out_f, DatumWriter(), self.schema) as writer:
                writer.append(merged_data)

        print(f"[OK] Merged Avro file written to: {output_file}")


if __name__ == "__main__":
    input_dir = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\avrodebug"
    output_file = r"C:\Users\GRK\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\avrodebug\combined\merged_full.avro"

    avro_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith(".avro")
    ]
    if not avro_files:
        raise RuntimeError("No .avro files found in input directory")

    merger = AvroMerger(avro_files)
    merger.merge_avro_files(output_file)
