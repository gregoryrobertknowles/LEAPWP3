import os
import json
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from avro.schema import parse as parse_schema
from datetime import datetime


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

    def trim_avro_by_time(self, input_file, output_file, start_time, end_time):
        """
        Trim an Avro file to keep only data between start_time and end_time.

        Args:
            input_file: Path to the input Avro file
            output_file: Path to write the trimmed Avro file
            start_time: Start time in microseconds
            end_time: End time in microseconds
        """
        print(f"\n[INFO] Trimming from {start_time} to {end_time}")

        # Load the Avro file
        data, schema = self.load_raw_avro(input_file)

        # Create a deep copy to avoid modifying the original
        trimmed_data = json.loads(json.dumps(data))

        # Process each sensor in rawData
        for sensor_key, sensor_data in trimmed_data["rawData"].items():
            if sensor_key == "systolicPeaks":
                # Filter peaks by time
                peaks = sensor_data.get("peaksTimeNanos", [])
                # Assuming peaks are in microseconds (based on the data inspection)
                filtered_peaks = [p for p in peaks if start_time <= p <= end_time]
                sensor_data["peaksTimeNanos"] = filtered_peaks
                print(f"  {sensor_key}: {len(peaks)} -> {len(filtered_peaks)} peaks")

            else:
                # For other sensors with timestampStart and arrays
                timestamp_start = sensor_data.get("timestampStart")

                if timestamp_start is None or timestamp_start == 0:
                    print(
                        f"  Warning: {sensor_key} has no valid timestampStart, skipping"
                    )
                    continue

                # Find all array fields
                array_fields = {
                    k: v for k, v in sensor_data.items() if isinstance(v, list)
                }

                if not array_fields:
                    continue

                # Get the length of arrays (assume all same length)
                original_length = len(next(iter(array_fields.values())))

                if original_length == 0:
                    print(f"  {sensor_key}: empty arrays, skipping")
                    continue

                # Get sampling frequency
                sampling_freq = sensor_data.get("samplingFrequency")

                if sampling_freq is None or sampling_freq == 0:
                    print(f"  Warning: {sensor_key} has no samplingFrequency, skipping")
                    continue

                # Calculate the interval between samples in microseconds
                interval_micros = int(1_000_000 / sampling_freq)

                # Calculate which indices to keep
                indices_to_keep = []
                for i in range(original_length):
                    timestamp = timestamp_start + (i * interval_micros)
                    if start_time <= timestamp <= end_time:
                        indices_to_keep.append(i)

                if not indices_to_keep:
                    print(f"  {sensor_key}: no samples in time range, clearing arrays")
                    # Clear all arrays
                    for field_name in array_fields.keys():
                        sensor_data[field_name] = []
                    continue

                # Trim all arrays
                for field_name, field_values in array_fields.items():
                    sensor_data[field_name] = [field_values[i] for i in indices_to_keep]

                # Update timestampStart to the first kept timestamp
                new_timestamp_start = timestamp_start + (
                    indices_to_keep[0] * interval_micros
                )
                sensor_data["timestampStart"] = new_timestamp_start

                print(
                    f"  {sensor_key}: {original_length} -> {len(indices_to_keep)} samples"
                )
                print(
                    f"    New timestampStart: {new_timestamp_start} ({datetime.fromtimestamp(new_timestamp_start / 1_000_000)})"
                )

        # Write trimmed data to output file
        with open(output_file, "wb") as out_f:
            with DataFileWriter(out_f, DatumWriter(), schema) as writer:
                writer.append(trimmed_data)

        print(f"[OK] Trimmed Avro file written to: {output_file}")

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
