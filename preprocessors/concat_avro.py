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

    def trim_avro_by_time(
        self, input_file, output_file, start_time_nanos, end_time_nanos
    ):
        """
        Trim an Avro file to keep only data between start_time_nanos and end_time_nanos.

        Args:
            input_file: Path to the input Avro file
            output_file: Path to write the trimmed Avro file
            start_time_nanos: Start time in nanoseconds
            end_time_nanos: End time in nanoseconds
        """
        # Load the Avro file
        data, schema = self.load_raw_avro(input_file)

        # Create a deep copy to avoid modifying the original
        trimmed_data = json.loads(json.dumps(data))

        # Process each sensor in rawData
        for sensor_key, sensor_data in trimmed_data["rawData"].items():
            if sensor_key == "systolicPeaks":
                # Filter peaks by time
                peaks = sensor_data.get("peaksTimeNanos", [])
                filtered_peaks = [
                    p for p in peaks if start_time_nanos <= p <= end_time_nanos
                ]
                sensor_data["peaksTimeNanos"] = filtered_peaks
                print(f"  {sensor_key}: {len(peaks)} -> {len(filtered_peaks)} peaks")

            else:
                # For other sensors with timestampStart and arrays
                timestamp_start = sensor_data.get("timestampStart")

                if timestamp_start is None:
                    print(f"  Warning: {sensor_key} has no timestampStart, skipping")
                    continue

                # Find all array fields
                array_fields = {
                    k: v for k, v in sensor_data.items() if isinstance(v, list)
                }

                if not array_fields:
                    continue

                # Assume all arrays have the same length
                original_length = len(next(iter(array_fields.values())))

                # Calculate indices to keep based on sampling rate or timestamps
                # If there's a "timestampNanos" array, use it directly
                if "timestampNanos" in array_fields:
                    timestamps = array_fields["timestampNanos"]
                    indices_to_keep = [
                        i
                        for i, ts in enumerate(timestamps)
                        if start_time_nanos <= ts <= end_time_nanos
                    ]
                else:
                    # Use timestampStart and calculate based on index
                    # Assuming uniform sampling, we need to know the sampling rate
                    # For now, let's assume there's enough context to calculate
                    indices_to_keep = []
                    for i in range(original_length):
                        # This assumes uniform sampling - you may need to adjust
                        # based on your actual data structure
                        estimated_time = timestamp_start + (
                            i * 1_000_000
                        )  # placeholder
                        if start_time_nanos <= estimated_time <= end_time_nanos:
                            indices_to_keep.append(i)

                # Trim all arrays
                for field_name, field_values in array_fields.items():
                    sensor_data[field_name] = [field_values[i] for i in indices_to_keep]

                # Update timestampStart if we have timestamps
                if indices_to_keep and "timestampNanos" in array_fields:
                    sensor_data["timestampStart"] = array_fields["timestampNanos"][
                        indices_to_keep[0]
                    ]

                print(
                    f"  {sensor_key}: {original_length} -> {len(indices_to_keep)} samples"
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
