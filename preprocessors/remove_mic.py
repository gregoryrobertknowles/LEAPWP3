# AI written code to remove microphone data from OE files and verify integrity of other sensor data.

# Author: Claude

# Need to verify that this works correctly and that all other sensor data is preserved by actually reading the output file and graphing the data.

import struct
import os
import datetime
from collections import defaultdict


def remove_microphone_data(input_filename, output_filename):
    """
    Remove microphone data from an OE file by copying all non-microphone packets.
    Verifies that all other sensor data has been preserved.

    Args:
        input_filename: Path to input .oe file
        output_filename: Path to output .oe file (without microphone data)
    """
    FILE_HEADER_FORMAT = "<HQ"
    FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FORMAT)
    MICROPHONE_SID = 2  # Sensor ID for microphone

    # Track packets by sensor ID for verification
    input_packets = defaultdict(int)
    output_packets = defaultdict(int)

    # First pass: read and write the file
    with open(input_filename, "rb") as fin, open(output_filename, "wb") as fout:
        # Copy file header
        header_data = fin.read(FILE_HEADER_SIZE)
        fout.write(header_data)

        version, timestamp = struct.unpack(FILE_HEADER_FORMAT, header_data)
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(input_filename)}")
        print(
            f"Version: {version}, Timestamp: {datetime.datetime.fromtimestamp(timestamp)}"
        )
        print(f"{'='*60}\n")

        # Process each packet
        while True:
            # Read packet header
            packet_header = fin.read(10)
            if len(packet_header) < 10:
                break

            sid, size, time = struct.unpack("<BBQ", packet_header)

            # Read packet data
            data = fin.read(size)
            if len(data) < size:
                break

            # Count input packets
            input_packets[sid] += 1

            # Write packet only if it's NOT microphone data
            if sid != MICROPHONE_SID:
                fout.write(packet_header)
                fout.write(data)
                output_packets[sid] += 1

    # Second pass: verify by reading the output file
    print("Verifying output file...\n")
    verification_packets = defaultdict(int)

    with open(output_filename, "rb") as f:
        # Skip file header
        f.read(FILE_HEADER_SIZE)

        # Count packets in output file
        while True:
            packet_header = f.read(10)
            if len(packet_header) < 10:
                break

            sid, size, time = struct.unpack("<BBQ", packet_header)
            data = f.read(size)
            if len(data) < size:
                break

            verification_packets[sid] += 1

    # Print results
    sensor_names = {
        0: "IMU",
        1: "Barometer",
        2: "Microphone",
        4: "PPG",
        7: "Bone Accelerometer",
    }

    print("INPUT FILE PACKETS:")
    print("-" * 60)
    for sid in sorted(input_packets.keys()):
        name = sensor_names.get(sid, f"Unknown (SID {sid})")
        print(f"  {name:20s} (SID {sid}): {input_packets[sid]:,} packets")
    print(f"\n  {'TOTAL':20s}        : {sum(input_packets.values()):,} packets")

    print("\n\nOUTPUT FILE PACKETS:")
    print("-" * 60)
    for sid in sorted(output_packets.keys()):
        name = sensor_names.get(sid, f"Unknown (SID {sid})")
        print(f"  {name:20s} (SID {sid}): {output_packets[sid]:,} packets")
    print(f"\n  {'TOTAL':20s}        : {sum(output_packets.values()):,} packets")

    # Verification
    print("\n\nVERIFICATION:")
    print("-" * 60)

    all_preserved = True
    for sid in input_packets.keys():
        if sid == MICROPHONE_SID:
            if sid in output_packets:
                print(f"  ❌ ERROR: Microphone data still present in output!")
                all_preserved = False
            else:
                print(
                    f"  ✓ Microphone data successfully removed ({input_packets[sid]:,} packets)"
                )
        else:
            if output_packets[sid] == input_packets[sid] == verification_packets[sid]:
                name = sensor_names.get(sid, f"SID {sid}")
                print(f"  ✓ {name} data preserved: {input_packets[sid]:,} packets")
            else:
                name = sensor_names.get(sid, f"SID {sid}")
                print(f"  ❌ ERROR: {name} packet count mismatch!")
                print(
                    f"      Input: {input_packets[sid]}, Output: {output_packets[sid]}, Verified: {verification_packets[sid]}"
                )
                all_preserved = False

    # File size comparison
    input_size = os.path.getsize(input_filename)
    output_size = os.path.getsize(output_filename)
    size_reduction = input_size - output_size
    size_reduction_pct = (size_reduction / input_size) * 100

    print(f"\n\nFILE SIZE:")
    print("-" * 60)
    print(f"  Input file:  {input_size:,} bytes ({input_size / 1024 / 1024:.2f} MB)")
    print(f"  Output file: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")
    print(f"  Reduction:   {size_reduction:,} bytes ({size_reduction_pct:.1f}%)")

    print("\n" + "=" * 60)
    if all_preserved:
        print("✓ SUCCESS: All non-microphone data preserved!")
    else:
        print("❌ WARNING: Data integrity issues detected!")
    print("=" * 60 + "\n")

    return all_preserved


# Usage example
if __name__ == "__main__":
    success = remove_microphone_data(
        r"C:\Users\HCI\Documents\GRK2025\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\data_preprocessing\data\pilotp07\Pilot07L.oe",
        r"C:\Users\HCI\Documents\GRK2025\OneDrive - University of Bath\Christopher Clarke's files - Research\WP3\Data\data_preprocessing\data\pilotp07\Pilot07Lmicremoved.oe",
    )

    if success:
        print(f"✓ Output saved to: Pilot07Lmicremoved.oe")
