import struct
import matplotlib.pyplot as plt
import sys
import os


def parse_imubin(filepath):
    """Parses the custom 60-byte header and 36-byte samples from an .imubin file."""
    with open(filepath, 'rb') as f:
        # --- 1. Read Header (60 bytes) ---
        header_data = f.read(60)
        if len(header_data) < 60:
            raise ValueError("File too small to contain a valid header.")

        # Unpack format:
        # <   : little-endian
        # 4s  : Magic (4 bytes)
        # B   : Version (1 byte)
        # B   : WristLocation (1 byte)
        # B   : CrownOrientation (1 byte)
        # B   : Padding (1 byte)
        # 36s : RunID (36 bytes)
        # i   : WorkoutID (4 bytes)
        # i   : ChunkIndex (4 bytes)
        # i   : TotalChunks (4 bytes)
        # i   : SampleCount (4 bytes)
        header_format = '<4sBBBB36siiii'
        unpacked_header = struct.unpack(header_format, header_data)

        magic = unpacked_header[0]
        if magic != b'IMU\x00':
            raise ValueError(f"Invalid magic bytes! Found: {magic}")

        # Decode the UUID and strip the null padding
        run_id = unpacked_header[5].decode('ascii').rstrip('\x00')
        print(f"Successfully read header for Run ID: {run_id}")

        # --- 2. Read Samples ---
        # Unpack format:
        # d  : timestamp (Double / Float64, 8 bytes)
        # i  : workoutTime (Int32, 4 bytes)
        # 6f : ax, ay, az, gx, gy, gz (6 x Float32, 24 bytes)
        sample_format = '<di6f'
        sample_size = struct.calcsize(sample_format)  # Exactly 36 bytes

        times, ax, ay, az, gx, gy, gz = [], [], [], [], [], [], []

        # Read chunks of 36 bytes until we hit the end of the file
        while True:
            sample_data = f.read(sample_size)
            if len(sample_data) < sample_size:
                break  # End of file

            unpacked_sample = struct.unpack(sample_format, sample_data)

            times.append(unpacked_sample[0])
            # unpacked_sample[1] is workoutTime, skipping for the plot
            ax.append(unpacked_sample[2])
            ay.append(unpacked_sample[3])
            az.append(unpacked_sample[4])
            gx.append(unpacked_sample[5])
            gy.append(unpacked_sample[6])
            gz.append(unpacked_sample[7])

    return run_id, times, ax, ay, az, gx, gy, gz


def plot_imu_data(filepath, output_png):
    """Plots accelerometer and gyroscope data and saves to a PNG."""
    print(f"Reading file: {filepath}...")
    run_id, times, ax, ay, az, gx, gy, gz = parse_imubin(filepath)

    if not times:
        print("Error: No samples found in the file.")
        return

    # Normalize time array so the plot starts at 0.0 seconds
    start_time = times[0]
    relative_times = [t - start_time for t in times]

    # Create a figure with two stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Accelerometer Plot ---
    ax1.plot(relative_times, ax, label='X', color='#e63946', linewidth=1)
    ax1.plot(relative_times, ay, label='Y', color='#2a9d8f', linewidth=1)
    ax1.plot(relative_times, az, label='Z', color='#457b9d', linewidth=1)
    ax1.set_title(f'Accelerometer Data (Run: {run_id})')
    ax1.set_ylabel('Acceleration (g)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Gyroscope Plot ---
    ax2.plot(relative_times, gx, label='X', color='#e63946', linewidth=1)
    ax2.plot(relative_times, gy, label='Y', color='#2a9d8f', linewidth=1)
    ax2.plot(relative_times, gz, label='Z', color='#457b9d', linewidth=1)
    ax2.set_title('Gyroscope Data')
    ax2.set_ylabel('Rotation Rate (rad/s)')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Clean up layout and save
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Success! Plot saved to: {output_png}")


if __name__ == "__main__":
    # If running from command line: python plot_imu.py my_data.imubin
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = os.path.splitext(input_file)[0] + ".png"
        plot_imu_data(input_file, output_file)
    else:
        print("Usage: python plot_imu.py data/1_2026-04-29_58C74154-3526-46B2-8CDF-A61CC0AFE457.imubin")