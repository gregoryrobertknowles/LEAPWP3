import struct
import os
import datetime
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Audio
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt, resample, stft, istft
from sklearn.decomposition import PCA


LABELS = {
    "imu": [
        "acc.x",
        "acc.y",
        "acc.z",
        "gyro.x",
        "gyro.y",
        "gyro.z",
        "mag.x",
        "mag.y",
        "mag.z",
    ],
    "barometer": ["barometer.temperature", "barometer.pressure"],
    "ppg": ["ppg.red", "ppg.ir", "ppg.green", "ppg.ambient"],
    "bone_acc": ["bone_acc.x", "bone_acc.y", "bone_acc.z"],
}

COLORS = {"ppg": ["red", "darkred", "green", "gray"]}


class SensorAccessor:
    def __init__(self, df: pd.DataFrame, labels: list):
        self._df = df
        self._data = {}
        groups = defaultdict(list)

        for label in labels:
            parts = label.split(".")
            if len(parts) == 2:
                group, field = parts
                if label in df:
                    groups[group].append(label)
            else:
                if label in df:
                    self._data[label] = df[label]

        for group, columns in groups.items():
            short_names = [label.split(".")[1] for label in columns]
            subdf = df[columns].copy()
            subdf.columns = short_names
            self._data[group] = subdf

        self._full_df = pd.concat(self._data.values(), axis=1) if self._data else df

    def __getitem__(self, key):
        return self._data.get(key, None)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        if hasattr(self._full_df, name):
            return getattr(self._full_df, name)
        raise AttributeError(f"'SensorAccessor' object has no attribute '{name}'")

    def __repr__(self):
        return repr(self._full_df)

    def __getattr__df__(self):
        return self._full_df

    def to_dataframe(self):
        return self._full_df


class SensorDataset:
    SENSOR_SID = {
        "imu": 0,
        "barometer": 1,
        "microphone": 2,
        "ppg": 4,
        "bone_acc": 7,
    }

    def __init__(self, filename):
        self.filename = filename
        self.data = defaultdict(list)
        self.audio_stereo = None
        self.bone_sound = None
        self.df = pd.DataFrame()

        self.imu = SensorAccessor(pd.DataFrame(columns=LABELS["imu"]), LABELS["imu"])
        self.barometer = SensorAccessor(
            pd.DataFrame(columns=LABELS["barometer"]), LABELS["barometer"]
        )
        self.ppg = SensorAccessor(pd.DataFrame(columns=LABELS["ppg"]), LABELS["ppg"])
        self.bone_acc = SensorAccessor(
            pd.DataFrame(columns=LABELS["bone_acc"]), LABELS["bone_acc"]
        )

        self.parse()
        self._build_accessors()

    def parse(self):
        FILE_HEADER_FORMAT = "<HQ"
        FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FORMAT)
        mic_samples = []
        _sid = None

        with open(self.filename, "rb") as f:
            # print(struct.unpack(FILE_HEADER_FORMAT, f.read(FILE_HEADER_SIZE)))
            version, timestamp = struct.unpack(
                FILE_HEADER_FORMAT, f.read(FILE_HEADER_SIZE)
            )
            print(
                f"📂 File: {os.path.basename(self.filename)}, Version: {version}, Timestamp: {datetime.datetime.fromtimestamp(timestamp)}"
            )

            while True:
                header = f.read(10)
                if len(header) < 10:
                    break
                sid, size, time = struct.unpack("<BBQ", header)
                if size > 192 or sid > 7:
                    if _sid != None and _sid in self.data.keys():
                        self.data[_sid].pop()
                    break

                _sid = sid

                data = f.read(size)
                if len(data) < size:
                    break
                timestamp_s = time / 1e6

                try:
                    if sid == 0:
                        values = struct.unpack("<9f", data)
                        self.data[sid].append((timestamp_s, values))
                    elif sid == 1:
                        values = struct.unpack("<2f", data)
                        self.data[sid].append((timestamp_s, values))
                    elif sid == 2:
                        samples = struct.unpack("<96h", data)
                        # self.data[sid].append((timestamp_s, samples))
                        mic_samples.extend(samples)
                    elif sid == 4:
                        values = struct.unpack("<4I", data)
                        self.data[sid].append((timestamp_s, values))
                    elif sid == 7:
                        self.data[sid].append((timestamp_s, data))
                    else:
                        continue
                except struct.error:
                    print(f"❌ Could not parse SID {sid} at timestamp {timestamp_s}")

        if mic_samples:
            mic_array = np.array(mic_samples, dtype=np.int16)
            self.audio_stereo = np.column_stack(
                (mic_array[1::2], mic_array[0::2])
            )  # [inner, outer] correctly

        if len(self.data[7]) > 0:
            all_samples = []
            sample_counts = []

            for _, d in self.data[7]:
                samples_per_packet = len(d) // struct.calcsize("<3h")
                sample_counts.append(samples_per_packet)

                for i in range(samples_per_packet):
                    offset = i * struct.calcsize("<3h")
                    sample = struct.unpack(
                        "<3h", d[offset : offset + struct.calcsize("<3h")]
                    )
                    all_samples.append(sample)

            int16_arrays = np.array(all_samples)
            total_samples = len(int16_arrays)

            detailed_times = []

            for i in range(len(self.data[7]) - 1):
                current_time = self.data[7][i][0]
                next_time = self.data[7][i + 1][0]
                samples_in_packet = sample_counts[i]
                if samples_in_packet > 0:
                    time_diff = (next_time - current_time) / samples_in_packet
                    detailed_times.extend(
                        [current_time + j * time_diff for j in range(samples_in_packet)]
                    )
                else:
                    detailed_times.extend([current_time] * samples_in_packet)

            if len(self.data[7]) > 0:
                last_time = self.data[7][-1][0]
                last_samples = sample_counts[-1]
                if len(self.data[7]) > 1:
                    last_diff = self.data[7][-1][0] - self.data[7][-2][0]
                    if last_samples > 0:
                        detailed_times.extend(
                            [
                                last_time + j * (last_diff / last_samples)
                                for j in range(last_samples)
                            ]
                        )
                    else:
                        detailed_times.extend([last_time] * last_samples)

                else:
                    detailed_times.extend(
                        [last_time + j * 0.001 for j in range(last_samples)]
                    )

            self.data[7] = list(zip(detailed_times, all_samples))

    def _build_accessors(self):
        dfs = []
        for name, sid in self.SENSOR_SID.items():
            labels = LABELS.get(name, [f"val{i}" for i in range(0)])
            if sid in self.data and self.data[sid]:
                times, values = zip(*self.data[sid])
                df = pd.DataFrame(values, columns=labels)
                df["timestamp"] = times
                df.set_index("timestamp", inplace=True)
                df = df[~df.index.duplicated(keep="first")]  # Drop duplicate timestamps
                dfs.append(df)
            else:
                df = pd.DataFrame(columns=labels)

            setattr(self, name, SensorAccessor(df, labels))

        if dfs:
            # Determine a common time index by taking the union of all indices and sorting
            common_index = pd.Index([])
            for df in dfs:
                common_index = common_index.union(df.index)
            common_index = common_index.sort_values()

            # Reindex each dataframe to the common index and then concatenate
            reindexed_dfs = [df.reindex(common_index) for df in dfs]
            self.df = pd.concat(reindexed_dfs, axis=1)
        else:
            self.df = pd.DataFrame()

    def get_dataframe(self):
        return self.df

    def save_csv(self, path):
        if not self.df.empty:
            self.df.to_csv(path)

    def play_audio(self, sampling_rate=48000):
        if self.audio_stereo is None:
            print("❌ No microphone data available.")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, sampling_rate, self.audio_stereo)
            display(Audio(tmp.name))

    def process_bone(
        self,
        target_sampling_rate=16000,
        enable_noise_reduction=True,
        enable_equalization=True,
    ):
        if not self.data[7]:
            print("❌ No bone sound data available.")
            return

        # Helper functions for filtering
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            return b, a

        def band_pass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data, axis=0)
            return y

        time_stamps = np.array([item[0] for item in self.data[7]])
        bone_sound = np.array([item[1] for item in self.data[7]])

        # Based on the provided bone sound data
        original_samplerate = 1 / np.median(np.diff(time_stamps))

        # Apply band-pass filter to the bone sound
        lowcut_frequency = 150
        highcut_frequency = 400
        filtered_signal = band_pass_filter(
            bone_sound, lowcut_frequency, highcut_frequency, original_samplerate
        )

        # Perform PCA on the filtered signal
        pca = PCA(n_components=1)  # Reduce to 1 principal component
        pca_result = pca.fit_transform(filtered_signal)

        # Project the filtered signal along the first principal component
        processed_signal = pca_result[:, 0]

        # Resample the projected signal to target sampling rate
        num_samples_target = int(
            len(processed_signal) * (target_sampling_rate / original_samplerate)
        )
        resampled_signal = resample(processed_signal, num_samples_target)
        current_samplerate = target_sampling_rate

        if enable_noise_reduction:
            # Parameter for STFT
            n_fft = 2048
            hop_length = n_fft // 8

            # STFT of the signal
            frequencies, times_stft, Zxx = stft(
                resampled_signal,
                fs=current_samplerate,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
            )

            # Extract magnitudes and phases
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            # Automatic Noise Estimation
            frame_energy = np.sum(magnitude**2, axis=0)
            noise_segment_length = int(current_samplerate * 0.5 / (n_fft / hop_length))
            min_energy_index = np.argmin(frame_energy)
            start_index = max(0, min_energy_index - noise_segment_length // 2)
            end_index = min(magnitude.shape[1], start_index + noise_segment_length)
            if end_index - start_index < noise_segment_length:
                start_index = max(0, end_index - noise_segment_length)

            noise_estimation_segment = magnitude[:, start_index:end_index]
            noise_estimation = np.mean(noise_estimation_segment, axis=1, keepdims=True)

            # Spectral Subtraction
            magnitude_denoised = np.maximum(magnitude - noise_estimation, 0)

            # Reconstruction of the signal
            Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
            _, denoised_signal = istft(
                Zxx_denoised,
                fs=current_samplerate,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
            )

            # Ensure the denoised signal has the same length as the resampled signal
            processed_signal = denoised_signal[: len(resampled_signal)]

        if enable_equalization:
            # Apply peaking EQ filters at 110Hz and 220Hz with -4dB reduction
            Q = 2
            gain = 10 ** (-4 / 20)  # -4dB reduction
            w0_110 = 2 * np.pi * 110 / current_samplerate
            w0_220 = 2 * np.pi * 220 / current_samplerate
            alpha_110 = np.sin(w0_110) / (2 * Q)
            alpha_220 = np.sin(w0_220) / (2 * Q)

            # 110Hz filter coefficients
            a0_110 = 1 + alpha_110 / gain
            a1_110 = -2 * np.cos(w0_110)
            a2_110 = 1 - alpha_110 / gain
            b0_110 = 1 + alpha_110 * gain
            b1_110 = -2 * np.cos(w0_110)
            b2_110 = 1 - alpha_110 * gain

            # 220Hz filter coefficients
            a0_220 = 1 + alpha_220 / gain
            a1_220 = -2 * np.cos(w0_220)
            a2_220 = 1 - alpha_220 / gain
            b0_220 = 1 + alpha_220 * gain
            b1_220 = -2 * np.cos(w0_220)
            b2_220 = 1 - alpha_220 * gain

            # Apply filters
            equalized_signal = filtfilt(
                [b0_110, b1_110, b2_110], [a0_110, a1_110, a2_110], processed_signal
            )
            processed_signal = filtfilt(
                [b0_220, b1_220, b2_220], [a0_220, a1_220, a2_220], equalized_signal
            )

        # Scale the final processed signal for playback
        self.bonse_sound = np.int16(
            processed_signal / np.max(np.abs(processed_signal)) * 32767
        )

        # To play the scaled_processed_signal:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, target_sampling_rate, self.bonse_sound)
            display(Audio(tmp.name))

    def plot(self):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False)
        axes = axes.flatten()

        col_titles = [
            "Acc",
            "Gyro",
            "Mag",
            "PPG",
            "Temperature",
            "Pressure",
            "Bone_Acc",
            "Mic",
        ]
        for ax, title in zip(axes, col_titles):
            ax.set_title(title)

        if not self.df.empty:
            for axis in ["x", "y", "z"]:
                series = self.imu.acc.get(axis)
                if series is not None:
                    axes[0].plot(series.index, series, label=f"acc.{axis}")

            for axis in ["x", "y", "z"]:
                series = self.imu.gyro.get(axis)
                if series is not None:
                    axes[1].plot(series.index, series, label=f"gyro.{axis}")

            for axis in ["x", "y", "z"]:
                series = self.imu.mag.get(axis)
                if series is not None:
                    axes[2].plot(series.index, series, label=f"mag.{axis}")

            for label, color in zip(LABELS["ppg"], COLORS["ppg"]):
                series = getattr(self.ppg, label.split(".")[1], None)
                if series is not None:
                    axes[3].plot(series.index, series, label=label, color=color)

            temp = getattr(self.barometer, "temperature", None)
            if temp is not None:
                axes[4].plot(temp.index, temp, label="Temperature")

            pressure = getattr(self.barometer, "pressure", None)
            if pressure is not None:
                axes[5].plot(pressure.index, pressure, label="Pressure")

            for axis in ["x", "y", "z"]:
                series = self.bone_acc.get(axis)  # Use .get() for SensorAccessor
                if series is not None:
                    axes[6].plot(series.index, series, label=f"bone_acc.{axis}")

        if self.audio_stereo is not None:
            inner, outer = self.audio_stereo[:, 0], self.audio_stereo[:, 1]
            sample_rate = 48000
            duration = len(inner) / sample_rate
            times = np.linspace(0, duration, num=len(inner))
            axes[7].plot(times, inner, label="Mic Inner", alpha=0.7)
            axes[7].plot(times, outer, label="Mic Outer", alpha=0.7)

        for ax in axes:
            ax.grid(True)
            if ax.get_legend_handles_labels()[1]:
                ax.legend()

        fig.suptitle(f"Recording: {os.path.basename(self.filename)}", fontsize=14)
        plt.tight_layout()
        plt.show()


def load_recordings(file_paths):
    return [SensorDataset(path) for path in file_paths if os.path.isfile(path)]


def display_recordings(recordings):
    for ds in recordings:
        ds.plot()
        ds.play_audio()
        print("")


if __name__ == "__main__":
    print("hello")

    ds = SensorDataset(r"data/P07/P07L.oe")
    print(ds.barometer)
    ds = SensorDataset(r"data/P07/P07R.oe")
    print(ds.barometer)
