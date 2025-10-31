from preprocessors.edfpreprocessor import EDFPreprocessorAcc as edfaccpp
from preprocessors.avropreprocessor import AvroAcc as avroaccpp
from preprocessors.oepreprocessor import OEPreprocessor as oeaccpp

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class synchotron:
    def __init__(
        self,
        avrodata,
        edfdata,
        oedataL,
        oedataR,
        avro_window=None,
        edf_window=None,
        oeL_window=None,
        oeR_window=None,
    ):
        self.avroaccdata = avrodata.reset_index()
        self.edfaccdata = edfdata
        self.oedataL = oedataL["imu"]["acc"].reset_index()
        self.oedataR = oedataR["imu"]["acc"].reset_index()
        self.avroaccdata = self.zscore_axes(self.avroaccdata)
        self.edfaccdata = self.zscore_axes(self.edfaccdata)
        self.oedataL = self.zscore_axes(self.oedataL)
        self.oedataR = self.zscore_axes(self.oedataR)

        self.avroFs = self._calculate_sample_rate(self.avroaccdata)
        self.edfFs = self._calculate_sample_rate(self.edfaccdata)
        self.oeFsL = self._calculate_sample_rate(self.oedataL, OE=True)
        self.oeFsR = self._calculate_sample_rate(self.oedataR, OE=True)

        self.avro_window = self.extract_window(
            self.avroaccdata,
            pre_sec=1,
            post_sec=4,
            Fs=self.avroFs,
            time_col="datetime",
            manual_window=avro_window,
        )
        self.edf_window = self.extract_window(
            self.edfaccdata,
            pre_sec=1,
            post_sec=4,
            Fs=self.edfFs,
            time_col="datetime",
            manual_window=edf_window,
        )
        self.oe_windowL = self.extract_window(
            self.oedataL,
            pre_sec=1,
            post_sec=4,
            Fs=self.oeFsL,
            time_col="timestamp",
            manual_window=oeL_window,
        )
        self.oe_windowR = self.extract_window(
            self.oedataR,
            pre_sec=1,
            post_sec=4,
            Fs=self.oeFsR,
            time_col="timestamp",
            manual_window=oeR_window,
        )
        # ==============================================================================================================================

        # 22 Aug 25
        # maybe directly calculate the frequency from the data instead of hardcoding it✅
        # this will then allow us to select the first datpoint that is above a certain threshold
        # and then get a certain amount of data before and after that point

        # need to figure out the units of each device as well, some appear to have gravity removed, some do not
        # some are in g, some in m/s^2
        # one of the accelerations appears to be huge compared to the others, need to check the units

        # leaning towards zscoring each of the axes first, as this will remove any ofssets sucgh as gravity
        # it will also make the units comparable between devices
        # then we can calculate the magnitude of the acceleration vector and use that for peak finding

        # ==============================================================================================================================

        # 29 Aug 25 - add in syncing functionality
        # https://chatgpt.com/c/68b1a426-1ee0-8321-91ab-ebab868dcddc for reference and some additional suggestions
        # basically cross-correlate the signals to find the lag between them after windowing the data to the relevant section
        # (i.e. around the jump, 1 second before and 4 seconds after the value which is 75% of the max value in the data)
        # we then set the data in these windows to begin at 0 seconds and use cross-correlation to find the lag between them
        # Still to do:
        # When we window, we introduce an offset in the timestamps, so we need to account for that when calculating the overall lag
        # Need to plot the overall aligned data to check it looks correct, and then return it as dataframes with corrected timestamps

        # ==============================================================================================================================
        self.avrosync = None
        self.edfsync = None
        self.oesyncL = None
        self.oesyncR = None

    def _calculate_sample_rate(self, data, OE=False):
        if OE:
            timestamps = data["timestamp"].values
            # timestamps are in seconds for OE
            diffs = np.diff(timestamps)  # already in seconds
        else:
            timestamps = data["datetime"].values
            diffs = np.diff(timestamps) / np.timedelta64(1, "s")  # convert to seconds
        avg_diff = np.mean(diffs)
        sample_rate = 1 / avg_diff if avg_diff != 0 else None
        print(f"Calculated sample rate: {sample_rate} Hz")

        return sample_rate

    def zscore_axes(self, df: pd.DataFrame) -> pd.DataFrame:
        # This is a ChatGPT generated function, handles the awkward column names in the different data types
        df = df.copy()

        # 1) Resolve axis columns (exact or *_g)
        cols = {c.lower(): c for c in df.columns}

        def pick(names):
            for n in names:
                if n in cols:
                    return cols[n]
            # accept *_g variants
            for n in names:
                cand = n + "_g"
                if cand in cols:
                    return cols[cand]
            return None

        x_col = pick(["x"])
        y_col = pick(["y"])
        z_col = pick(["z"])
        if not all([x_col, y_col, z_col]):
            raise ValueError(f"Axis columns not found in {list(df.columns)}")

        # 2) Coerce numeric
        for c in [x_col, y_col, z_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 3) Safe z-score
        def zscore(s):
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - mu) / sd

        df["xz"] = zscore(df[x_col])
        df["yz"] = zscore(df[y_col])
        df["zz"] = zscore(df[z_col])

        # 4) Magnitude from z-scores (don’t subtract 9.81 here)
        df["abs_mag_z"] = np.sqrt(df["xz"] ** 2 + df["yz"] ** 2 + df["zz"] ** 2)
        # zscore the magnitude too

        df["abs_mag_z"] = zscore(df["abs_mag_z"])

        # 5) Normalize time → t_datetime (optional but handy)
        if "datetime" in df.columns:
            df["t_datetime"] = pd.to_datetime(
                df["datetime"], errors="coerce", utc=False
            )

        return df

    def extract_window(  # in here add manual window option
        self,
        df,
        threshold=None,
        pre_sec=1,
        post_sec=4,
        Fs=None,
        time_col=None,
        manual_window=None,
    ):
        # Determine sample rate and time column
        if Fs is None:
            if "datetime" in df.columns:
                Fs = self._calculate_sample_rate(df)
                time_col = "datetime"
            elif "timestamp" in df.columns:
                Fs = self._calculate_sample_rate(df, OE=True)
                time_col = "timestamp"
            else:
                raise ValueError("No suitable time column found.")
        if time_col is None:
            time_col = "datetime" if "datetime" in df.columns else "timestamp"

        # If manual window is provided, use it
        if manual_window is not None:
            start, end = manual_window

            # If the data uses datetime values (Avro, EDF)
            if time_col == "datetime":
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                mask = (df[time_col] >= start) & (df[time_col] <= end)
                windowed_df = df.loc[mask].reset_index(drop=True)
                print(f"Using manual datetime window: {start} → {end}")

            # If the data uses float timestamps (OE sensors)
            elif time_col == "timestamp":
                start = float(start)
                end = float(end)
                mask = (df[time_col] >= start) & (df[time_col] <= end)
                windowed_df = df.loc[mask].reset_index(drop=True)
                print(f"Using manual timestamp window: {start:.3f}s → {end:.3f}s")

            else:
                raise ValueError(f"Unknown time_col '{time_col}'")

            return windowed_df

        # Set threshold to 95% of max if not provided
        if threshold is None:
            threshold = 0.75 * df["abs_mag_z"].max()

        # Find first index above threshold
        idx = df[df["abs_mag_z"] >= threshold].index
        if len(idx) == 0:
            raise ValueError("No data points above 75% of max found in abs_mag_z.")
        first_idx = idx[0]

        # Calculate window indices
        pre_samples = int(pre_sec * Fs)
        post_samples = int(post_sec * Fs)
        start_idx = max(0, first_idx - pre_samples)
        end_idx = min(len(df), first_idx + post_samples)
        windowed_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(
            f"Extracted window from index {start_idx} to {end_idx} ({pre_sec}s before and {post_sec}s after first 95% max point)"
        )
        # print(windowed_df[[time_col, "abs_mag_z"]].head(10))
        return windowed_df

        # ----------------------------------------------------------------------

    # Helpers
    # ----------------------------------------------------------------------

    def _resample_to_uniform(self, t_rel, x, Fs):
        """Resample (t_rel, x) onto a 0-based uniform grid at Fs Hz."""
        t_rel = np.asarray(t_rel, dtype=float)
        x = np.asarray(x, dtype=float)
        if t_rel.size < 2:
            return np.array([]), np.array([])
        dur = float(t_rel[-1] - t_rel[0])
        if dur <= 0:
            # fallback by index
            N = x.size
            t_rel = np.arange(N) / float(Fs)
            dur = t_rel[-1]
        grid = np.arange(0.0, dur, 1.0 / float(Fs))
        if grid.size < 2:
            grid = np.linspace(
                0.0, max(dur, 1.0 / Fs), num=max(x.size, 2), endpoint=True
            )
        x_uni = np.interp(grid, t_rel, x)
        return x_uni, grid

    def _xcorr_align(self, ref_t_abs, ref_y, tgt_t_abs, tgt_y, Fs, max_lag_s=None):
        """
        Align target to reference using cross-correlation on resampled relative times.

        Args:
            ref_t_abs: absolute times (s since epoch or device boot)
            ref_y:     signal values
            tgt_t_abs: absolute times (s)
            tgt_y:     signal values
            Fs:        resampling frequency (Hz)
            max_lag_s: optional, restrict lag search

        Returns:
            dict with lag_seconds, corr_peak, epoch_shift_seconds
        """
        # 0-based relative times
        ref_t_rel = ref_t_abs - ref_t_abs[0]
        tgt_t_rel = tgt_t_abs - tgt_t_abs[0]

        # resample
        ref_seq, _ = self._resample_to_uniform(ref_t_rel, ref_y, Fs)
        tgt_seq, _ = self._resample_to_uniform(tgt_t_rel, tgt_y, Fs)

        if ref_seq.size == 0 or tgt_seq.size == 0:
            raise ValueError("Empty sequences after resampling.")

        # correlation
        corr = np.correlate(tgt_seq, ref_seq, mode="full")
        lags = np.arange(-len(ref_seq) + 1, len(tgt_seq))
        if max_lag_s is not None:
            L = int(abs(max_lag_s) * Fs)
            keep = (lags >= -L) & (lags <= L)
            corr, lags = corr[keep], lags[keep]
        k = lags[np.argmax(corr)]
        lag_seconds = -k / float(Fs)  # add to target (0-based) to align
        corr_peak = float(np.max(corr))

        # epoch shift: align absolute clocks
        epoch_shift_seconds = float((ref_t_abs[0] - tgt_t_abs[0]) + lag_seconds)

        return {
            "lag_seconds": lag_seconds,
            "epoch_shift_seconds": epoch_shift_seconds,
            "corr_peak": corr_peak,
            "Fs": Fs,
        }

    def sync_to_edf(self, Fs_target=None, max_lag_s=None, show=True):
        if Fs_target is None:
            Fs_target = float(self.edfFs or 100.0)

        edf_mag = self.edf_window["abs_mag_z"].to_numpy()
        avro_mag = self.avro_window["abs_mag_z"].to_numpy()
        oeL_mag = self.oe_windowL["abs_mag_z"].to_numpy()
        oeR_mag = self.oe_windowR["abs_mag_z"].to_numpy()

        edf_t_abs = (
            pd.to_datetime(self.edf_window["datetime"]).astype("int64") / 1e9
        ).to_numpy()
        avr_t_abs = (
            pd.to_datetime(self.avro_window["datetime"]).astype("int64") / 1e9
        ).to_numpy()
        oeL_t_abs = self.oe_windowL["timestamp"].to_numpy()
        oeR_t_abs = self.oe_windowR["timestamp"].to_numpy()

        self.avrosync = self._xcorr_align(
            edf_t_abs, edf_mag, avr_t_abs, avro_mag, Fs_target, max_lag_s
        )

        self.oesyncL = self._xcorr_align(
            edf_t_abs, edf_mag, oeL_t_abs, oeL_mag, Fs_target, max_lag_s
        )
        self.oesyncR = self._xcorr_align(
            edf_t_abs, edf_mag, oeR_t_abs, oeR_mag, Fs_target, max_lag_s
        )

        if show:
            plt.figure(figsize=(14, 6))

            # EDF reference
            ref_dt = pd.to_datetime(edf_t_abs, unit="s")
            plt.plot(
                ref_dt, edf_mag, label="EDF (ref)", linewidth=1.5, color="tab:orange"
            )

            # Avro aligned
            avr_dt = pd.to_datetime(
                avr_t_abs + self.avrosync["epoch_shift_seconds"], unit="s"
            )
            plt.plot(
                avr_dt,
                avro_mag,
                label="Avro (aligned)",
                alpha=0.9,
                linewidth=1.2,
                color="tab:blue",
            )

            # OE Left aligned
            oeL_dt = pd.to_datetime(
                oeL_t_abs + self.oesyncL["epoch_shift_seconds"], unit="s"
            )
            plt.plot(
                oeL_dt,
                oeL_mag,
                label="OE Left (aligned)",
                alpha=0.9,
                linewidth=1.0,
                color="tab:green",
            )

            # OE Right aligned
            oeR_dt = pd.to_datetime(
                oeR_t_abs + self.oesyncR["epoch_shift_seconds"], unit="s"
            )
            plt.plot(
                oeR_dt,
                oeR_mag,
                label="OE Right (aligned)",
                alpha=0.9,
                linewidth=1.0,
                color="tab:red",
            )

            plt.title(
                f"EDF vs Avro/OE-L/OE-R (aligned)\n"
                f"Avro epoch shift={self.avrosync['epoch_shift_seconds']:.3f}s | "
                f"OE-L epoch shift={self.oesyncL['epoch_shift_seconds']:.3f}s | "
                f"OE-R lag={self.oesyncR['epoch_shift_seconds']:.3f}s"
            )
            plt.xlabel("Wall-clock time")
            plt.ylabel("abs_mag_z (z-scored)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.avrosync, self.oesyncL, self.oesyncR

    def sync(self, show=False):
        self.avrosync, self.oesyncL, self.oesyncR = self.sync_to_edf(show=False)
        avro_shift = self.avrosync["epoch_shift_seconds"]
        self.avroaccdata["datetime"] = self.avroaccdata["datetime"] + pd.to_timedelta(
            avro_shift, unit="s"
        )

        oeL_shift = self.oesyncL["epoch_shift_seconds"]
        oeR_shift = self.oesyncR["epoch_shift_seconds"]

        self.oedataL["datetime"] = pd.to_datetime(
            self.oedataL["timestamp"] + oeL_shift, unit="s"
        )
        self.oedataR["datetime"] = pd.to_datetime(
            self.oedataR["timestamp"] + oeR_shift, unit="s"
        )
        if not show:
            plt.figure(figsize=(12, 6))
            plt.plot(
                self.edfaccdata["datetime"],
                self.edfaccdata["abs_mag_z"],
                label="EDF (original)",
                color="tab:orange",
            )
            plt.plot(
                self.avroaccdata["datetime"],
                self.avroaccdata["abs_mag_z"],
                label="Avro (shifted)",
                color="tab:blue",
            )

            plt.plot(
                self.oedataL["datetime"],
                self.oedataL["abs_mag_z"],
                label="OE Left (shifted)",
                color="tab:green",
            )
            plt.plot(
                self.oedataR["datetime"],
                self.oedataR["abs_mag_z"],
                label="OE Right (shifted)",
                color="tab:red",
            )

            plt.xlabel("Datetime")
            plt.ylabel("abs_mag_z (z-scored)")
            plt.title("Shifted against EDF (original)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return avro_shift, oeL_shift, oeR_shift

    def plot_unsynced(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

        axs[0].plot(
            self.avroaccdata["datetime"],
            self.avroaccdata["abs_mag_z"],
            label="Avro",
            color="tab:blue",
        )
        axs[0].set_ylabel("Avro\nAbs Z-scored Acc")
        axs[0].legend(loc="upper right")
        axs[0].set_xlabel("Avro datetime")

        axs[1].plot(
            self.edfaccdata["datetime"],
            self.edfaccdata["abs_mag_z"],
            label="EDF",
            color="tab:orange",
        )
        axs[1].set_ylabel("EDF\nAbs Z-scored Acc")
        axs[1].legend(loc="upper right")
        axs[1].set_xlabel("EDF datetime")

        axs[2].plot(
            self.oedataL["timestamp"],
            self.oedataL["abs_mag_z"],
            label="OE Left",
            color="tab:green",
        )
        axs[2].set_ylabel("OE Left\nAbs Z-scored Acc")
        axs[2].legend(loc="upper right")
        axs[2].set_xlabel("OE Left seconds since boot")

        axs[3].plot(
            self.oedataR["timestamp"],
            self.oedataR["abs_mag_z"],
            label="OE Right",
            color="tab:red",
        )
        axs[3].set_ylabel("OE Right\nAbs Z-scored Acc")
        axs[3].legend(loc="upper right")
        axs[3].set_xlabel("OE Right seconds since boot")

        fig.suptitle(
            "Absolute Z-scored Acceleration from Each Datatype (Independent X Axes)"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_windowed(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

        axs[0].plot(
            self.avro_window["datetime"],
            self.avro_window["abs_mag_z"],
            label="Avro (windowed)",
            color="tab:blue",
        )
        axs[0].set_ylabel("Avro\nAbs Z-scored Acc")
        axs[0].legend(loc="upper right")
        axs[0].set_xlabel("Avro datetime")

        axs[1].plot(
            self.edf_window["datetime"],
            self.edf_window["abs_mag_z"],
            label="EDF (windowed)",
            color="tab:orange",
        )
        axs[1].set_ylabel("EDF\nAbs Z-scored Acc")
        axs[1].legend(loc="upper right")
        axs[1].set_xlabel("EDF datetime")

        axs[2].plot(
            self.oe_windowL["timestamp"],
            self.oe_windowL["abs_mag_z"],
            label="OE Left (windowed)",
            color="tab:green",
        )
        axs[2].set_ylabel("OE Left\nAbs Z-scored Acc")
        axs[2].legend(loc="upper right")
        axs[2].set_xlabel("OE Left seconds since boot")

        axs[3].plot(
            self.oe_windowR["timestamp"],
            self.oe_windowR["abs_mag_z"],
            label="OE Right (windowed)",
            color="tab:red",
        )
        axs[3].set_ylabel("OE Right\nAbs Z-scored Acc")
        axs[3].legend(loc="upper right")
        axs[3].set_xlabel("OE Right seconds since boot")

        fig.suptitle(
            "Absolute Z-scored Acceleration from Each Datatype (Windowed, Independent X Axes)"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":

    # avrofilepath = r"syncingdata\1-1-JUL28TEST_1755855861.avro"
    avrofilepath = r"syncingdata\1-1-JUL28TEST_1755855882.avro"
    avroaccdata = avroaccpp(avrofilepath).load()

    # for this particular trial I accidentally did two sets of jumps
    # which are visible in the avro data
    # so here we will trim the data to just the second set
    avroaccdata = avroaccdata[
        (avroaccdata["datetime"] > "2025-08-22 10:56:00")
        & (avroaccdata["datetime"] < "2025-08-22 11:00:00")
    ]

    edffilepath = r"syncingdata\Bittiumfaros.EDF"
    edfaccdata = edfaccpp(edffilepath).load()
    oefilepathL = r"syncingdata\leftOE22Aug1057.oe"
    oedataL = oeaccpp(oefilepathL).load()
    oefilepathR = r"syncingdata\rightOE22Aug1057.oe"
    oedataR = oeaccpp(oefilepathR).load()
    syncer = synchotron(avroaccdata, edfaccdata, oedataL, oedataR)
    # syncer.plot_unsynced()
    # syncer.plot_windowed()
    # syncer.sync()
    avro_shift, oeL_shift, oeR_shift = syncer.sync()
