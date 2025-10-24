from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


class TimeSeriesPlot:
    def __init__(self, datetime, values, title="Time Series Plot"):
        self.datetime = datetime
        self.values = values
        self.title = title
        self.span = None
        self.line2 = None

    def plot(self):
        # Convert datetime to Matplotlib's numeric date format
        x = mdates.date2num(np.asarray(self.datetime))
        y = np.asarray(self.values)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # Format x-axis as dates
        ax1.xaxis_date()
        ax2.xaxis_date()
        fig.autofmt_xdate()

        # Plot full signal
        ax1.plot_date(x, y, "-", label="Signal", color="blue")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"{self.title} (select a region on the top plot)")
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(x[0], x[-1])

        # Bottom detailed view
        (line2,) = ax2.plot_date(x, y, "-", color="tab:blue")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        self.line2 = line2

        def onselect(xmin, xmax):
            indmin, indmax = np.searchsorted(x, (xmin, xmax))
            indmax = min(len(x) - 1, indmax)

            region_x = x[indmin:indmax]
            region_y = y[indmin:indmax]

            if len(region_x) >= 2:
                self.line2.set_data(region_x, region_y)
                ax2.set_xlim(region_x[0], region_x[-1])
                ymin, ymax = region_y.min(), region_y.max()
                dy = (ymax - ymin) * 0.05 or 0.5
                ax2.set_ylim(ymin - dy, ymax + dy)
                fig.canvas.draw_idle()

        self.span = SpanSelector(
            ax1,
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True,
        )

        plt.show()
