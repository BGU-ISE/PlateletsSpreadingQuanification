import sys
import time
import os
import math
import cv2
import matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ToTimeSeries import ToTimeSeries as ts_main


class IntensityDistribution:
    """

    """
    def __init__(self):

        self.INTERVALS_LOWER_BOUND = 0
        self.INTERVALS_UPPER_BOUND = 257
        self.INTERVALS_SKIP_VALUE = 8
        self.CUT_FILE = 0
        self.FRAMES_IN_MEMORY = 2
        self.BIN_SIZE = 4
        self.DELTA_MAX_VALUE = 30
        self.DELTA_MIN_VALUE = -30
        self.DELTA_BINS_NUM = 61

        self.INTERVALS = np.arange(self.INTERVALS_LOWER_BOUND, self.INTERVALS_UPPER_BOUND, self.INTERVALS_SKIP_VALUE)
        self.DELTAS = np.linspace(self.DELTA_MIN_VALUE, self.DELTA_MAX_VALUE, num=self.DELTA_BINS_NUM)

        self.delta_distribution_per_interval = np.zeros((len(self.INTERVALS), len(self.DELTAS)))
        self.previous_bin_frame_averages = None

        # PLOTTING PARAMETERS
        self.MAX_COLOR = 0.05
        self.LABEL_FONT_SIZE = 15
        self.LABEL_OUT_OF_AXIS_PARAM = 15
        self.FONT_SIZE = 50
        self.EVERY_X_LABEL_TO_PRINT = 8
        self.EVERY_Y_LABEL_TO_PRINT = 4

    # def _in_which_interval(self, val, intervals):
    #     idx = np.where((intervals <= val))
    #     idx = idx[0][len(idx[0]) - 1]
    #     return idx
    #
    # def _which_delta(self, val, deltas):
    #     # array = np.asarray(deltas)
    #     idx = (np.abs(deltas - val)).argmin()
    #     return idx

    def _count_deltas(self, frame_count, buf_bins, bin_number):
        global previous_bin_frame_averages
        if previous_bin_frame_averages is not None:
            bin_ctr = 0
            while bin_ctr < bin_number:
                frame_ctr = 0
                while frame_ctr < frame_count - 1:
                    # getting the average previously calculated

                    bin_current_frame_average = previous_bin_frame_averages[bin_ctr]
                    bin_next_frame_average = np.average(buf_bins[bin_ctr][frame_ctr + 1])

                    # updating the average for the next set of frames
                    previous_bin_frame_averages[bin_ctr] = bin_next_frame_average

                    current_bin_avgs_delta = bin_next_frame_average - bin_current_frame_average
                    bin_current_interval = np.where((self.INTERVALS <= bin_current_frame_average))[0][
                        -1]  # _in_which_interval(bin_current_frame_average, INTERVALS)
                    bin_current_delta_index = (np.abs(
                        self.DELTAS - current_bin_avgs_delta)).argmin()  # _which_delta(current_bin_avgs_delta, DELTAS)
                    delta_distribution_per_interval[bin_current_interval][bin_current_delta_index] += 1

                    frame_ctr += 1

                bin_ctr += 1
        else:
            # if does not exist yet, create a new array
            previous_bin_frame_averages = np.zeros(bin_number)

            bin_ctr = 0
            while bin_ctr < bin_number:
                frame_ctr = 0
                while frame_ctr < frame_count - 1:
                    bin_current_frame_average = np.average(buf_bins[bin_ctr][frame_ctr])
                    bin_next_frame_average = np.average(buf_bins[bin_ctr][frame_ctr + 1])
                    previous_bin_frame_averages[bin_ctr] = bin_next_frame_average
                    current_bin_avgs_delta = bin_next_frame_average - bin_current_frame_average

                    bin_current_interval = np.where((self.INTERVALS <= bin_current_frame_average))[0][
                        -1]  # _in_which_interval(bin_current_frame_average, INTERVALS)
                    bin_current_delta_index = (np.abs(
                        self.DELTAS - current_bin_avgs_delta)).argmin()  # _which_delta(current_bin_avgs_delta, DELTAS)
                    delta_distribution_per_interval[bin_current_interval][bin_current_delta_index] += 1

                    frame_ctr += 1

                bin_ctr += 1

    def _normalizing_deltas(self):
        interval_ctr = 0
        while interval_ctr < len(self.INTERVALS) - 1:
            current_interval_sum = np.sum(delta_distribution_per_interval[interval_ctr])
            delta_distribution_per_interval[interval_ctr] = delta_distribution_per_interval[
                                                                interval_ctr] / current_interval_sum
            interval_ctr += 1

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={"shrink": 0.3}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels, fontsize=self.LABEL_FONT_SIZE, fontweight='bold')
        ax.set_yticklabels(row_labels, fontsize=self.LABEL_FONT_SIZE, fontweight='bold')

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        # for edge, spine in ax.spines.items():
        #     spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=["black", "white"],
                         threshold=None, **textkw):

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm((data[i, j])) > threshold)])
                text = im.axes.text(j, i, "", **kw)  # ORG: "{0:.4f}".format(deltas_distribution[i, j]), **kw)
                texts.append(text)

        return texts

    def analyze_file(self, file_name, **kwargs):
        global delta_distribution_per_interval

        # LAST DIMENSION MUST DIVIDE 256 WITHOUT A REMAINDER. (THE POWER OF 2)
        delta_distribution_per_interval = np.zeros((len(self.INTERVALS), len(self.DELTAS)))

        cap = cv2.VideoCapture("ForAnalyze/{0}".format(file_name))
        final_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret = True


        total_start = time.time()
        # print("starting video {0}, frames: {1}".format(file_name, int(final_frame_count)))
        fc = 0
        buf = np.zeros((self.FRAMES_IN_MEMORY, frame_height, frame_width, 3), np.dtype('uint8'))

        while fc < final_frame_count - self.CUT_FILE - (self.FRAMES_IN_MEMORY - 1) and ret:
            # loading the data according
            in_memory_frames_ctr = 0
            while in_memory_frames_ctr < self.FRAMES_IN_MEMORY:
                if np.sum(buf[self.FRAMES_IN_MEMORY - 1]) > 0:
                    temp = buf[self.FRAMES_IN_MEMORY - 1].copy()
                    # clearing memory
                    del buf

                    buf = np.zeros((self.FRAMES_IN_MEMORY, frame_height, frame_width, 3), np.dtype('uint8'))
                    buf[0] = temp.copy()
                    del temp
                else:
                    ret, frame = cap.read()
                    buf[in_memory_frames_ctr] = frame
                    fc += 1
                in_memory_frames_ctr += 1
            # fc += 1
            # converting the data to a single channel
            start = time.time()
            single_channel_buf = buf[:, :, :, :1].copy()
            # converting the data into bins
            ts = ts_main(bin_size=self.BIN_SIZE, y_size=self.BIN_SIZE, original_file=single_channel_buf,
                         frame_count=self.FRAMES_IN_MEMORY, single_frame_width=frame_width, single_frame_height=frame_height,
                         channel_amount=1)
            buf_in_bins = ts.into_time_series()
            number_of_bins = ts.number_of_bins
            # counting the deltas
            self._count_deltas(frame_count=len(buf), buf_bins=buf_in_bins, bin_number=number_of_bins)
            del single_channel_buf
            del buf_in_bins
            end = time.time()

            print("{0}: {1:.2f} seconds to count deltas".format(fc, end - start))
        del buf
        del frame
        del ret

        cap.release()
        total_end = time.time()
        if kwargs.get('prog_print'):
            prog_print = kwargs.get('prog_print')
        else:
            prog_print = False
        if prog_print:
            print("{0}: {1:.2f} minutes total".format(fc, (total_end - total_start) / 60))
        if kwargs.get('end_proc_alarm'):
            end_proc_alarm = kwargs.get('end_proc_alarm')
        else:
            end_proc_alarm = False
        if end_proc_alarm:
            os.system('afplay alarm.m4a')

        self._normalizing_deltas()

        # for figure presentation
        intervals_list = ["{0} >".format(inter) for inter in self.INTERVALS]
        deltas_list = self.DELTAS
        data = delta_distribution_per_interval[:]
        deltas_distribution = np.array(np.around(data, 6))

        fig, ax = plt.subplots(figsize=(2 ^ 16, 2 ^ 10))
        plt.title("EXP:{0}\nDelta distribution in Intervals\n Max Color data value = {1}".format(file_name, self.MAX_COLOR),
                  fontsize=40)
        im, cbar = self.heatmap(deltas_distribution, intervals_list, np.around(deltas_list, 1), ax=ax,
                           cmap="YlGnBu", cbarlabel="Delta Distribution/ Gray Value Interval", vmax=self.MAX_COLOR)
        # setting the x labels
        # plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
        every_x_nth = self.EVERY_X_LABEL_TO_PRINT
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_x_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.xaxis.get_ticklines()):
            if n % every_x_nth != 0:
                line.set_visible(False)
        # setting the y labels
        # plt.rc('ytick', labelsize=FONT_SIZE)
        every_y_nth = self.EVERY_Y_LABEL_TO_PRINT
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_y_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.yaxis.get_ticklines()):
            if n % every_y_nth != 0:
                line.set_visible(False)
        # ANNOTATE
        texts = self.annotate_heatmap(im, valfmt="{x:.3f}",
                                 ha="center", va="center", fontsize=2)

        ax.tick_params(direction='out', pad=self.LABEL_OUT_OF_AXIS_PARAM)
        plt.tight_layout()
        fig.savefig(
            'Heatmaps/{0}-heatmapMaxColor{1}__DeltaSpan{2}>>{3}.png'.format(file_name, self.MAX_COLOR, int(np.min(self.DELTAS)),
                                                                            int(np.max(self.DELTAS))), dpi=100)
        print('saved plot file: {}'.format(file_name))
        os.system('clear')

    def show_different_max_color(self, data, name, max_color, **kwargs):
        intervals_list = ["{0} >".format(inter) for inter in self.INTERVALS]
        deltas_list = self.DELTAS
        deltas_distribution = np.array(np.around(data, 6))

        fig, ax = plt.subplots(figsize=(2 ^ 16, 2 ^ 10))
        plt.title("EXP:{0}\nDelta distribution in Intervals\n Max Color data value = {1}".format(name, max_color),
                  fontsize=40)
        im, cbar = self.heatmap(deltas_distribution, intervals_list, np.around(deltas_list, 1), ax=ax,
                           cmap="YlGnBu", cbarlabel="Delta Distribution/ Gray Value Interval", vmax=max_color)
        # setting the x labels
        # plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
        every_x_nth = self.EVERY_X_LABEL_TO_PRINT
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_x_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.xaxis.get_ticklines()):
            if n % every_x_nth != 0:
                line.set_visible(False)
        # setting the y labels
        # plt.rc('ytick', labelsize=FONT_SIZE)
        every_y_nth = self.EVERY_Y_LABEL_TO_PRINT
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_y_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.yaxis.get_ticklines()):
            if n % every_y_nth != 0:
                line.set_visible(False)
        # ANNOTATE
        texts = self.annotate_heatmap(im, valfmt="{x:.3f}",
                                 ha="center", va="center", fontsize=2)

        ax.tick_params(direction='out', pad=self.LABEL_OUT_OF_AXIS_PARAM)
        plt.tight_layout()
        if kwargs.get('save_fig'):
            save_fig = kwargs.get('save_fig')
        else:
            save_fig = False
        if save_fig:
            fig.savefig('Heatmaps/{0}-heatmapMaxColor{1}__DeltaSpan{2}>>{3}.png'.format(name, max_color, int(np.min(self.DELTAS)),
                                                                            int(np.max(self.DELTAS))), dpi=100)
        if kwargs.get('save_np_array'):
            save_np_array = kwargs.get('save_np_array')
        else:
            save_np_array = False
        if save_np_array:
            np.save('deltaDistributionArrays/{0}_delta_dist'.format(name), data)
        # print('saved plot file: {}'.format(name))
        # os.system('clear')