import math
import os

import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
from ToTimeSeries import ToTimeSeries as tts
from InputReader import Simple_Input_Reader as sir


class DynamicsDistributionOverTime:

    def __init__(self,  **kwargs):
        if kwargs.get('time_frame_size') is not None:
            time_frame_size = kwargs.get('time_frame_size')
        else:
            time_frame_size = 5
        self.TIME_FRAME_SIZE = time_frame_size
        self.DYNAMICS_RANGE = np.linspace(-30, 30, 61)

    def calc_dynamics_distribution_for_single_time_frame(self, time_frame_data, **kwargs):
        """
        KWARGS:
        normalize_by_max : boolean, default is 'True'
        :param time_frame_data:
        :param kwargs:
        :return:
        """
        # COUNT DELTAS
        deltas_counters = np.zeros((len(self.DYNAMICS_RANGE),))
        for frame_ctr in range(0, len(time_frame_data) - 1):
            deltas_between_current_frames = (time_frame_data[frame_ctr].flatten() - time_frame_data[frame_ctr + 1].flatten())#.flatten()
            # count appearance of each delta
            for delta_ctr in range(0, len(deltas_counters)):
                if self.DYNAMICS_RANGE[delta_ctr] != 0:
                    deltas_counters[delta_ctr] += len(deltas_between_current_frames[deltas_between_current_frames == self.DYNAMICS_RANGE[delta_ctr]])

        # NORMALIZE DELTA COUNTERS
        if kwargs.get('normalize') is not None:
            normalize = kwargs.get('normalize')
            if not normalize:
                return deltas_counters
        else:
            normalize = None
        if kwargs.get('z_score_normalize') is not None:
            z_score_normalize = kwargs.get('z_score_normalize')
            if z_score_normalize:
                return  scipy_stats.zscore(deltas_counters)
        else:
            z_score_normalize = None
        if kwargs.get('normalize_by_max') is not None:
            normalize_by_max = kwargs.get('normalize_by_max')
        else:
            normalize_by_max = True
        if not normalize_by_max:
            deltas_counters = deltas_counters / np.sum(deltas_counters)
        else:
            deltas_counters = deltas_counters / np.max(deltas_counters)

        return deltas_counters

    def calc_dynamics_distribution(self, file_path, **kwargs):
        if kwargs.get('calc_dynamics_distribution_differences') is not None:
            calc_dynamics_distribution_differences = kwargs.get('calc_dynamics_distribution_differences')
        else:
            calc_dynamics_distribution_differences = False
        sir_instance = sir()
        vid_stack, frame_number, frame_width, frame_height = sir_instance.input_to_np(file_path)

        self.number_of_time_frames = int(frame_number/self.TIME_FRAME_SIZE)
        if calc_dynamics_distribution_differences:
            time_frame_dynamics_distributions = np.zeros((self.number_of_time_frames))
        else:
            time_frame_dynamics_distributions = np.zeros((self.number_of_time_frames, len(self.DYNAMICS_RANGE)))

        time_frame_array = list()
        time_frame_start_range = 0
        time_frame_end_range = self.TIME_FRAME_SIZE

        for time_frame_ctr in range(0, self.number_of_time_frames):
            if time_frame_start_range > len(vid_stack): break
            if time_frame_end_range > len(vid_stack):
                time_frame_end_range = len(vid_stack)
            time_frame_array.append(vid_stack[time_frame_start_range: time_frame_end_range, :, :, 0].copy().astype('float'))
            time_frame_end_range += self.TIME_FRAME_SIZE
            time_frame_start_range += self.TIME_FRAME_SIZE

        if(not calc_dynamics_distribution_differences):
            for time_frame_ctr in range(0, len(time_frame_array)):
                frames_distribution_of_dynamics = self.calc_dynamics_distribution_for_single_time_frame(time_frame_array[time_frame_ctr], **kwargs)
                time_frame_dynamics_distributions[time_frame_ctr] = frames_distribution_of_dynamics
            return time_frame_dynamics_distributions
        else:
            for time_frame_ctr in range(0, len(time_frame_array)):
                frames_distribution_of_dynamics = self.calc_dynamics_distribution_for_single_time_frame(time_frame_array[time_frame_ctr], **kwargs)
                time_frame_dynamics_distributions[time_frame_ctr] = np.sum(frames_distribution_of_dynamics[
                   :int(len(frames_distribution_of_dynamics) / 2)] - frames_distribution_of_dynamics[int(
                len(frames_distribution_of_dynamics) / 2) + 1:])
            return time_frame_dynamics_distributions


if __name__ =="__main__":

    directory = os.fsencode("ForAnalyze")

    for file in os.listdir(directory):
        ddot = DynamicsDistributionOverTime(time_frame_size=10)
        FILENAME = os.fsdecode(file)
        if FILENAME == ".DS_Store":
            continue
        ax = plt.axes()
        fig, axis = plt.subplots(2)
        to_plot = ddot.calc_dynamics_distribution("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/"+FILENAME, z_score_normalize=False)
        col_labels = []
        for i in range(0, len(ddot.DYNAMICS_RANGE)):
            if i % 3 == 0:
                col_labels.append(ddot.DYNAMICS_RANGE[i])
        row_labels = ["{0}".format(x * ddot.TIME_FRAME_SIZE) for x in range(0, ddot.number_of_time_frames)]
        # sns.set()
        ax_heatmap = sns.heatmap(data=to_plot, vmax=.5, ax=ax)
        ax_heatmap.set_yticklabels(labels=row_labels, rotation=0)
        ax_heatmap.set_xticklabels(labels=col_labels, rotation=45)
        ax_heatmap.set(title='{0}'.format(FILENAME[:-4]), ylabel="Time", xlabel="UP <= ∂-I => DOWN")
        plt.show()
        axis[0] = ax_heatmap

        # figure = ax.get_figure()
        # plt.savefig("Heatmaps/DynamicsOverTimeResults/{0}_DynamicsOverTimeHeatmap.png".format(FILENAME[:-4]), dpi=200)
        #
        to_plot = ddot.calc_dynamics_distribution("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/" + FILENAME,
                                                  calc_dynamics_distribution_differences=True, z_score_normalize=False)
        ax_barplot = sns.barplot(np.uint8(np.linspace(0,len(to_plot), len(to_plot))), to_plot)
        ax_barplot.set(xlabel="Frame Number/{0}".format(ddot.TIME_FRAME_SIZE), ylabel="Up-Down Dynamics", title=FILENAME)
        ax_barplot.tick_params(axis='both', which='major', labelsize=5)
        ax_barplot.tick_params(axis='both', which='minor', labelsize=3)
        # axis[1] = ax_barplot
        plt.show()
        # figure.savefig("Heatmaps/{0}_DynamicsOverTimeDifferences.png".format(FILENAME[:-4]), dpi=200)


