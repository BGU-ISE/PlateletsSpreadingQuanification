import math

import numpy as np
import matplotlib.pyplot as plt
from librosa import feature as libzcr
from ToTimeSeries import ToTimeSeries as tts
from InputReader import Simple_Input_Reader as sir


class DynamicsDirectionChangeRate:

    def zero_crossing_rate(self, vector: np.ndarray):
        return libzcr.zero_crossing_rate(vector)

    def dynamics_intensities_sum_of_change(self, vector: np.ndarray):
        return np.sum(vector)

    def dynamics_intensities_range_of_change(self, vector: np.ndarray):
        return np.max(vector) - np.min(vector)


class ZeroCrossingRateFeature:
    def calc_dynamics_vector(self, vector):
        dynamics_vector = np.zeros((len(vector)-1, ))
        for i in range(0, len(dynamics_vector)-1):
            dynamics_vector[i] = np.average(vector[i]) - np.average(vector[i+1])
        return dynamics_vector

    def split_into_bins(self, stack_of_images: np.ndarray, bin_size: int = 4):
        bin_transformer = tts(bin_size=bin_size, original_file=stack_of_images, frame_count=len(stack_of_images), single_frame_width=len(stack_of_images[1]), single_frame_height=len(stack_of_images[1][1]))
        return bin_transformer.into_time_series()

    def calc_zcr_per_bin(self, single_bin_stack: np.ndarray, time_intervals: int = 5, **kwargs):
        if kwargs.get('use_zcr'):
            use_zcr = kwargs.get('use_zcr')
        else:
            use_zcr = True

        if kwargs.get('function_to_use'):
            function_to_use = kwargs.get('function_to_use')
        else:
            function_to_use = DynamicsDirectionChangeRate.zero_crossing_rate
        zcr_per_bin = np.zeros((int(len(single_bin_stack)/time_intervals), ))
        for i in range(0, len(zcr_per_bin)):
            j = i * time_intervals
            bins_stack_per_interval = single_bin_stack[j:j+time_intervals]
            ddcr = DynamicsDirectionChangeRate()
            if use_zcr:
                zcr_per_bin[int(j/time_intervals)] = function_to_use(self.calc_dynamics_vector(bins_stack_per_interval))
            else:
                zcr_per_bin[int(j / time_intervals)] = ddcr.dynamics_intensities_sum_of_change(self.calc_dynamics_vector(bins_stack_per_interval))
        return zcr_per_bin

    def calc_zcr_for_all_bins(self, all_bin_stack, time_intervals: int = 5, **kwargs):
        zcr_list_per_bin = list()
        for bin_ctr in range(0, len(all_bin_stack)):
            zcr_list_per_bin.append(self.calc_zcr_per_bin(all_bin_stack[bin_ctr], time_intervals, **kwargs))

        return zcr_list_per_bin

    def plot_all_bins(self, all_bin_stack, time_intervals: int = 5, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if kwargs.get('limit_bin_number'):
            limit_bin_number = kwargs.get('limit_bin_number')
        else:
            limit_bin_number = 400

        zcr_results_for_bin = self.calc_zcr_for_all_bins(all_bin_stack=all_bin_stack, time_intervals=time_intervals, **kwargs)
        for i in range(0, min(limit_bin_number, len(zcr_results_for_bin))):
            time_vector = np.linspace(1, len(zcr_results_for_bin[i])+1, len(zcr_results_for_bin[i]))
            ax.scatter(time_vector, zcr_results_for_bin[i])
            # plt.plot(time_vector, zcr_results_for_bin[i], linestyle=':', linewidth=2, markersize=12)


        plt.show()



sir = sir()
zcr = ZeroCrossingRateFeature()
vid_stack, frame_number, frame_width, frame_height = sir.input_to_np("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/PLT_coll4_exp.63_Mn2_PI_1.avi")
# vid_stack, frame_number, frame_width, frame_height = sir.input_to_np("/Users/yishaiazabary/Desktop/University/FinalProject/videoes/IRM Artifact research videos and results/PLT_coll4_exp.63_control_BACKGROUND.avi")
vid_stack_in_bins = zcr.split_into_bins(vid_stack, bin_size=5)
# print(zcr.calc_zcr_per_bin(vid_stack_in_bins[0], 5))
# print(zcr.calc_zcr_for_all_bins(vid_stack_in_bins)
zcr.plot_all_bins(vid_stack_in_bins, time_intervals=int(frame_number/10), limit_bin_number=math.inf, function_to_use=DynamicsDirectionChangeRate.zero_crossing_rate)


# todo: change the zcr to a custom class which detects the