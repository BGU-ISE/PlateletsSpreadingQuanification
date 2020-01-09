import math
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
from ToTimeSeries import ToTimeSeries as tts
from InputReader import Simple_Input_Reader as sir


class DynamicsDistributionOverRadii:

    def __init__(self,  **kwargs):
        self.DYNAMICS_RANGE = np.linspace(-30, 30, 61)

    def calc_dynamics_distribution_for_single_radii(self, radii_data, **kwargs):
        """
        radii_data.shape = (frames_number, pixel_rows[x], pixel_columns[y])
        KWARGS:
        normalize_by_max : boolean, default is 'True'
        :param time_frame_data:
        :param kwargs:
        :return:
        """
        # COUNT DELTAS
        deltas_counters = np.zeros((len(self.DYNAMICS_RANGE),))
        for frame_ctr in range(0, len(radii_data) - 1):
            deltas_between_current_frames = (radii_data[frame_ctr].flatten() - radii_data[frame_ctr + 1].flatten())#.flatten()
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
                return scipy_stats.zscore(deltas_counters)
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
        sir_instance = sir()
        vid_stack, frame_number, frame_width, frame_height = sir_instance.input_to_np(file_path)

        pixels_in_time_by_radii_array = list()
        center_point_x, center_point_y = int(frame_width/2), int(frame_height/2) # center point by frame size, NOT center of platelet
        possible_radii_array = np.linspace(1, center_point_x, center_point_x-1)
        radii_dynamics_distributions = np.zeros((len(possible_radii_array), len(self.DYNAMICS_RANGE)))

        # todo: extract the pixels by radii for each radii - preformed in the hands-on notebook from Adir's class
        for radius in possible_radii_array:
            if radius > center_point_x or radius > center_point_y:
                pass
            else:
                y, x = np.ogrid[-center_point_y:frame_height-center_point_y, -center_point_x:frame_width-center_point_x]
                mask = x**2 + y**2 <= radius**2
                raddi_vid_stack = np.zeros((frame_number, mask[mask].shape[0]))
                for i, frame in enumerate(vid_stack):
                    frame = frame[:,:,0][mask]
                    raddi_vid_stack[i] = frame
                pixels_in_time_by_radii_array.append(raddi_vid_stack)
        #     if time_frame_start_range > len(vid_stack): break
        #     if time_frame_end_range > len(vid_stack):
        #         time_frame_end_range = len(vid_stack)
        #     time_frame_array.append(vid_stack[time_frame_start_range: time_frame_end_range, :, :, 0].copy().astype('float'))
        #     time_frame_end_range += self.TIME_FRAME_SIZE
        #     time_frame_start_range += self.TIME_FRAME_SIZE
        # todo: send each radii data for distribution measurement
        for raddi_ctr in range(0, len(pixels_in_time_by_radii_array)):
            radii_dynamics_distributions[raddi_ctr] = self.calc_dynamics_distribution_for_single_radii(pixels_in_time_by_radii_array[raddi_ctr], **kwargs)

        return radii_dynamics_distributions


ax = plt.axes()
FILENAME = "PRP_FBG_exp.63_Mn2_at15min_SP2.avi"
ddot = DynamicsDistributionOverRadii()
to_plot = ddot.calc_dynamics_distribution("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/"+FILENAME, z_score_normalize=True)
# to_plot[:, 31:32] = 0
# to_plot[0:2] = 0
col_labels = []
for i in range(0, len(ddot.DYNAMICS_RANGE)):
    if i % 3 == 0:
        col_labels.append(ddot.DYNAMICS_RANGE[i])
row_labels = ["{0}".format(x) for x in range(0, len(to_plot))]
sns.set()
ax = sns.heatmap(data=to_plot, vmax=0.2, ax=ax)
ax.set_yticklabels(labels=row_labels, rotation=0)
ax.set_xticklabels(labels=col_labels, rotation=45)
ax.set(title='{0}'.format(FILENAME[:-4]), ylabel="Radii", xlabel="UP <= ∂-I => DOWN")
plt.show()
figure = ax.get_figure()
figure.savefig("Heatmaps/DynamicsOverRadii/{0}_DynamicsOverTimeHeatmap.png".format(FILENAME[:-4]), dpi=200)




