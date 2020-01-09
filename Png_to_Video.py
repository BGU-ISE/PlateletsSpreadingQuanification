import glob
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from ToTimeSeries import ToTimeSeries as ts_main

INTERVALS_LOWER_BOUND = 0
INTERVALS_UPPER_BOUND = 257
INTERVALS_SKIP_VALUE = 8
CUT_FILE = 0
FRAMES_IN_MEMORY = 2
BIN_SIZE = 4
DELTA_MAX_VALUE = 30
DELTA_MIN_VALUE = -30
DELTA_BINS_NUM = 61
VideoFilesIntensityDynamicsOverTimeResultsPath = "VideoFilesIntensityDynamicsOverTimeResults"
VideoFilesToProcessIntensityDynamicsOverTimePath = "VideoFilesToProcessIntensityDynamicsOverTime"
INTERVALS = np.arange(INTERVALS_LOWER_BOUND, INTERVALS_UPPER_BOUND, INTERVALS_SKIP_VALUE)
DELTAS = np.linspace(DELTA_MIN_VALUE, DELTA_MAX_VALUE, num=DELTA_BINS_NUM)

# delta_distribution_per_interval = np.zeros((,len(INTERVALS), len(DELTAS)))
previous_bin_frame_averages = None

# PLOTTING PARAMETERS
MAX_COLOR = 0.2
LABEL_FONT_SIZE = 15
LABEL_OUT_OF_AXIS_PARAM = 15
FONT_SIZE = 50
EVERY_X_LABEL_TO_PRINT = 8
EVERY_Y_LABEL_TO_PRINT = 4
print("done")


# %%

def _in_which_interval(val, intervals):
    idx = np.where((intervals <= val))
    idx = idx[0][len(idx[0]) - 1]
    return idx


def _which_delta(val, deltas):
    # array = np.asarray(deltas)
    idx = (np.abs(deltas - val)).argmin()
    return idx


print("done")


# FUNCTIONS:
def _count_deltas(frame_count, buf_bins, bin_number, frame_distribution_ctr):
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
                bin_current_interval = _in_which_interval(bin_current_frame_average, INTERVALS)
                bin_current_delta_index = _which_delta(current_bin_avgs_delta, DELTAS)
                delta_distribution_per_interval[frame_distribution_ctr][bin_current_interval][
                    bin_current_delta_index] += 1

                frame_ctr += 1

            bin_ctr += 1
    else:
        # if does not exist yet, create a new array
        previous_bin_frame_averages = np.zeros((bin_number))

        bin_ctr = 0
        while bin_ctr < bin_number:
            frame_ctr = 0
            while frame_ctr < frame_count - 1:
                bin_current_frame_average = np.average(buf_bins[bin_ctr][frame_ctr])
                bin_next_frame_average = np.average(buf_bins[bin_ctr][frame_ctr + 1])
                previous_bin_frame_averages[bin_ctr] = bin_next_frame_average
                current_bin_avgs_delta = bin_next_frame_average - bin_current_frame_average

                bin_current_interval = np.where((INTERVALS <= bin_current_frame_average))[0][
                    -1]  # _in_which_interval(bin_current_frame_average, INTERVALS)
                bin_current_delta_index = (
                    np.abs(DELTAS - current_bin_avgs_delta)).argmin()  # _which_delta(current_bin_avgs_delta, DELTAS)
                delta_distribution_per_interval[frame_distribution_ctr][bin_current_interval][
                    bin_current_delta_index] += 1

                frame_ctr += 1

            bin_ctr += 1


def _normalizing_deltas():
    interval_ctr = 0
    frame_ctr = 0
    while (frame_ctr < len(delta_distribution_per_interval)):
        interval_ctr = 0
        while interval_ctr < len(INTERVALS) - 1:
            current_interval_sum = np.sum(delta_distribution_per_interval[frame_ctr][interval_ctr])
            if current_interval_sum == 0:
                current_interval_sum = 1
            delta_distribution_per_interval[frame_ctr][interval_ctr] = delta_distribution_per_interval[frame_ctr][
                                                                           interval_ctr] / current_interval_sum
            interval_ctr += 1
        frame_ctr += 1


def heatmap(data, row_labels, col_labels, ax=None,
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
    ax.set_xticklabels(col_labels, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=LABEL_FONT_SIZE, fontweight='bold')

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


def analyze_file(file_name):
    global delta_distribution_per_interval

    # LAST DIMENSION MUST DIVIDE 256 WITHOUT A REMAINDER. (THE POWER OF 2)

    cap = cv2.VideoCapture("{0}/{1}".format(VideoFilesToProcessIntensityDynamicsOverTimePath, file_name))
    final_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delta_distribution_per_interval = np.zeros((int(final_frame_count), len(INTERVALS), len(DELTAS)))

    ret = True

    print("done")

    total_start = time.time()
    print("starting video {0}, frames: {1}".format(file_name, int(final_frame_count)))
    fc = 0
    buf = np.zeros((FRAMES_IN_MEMORY, frame_height, frame_width, 3), np.dtype('uint8'))

    while fc < final_frame_count - CUT_FILE - (FRAMES_IN_MEMORY - 1) and ret:
        # loading the data according
        in_memory_frames_ctr = 0
        while in_memory_frames_ctr < FRAMES_IN_MEMORY:
            if np.sum(buf[FRAMES_IN_MEMORY - 1]) > 0:
                temp = buf[FRAMES_IN_MEMORY - 1].copy()
                # clearing memory
                del buf

                buf = np.zeros((FRAMES_IN_MEMORY, frame_height, frame_width, 3), np.dtype('uint8'))
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
        ts = ts_main(bin_size=BIN_SIZE, y_size=BIN_SIZE, original_file=single_channel_buf, frame_count=FRAMES_IN_MEMORY,
                     single_frame_width=frame_width, single_frame_height=frame_height, channel_amount=1)
        buf_in_bins = ts.into_time_series()
        number_of_bins = ts.number_of_bins
        # counting the deltas
        _count_deltas(frame_count=len(buf), buf_bins=buf_in_bins, bin_number=number_of_bins, frame_distribution_ctr=fc)
        del single_channel_buf
        del buf_in_bins
        end = time.time()

        print("{0}: {1:.2f} seconds to count deltas".format(fc, end - start))
    del buf
    del frame
    del ret

    cap.release()
    total_end = time.time()
    print("{0}: {1:.2f} minutes total".format(fc, (total_end - total_start) / 60))
    # os.system('afplay alarm.m4a')

    _normalizing_deltas()
    # emptying the frames folder
    frames_t0_delete = glob.glob("{0}/Frames/*".format(VideoFilesIntensityDynamicsOverTimeResultsPath))
    for f in frames_t0_delete:
        os.remove(f)
    # for figure presentation
    intervals_list = ["{0} >".format(inter) for inter in INTERVALS]
    deltas_list = DELTAS
    fc = 0
    while fc < len(delta_distribution_per_interval):
        data = delta_distribution_per_interval[fc]
        deltas_distribution = np.array(np.around(data, 6))

        fig, ax = plt.subplots(figsize=(2 ^ 16, 2 ^ 10))
        plt.title("frame No. {0}".format(fc), fontsize=20)
        im, cbar = heatmap(deltas_distribution, intervals_list, np.around(deltas_list, 1), ax=ax,
                           cmap="YlGnBu", cbarlabel="", vmax=MAX_COLOR)
        # setting the x labels
        # plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
        every_x_nth = EVERY_X_LABEL_TO_PRINT
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_x_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.xaxis.get_ticklines()):
            if n % every_x_nth != 0:
                line.set_visible(False)
        # setting the y labels
        # plt.rc('ytick', labelsize=FONT_SIZE)
        every_y_nth = EVERY_Y_LABEL_TO_PRINT
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_y_nth != 0:
                label.set_visible(False)

        for n, line in enumerate(ax.yaxis.get_ticklines()):
            if n % every_y_nth != 0:
                line.set_visible(False)

        ax.tick_params(direction='out', pad=LABEL_OUT_OF_AXIS_PARAM)
        plt.tight_layout()
        fig.savefig('{0}/Frames/frame_{1}.png'.format(VideoFilesIntensityDynamicsOverTimeResultsPath, fc), dpi=100)
        # np.save('deltaDistributionArrays/frame_{0}'.format(filename),delta_distribution_per_interval)
        print('saved frame no.: {}'.format(fc))
        os.system('clear')
        fc += 1


directory = os.fsencode(VideoFilesToProcessIntensityDynamicsOverTimePath)
total_files = ""
finished_files = list()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
        analyze_file(filename)
        finished_files.append(filename)


def saving_png_files_as_videos(directory_path: str):

    def bubbleSort(arr):
        def custom_cmp(a, b):
            return int((a[6:])[:-4]) > int((b[6:])[:-4])

        # Traverse through all array elements
        for i in range(len(arr)):
            # Last i elements are already in place
            for j in range(0, len(arr) - i - 1):
                if custom_cmp(arr[j], arr[j + 1]):
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def get_file_path(directory_path: str):
        directory = os.fsencode(directory_path)
        found_images = list()
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith('.png'):
                found_images.append(filename)
        return found_images

    # print("s/")

    found_images = bubbleSort(get_file_path(directory_path))
    image_path = "{0}/{1}".format(directory_path, found_images[0])
    image = cv2.imread(filename=image_path)
    final_frame_count = len(found_images)
    frame_width = len(image[0])
    frame_height = len(image)
    fps = 7
    stat_time_for_video_saving = time.time()
    out = cv2.VideoWriter("{0}/Video/to_name.avi".format(VideoFilesIntensityDynamicsOverTimeResultsPath),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    # images_list = list()
    for image_path in found_images:
        image_full_path = "{0}/{1}".format(directory_path, image_path)
        image = cv2.imread(filename=image_full_path)
        # images_list.append(image)
        out.write(image)
        print('saved frame no.: {} to video file'.format((image_path[6:])[:-4]))
        # out.write(image)
    # for image in images_list:
    end_time_for_video_saving = time.time()
    print("total time for video saving: {0:.2f}".format(end_time_for_video_saving-stat_time_for_video_saving))
    out.release()
#     time for separate saving png is 3.13 seconds.
#   time for combined is 6.50


saving_png_files_as_videos("{0}/Frames".format(VideoFilesIntensityDynamicsOverTimeResultsPath))
