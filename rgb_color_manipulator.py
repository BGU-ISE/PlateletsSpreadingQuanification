import os
import time as time
import numpy as np
import cv2
from abc import ABC, abstractmethod
from InputReader import Simple_Input_Reader
from ToTimeSeries import ToTimeSeries as ts_main


class Abstract_Color_Manipulator(ABC):

    @abstractmethod
    def manipulate_pixel(self, rgb_to_manipulate):
        pass

    def manipulate_frame(self, frame):
        for i in range(len(frame)):
            for j in range(len(frame[i])):
                frame[i][j] = self.manipulate_pixel(frame[i][j])
        return frame.copy()

    def manipulate_video(self, video):
        for i in range(len(video)):
            video[i] = self.manipulate_frame(video[i])
        return video

class Simple_Comparison_Manipulator(Abstract_Color_Manipulator):

    def __init__(self, new_color, comperator, **kwargs):
        '''

        :param new_color: new color to return in case of change
        :param comperator: comperator used to check if need to change rgb
        :param kwargs: deep_copy_return_values, True if deep copy of return value is desired, False if shallow copy. default is false
        '''
        self.deep_copy_return_values = False if kwargs.get('deep_copy_return_values', None) else kwargs.get('deep_copy_return_values')
        self._new_color = new_color
        self._comperator = comperator

    def manipulate_pixel(self, rgb_to_manipulate):
        if self._comperator.need_to_change(rgb_to_manipulate):
            to_return = self._new_color
        else:
            to_return = rgb_to_manipulate
        return to_return.copy() if self.deep_copy_return_values else to_return


    def get_new_color(self):
        return self._new_color

    def get_comperator(self):
        return self._comperator

    def set_new_color(self, new_color):
        self._new_color=new_color

    def set_comperator(self, comperator):
        self._comperator=comperator

    def __str__(self):
        return 'simple color comperator:\nnew color: [{},{},{}]\ncomperator: {}'.format(self._new_color[0],self._new_color[1],self._new_color[2],str(self._comperator))


class Comperator(ABC):

    @abstractmethod
    def need_to_change(self, rgb_to_change):
        pass

class In_All_Ranges_Comperator(Comperator):


    @staticmethod
    def _validate_ranges(*args):
        for arg in args:
            if not isinstance(arg,range):
                raise Exception('the range {}, is not a valid range'.format(str(arg)))

    def __init__(self, r_range, g_range, b_range):
        self._validate_ranges(r_range,g_range,b_range)
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range
    def need_to_change(self, rgb_to_change):
        return ((rgb_to_change[0] in self.r_range) and (rgb_to_change[1] in self.g_range) and (rgb_to_change[2] in self.b_range))

    def __str__(self):
        return 'In all Range Comperator: range red - [{},{}]; range green - [{},{}]; range blue - [{},{}]'.\
            format(self.r_range[0], self.r_range[1], self.g_range[0], self.g_range[1], self.b_range[0],self.b_range[1])


def read_video(video_location, **kwargs):
    new_color = kwargs.get('new_color', None)
    ir = Simple_Input_Reader()
    video_frames, frames_amount, frame_width, frame_height = ir.input_to_np(video_location, 1 if kwargs.get('grouped_frames') is None else kwargs.get('grouped_frames'))

    if new_color is None:
        return video_frames, frames_amount, frame_width, frame_height
    else:
        ranges = [range(0, 0), range(0, 0), range(0, 0)] if kwargs.get('ranges', None) is None else kwargs.get('ranges')
        comperator = In_All_Ranges_Comperator(ranges[0],ranges[1],ranges[2]) if kwargs.get('comperator', None) is None else kwargs.get('comperator')
        manipulator = Simple_Comparison_Manipulator(new_color,comperator) if kwargs.get('manipulator',None) is None else kwargs.get('manipulator')
        return manipulator.manipulate_video(video_frames), frames_amount, frame_width, frame_height


class ChannelManipulator(Abstract_Color_Manipulator):
    """
    uses the manipulate_video method as the only running function.
    requires a constructor with defaultive values
    """

    def manipulate_pixel(self, rgb_to_manipulate):
        pass

    def __init__(self, new_channel_number: int = 1, **kwargs):
        self.new_channel_number = new_channel_number

    def manipulate_video(self, video):
        new_single_channel_file = np.empty((len(video), len(video[0]), len(video[0][0]), self.new_channel_number))
        for i in range(len(video)):
            for j in range(len(video[i])):
                for k in range(len(video[i][j])):
                    new_single_channel_file[i][j][k] = np.array([int(np.average(video[i][j][k]))])

        return new_single_channel_file.copy()


class IntervalAndDeltaInspector(Abstract_Color_Manipulator):
    """
    uses the manipulate_video method as the only running function.
    requires a constructor with defaultive values
    """

    def __init__(self, interval: list = [0, 8], delta_value: list = [0], **kwargs):
        """

        :param interval:
        :param delta_value:
        :param kwargs:
        color = a list of rgb values to replace with the detected pixels.
        """
        self.interval_of_interest = interval
        self.delta_of_interest = delta_value
        if kwargs.get('color'):
            self.color = kwargs.get('color')
        else:
            self.color = np.array([255, 0, 0])

    def manipulate_pixel(self, rgb_to_manipulate):
        pass

    def manipulate_frame(self, buf):
        interval_mask_first_frame = np.where((buf[0] >= self.interval_of_interest[0]) & (buf[0] < self.interval_of_interest[1]))
        first_frame_vals = buf[0][
            interval_mask_first_frame[0], interval_mask_first_frame[1], interval_mask_first_frame[2]]
        next_frame_vals = buf[1][
            interval_mask_first_frame[0], interval_mask_first_frame[1], interval_mask_first_frame[2]]
        for i in range(len(interval_mask_first_frame[0])):
            delta_between_frames = next_frame_vals[i] - first_frame_vals[i]
            if self.delta_of_interest[0] <= delta_between_frames < self.delta_of_interest[1]:
                x, y, z = interval_mask_first_frame[0][i], interval_mask_first_frame[1][i], \
                          interval_mask_first_frame[2][i]
                buf[0][x][y] = np.array(self.color, np.dtype('int64'))
        return buf

    def manipulate_video(self, file_name):
        frames_in_memory = 2

        cap = cv2.VideoCapture(file_name + ".avi")
        final_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_video = np.zeros((final_frame_count, frame_height, frame_width, 3), np.dtype('int64'))
        buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
        ret = True

        fc = 0
        while fc < final_frame_count and ret:
            # loading the data according
            in_memory_frames_ctr = 0
            while in_memory_frames_ctr < frames_in_memory:
                if np.sum(buf[frames_in_memory - 1]) > 0:
                    temp = buf[frames_in_memory - 1].copy()
                    # buf[FRAMES_IN_MEMORY-1] = np.zeros((frame_height, frame_width, 3))
                    buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
                    buf[0] = temp
                else:
                    ret, frame = cap.read()
                    buf[in_memory_frames_ctr] = frame
                    fc += 1
                in_memory_frames_ctr += 1
            buf = self.manipulate_frame(buf)
            new_video[fc - 2] = buf[0]
            new_video[fc - 1] = buf[1]

        cap.release()