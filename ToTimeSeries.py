import numpy as np
from InputReader import Simple_Input_Reader


class ToTimeSeries:
    def __init__(self, bin_size, original_file, frame_count, single_frame_width, single_frame_height, **kwargs):
        """
        reads the file using the InputReader class.
        calculates the amount of bins in frame according the bin and frame dimensions.
        :param bin_size: the bin's x axis size
        :param y_size: the bin's y axis size
        :param kwargs: 'file_name': if you wish to load anything different than default file ('t1.avi'), 'channel_amount': if you use a different channel amount than [R,G,B]
        """
        self.original_file, self.frame_count, self.single_frame_width, self.single_frame_height = original_file, frame_count, single_frame_width, single_frame_height

        if kwargs.get('file_name', None):
            self.file_name = kwargs.get('file_name')
        else:
            self.file_name = 't1.avi'
        self.channel_amount = kwargs.get('channel_amount') if kwargs.get('channel_amount', None) else 3

        # self._input_reader = Simple_Input_Reader()
        self.bin_x_size = bin_size
        self.bin_y_size = bin_size
        self.bin_size = int(self.bin_x_size * self.bin_y_size)
        self.single_frame_size = int(self.single_frame_width * self.single_frame_height)
        self.number_of_bins = int(self.single_frame_size / self.bin_size)

    def into_time_series(self):
        """
        creates an array according to bins that includes the entire video of each bin seperately.

        :return: np.ndarray shape(***,**,*,*,3)
        """
        time_series_of_bins = np.zeros([self.number_of_bins, self.frame_count, self.bin_y_size, self.bin_x_size, self.channel_amount], dtype=int)
        frames_amount = self.frame_count
        x_min = 0
        x_max = self.bin_x_size
        y_min = 0
        y_max = self.bin_y_size
        bin_to_slice_index = 0
        while bin_to_slice_index < self.number_of_bins:
            x_indices = np.arange(x_min, x_max)
            #     runs on each frame
            for i in range(0, frames_amount):
                shape = self.original_file[i][y_min:y_max, x_indices].shape
                time_series_of_bins[bin_to_slice_index, i] = self.original_file[i][y_min:y_max, x_indices].copy()

            if x_max - self.single_frame_width >= 0 or (x_max+self.bin_x_size) - self.single_frame_width >= 0:
                x_min = 0
                x_max = self.bin_x_size
                y_min = y_max
                y_max = y_max + self.bin_y_size
            else:
                x_min = x_max
                x_max = x_max + self.bin_x_size
            if y_max >= self.single_frame_height:
                bin_to_slice_index=self.number_of_bins
            bin_to_slice_index += 1

        return time_series_of_bins

