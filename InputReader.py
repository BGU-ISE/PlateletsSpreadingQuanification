from abc import ABC, abstractmethod

import cv2
import numpy as np


class Abstract_Input_Reader(ABC):

    @abstractmethod
    def input_to_np(self, input_location, grouped_frames=1):
        pass


class Simple_Input_Reader(Abstract_Input_Reader):

    def _average_grouped_ndarrays(self, group):
        return np.mean(np.array(group), axis=0)

    def _get_frame_count(self, cap, grouped_frames):
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = frame_count / grouped_frames
        return int(frame_count)

    def input_to_np(self, input_location, grouped_frames=1, **kwargs ):
        if kwargs.get('cut_file'):
            cut_file = kwargs.get('cut_file')
        else:
            cut_file = 0
        cap = cv2.VideoCapture(input_location)
        final_frame_count = self._get_frame_count(cap, grouped_frames)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        buf = np.empty((final_frame_count - cut_file, frame_height, frame_width, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while fc < final_frame_count - cut_file and ret:
            frames_grouped = 0
            group = [None]*grouped_frames
            while frames_grouped < grouped_frames and ret:
                ret, group[frames_grouped] = cap.read()
                frames_grouped += 1

            average = self._average_grouped_ndarrays(group)
            buf[fc] = average
            fc += 1
        cap.release()

        # print(np.ndarray(shape=(3,), dtype=int, order='F', buffer=np.array([180,180,180])))
        return buf, len(buf), frame_width, frame_height

