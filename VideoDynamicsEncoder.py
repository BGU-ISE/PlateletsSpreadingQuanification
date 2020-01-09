import os
import numpy as np
import time as time
import cv2

TEST_FILE_NAME = "exp89_PRP_FBG_CONTROL01_R3D_SP1.avi"

class videoDynamicsEncoder:

    def __init__(self, color_scheme1: np.array = [], color_scheme2: np.array = []):
        pass

    def manipulate_frame(self, buf:np.ndarray, **kwargs):
        """
        colors are [Blue,Green,Red]
        :param buf:
        :return:
        """
        if kwargs.get('color') is not None:
            color = kwargs.get('color')
        else:
            color = [240, 0, 0]
        if kwargs.get('limit') is not None:
            limit = kwargs.get('limit')
        else:
            limit = -9

        dynamics = buf[0]-buf[1]
        buf[0] = np.where(dynamics >= limit, buf[0], color)
        return buf

    def manipulate_video(self, video_path: str, manipulated_video_path, condition, **kwargs):
        """
        condition is a function type object that receives pixel
        dynamic value (current_value-next_frame_value),current pixel time, current intensity.
        :param condition:
        :return:
        """
        video_cap = cv2.VideoCapture("{0}".format(video_path))
        # get video meta data
        final_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frames_in_memory = 2
        # video writer to AVI
        out = cv2.VideoWriter(
            "{0}".format(manipulated_video_path),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        # vid_writer = cv2.VideoWriter("videos/suspicious_dynamics_videoes/{0}".format(video_name), cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V'), fps, (frame_width, frame_height))

        buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
        ret = True

        fc = 0
        while fc < final_frame_count and ret:
            # loading the data according
            in_memory_frames_ctr = 0
            single_frame_start_time = time.time()
            while in_memory_frames_ctr < frames_in_memory:
                if np.sum(buf[frames_in_memory - 1]) > 0:
                    temp = buf[frames_in_memory - 1].copy()
                    # buf[FRAMES_IN_MEMORY-1] = np.zeros((frame_height, frame_width, 3))
                    buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
                    buf[0] = temp
                else:
                    ret, frame = video_cap.read()
                    buf[in_memory_frames_ctr] = frame
                    fc += 1
                in_memory_frames_ctr += 1
            buf = condition(buf, **kwargs)
            im = np.uint8(buf[0])
            out.write(im)
            # cv2.imshow("test", np.array(im, dtype=np.uint8))
            # new_video[fc-2] = buf[0]
            single_frame_end_time = time.time()
            print("frame No:{0} has been manipulated.took:{1:.2} seconds".format(fc - 2,
                                                                                 single_frame_end_time - single_frame_start_time))

        out.release()
        video_cap.release()
        total_end = time.time()
        # print("{0}: {1:.2f} minutes total".format(fc, (total_end - total_start) / 60))


if __name__ == "__main__":
    # color_scheme1 = np.array(['#FFA500'])
    # color_scheme2 = np.array(['#0000FF'])
    main_directory = os.fsencode("ForAnalyze")
    videoDynamicsEncoder = videoDynamicsEncoder()
    for file in os.listdir(main_directory):
        file_name = os.fsdecode(file)
        if file_name.__contains__("PRP"):
            videoDynamicsEncoder.manipulate_video("ForAnalyze/"+file_name, "videos/suspicious_dynamics_videoes/"+file_name,videoDynamicsEncoder.manipulate_frame, limit=-12)

