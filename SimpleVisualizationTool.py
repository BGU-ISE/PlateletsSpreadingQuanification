import numpy as np
import cv2
import time


class simpleVisualization:
    def __init__(self):
        pass

    @staticmethod
    def visualize_single_image(image: np.ndarray, title: str = "Image {0}".format(":")):
        """
        :param image: a np.ndarray shape (*, *, 3), when '*' means and positive integer
        :param title: the title you would like the image window to have
        :return:
        """
        image = image.astype(np.uint8)
        cv2.imshow(title, image)
        cv2.waitKey(0)

    @staticmethod
    def visualize_video(video, **kwargs):
        """
        kwardgs can be:
        {
        frame_update_frequncy: int - default 1/0.5 seconds,
        gray: boolean - default is False,
        frame_title: str - default is 'Video'
        }
        :param video: a np.ndarray shape (**,*, *, 3), when '*' means and positive integer, and '**' is the number of frames dimension.
        :param kwargs: dictionary
        :return:
        """
        frame_num = video.shape[0]
        frame_title = "Video"
        if kwargs.get('frame_title'):
            frame_title = kwargs.get('frame_title')
        frame_rate = 0.5
        if kwargs.get('frame_update_frequency'):
            frame_rate = kwargs.get('frame_update_frequency')
        fc = 0
        while fc < frame_num:
            image = video[fc]
            image = image.astype(np.uint8)
            if kwargs.get('gray'):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow(frame_title, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fc += 1
            time.sleep(frame_rate)
        cv2.destroyAllWindows()

