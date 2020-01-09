from SimpleVisualizationTool import *
from rgb_color_manipulator import read_video
from ToTimeSeries import ToTimeSeries
import numpy as np

print('new color green')
new_color = [0,255,0]
print('range is 150-200')
gray_range = range(150,200)
ranges = [gray_range,gray_range,gray_range]
print('reading video and manipulating frames')
video, frames_amount, frame_width, frame_height = read_video('t1.avi', grouped_frames=11, ranges=ranges)
print('displaying video')
simpleVisualization.visualize_video(video)

tts = ToTimeSeries(90, 90, video, frames_amount, frame_width, frame_height)
time_series = tts.into_time_series()
for bin in time_series:
    simpleVisualization.visualize_video(bin)


# def get_time_series(video_location='t1.avi', ranges=[range(150,200),range(150,200),range(150,200)], side_of_square=2,new_color=np.array([0,255,0])):
#     video, frames_amount, frame_width, frame_height = read_video(video_location, new_color=new_color, grouped_frames=20, ranges=ranges)
#     x = frame_width/side_of_square
#     y = frame_height/side_of_square
#
#     tts = ToTimeSeries(x,y,video,frames_amount,frame_width,frame_height)
#     return  tts.into_time_series()
#
# bins = get_time_series()
# for bin in bins:
#     simpleVisualization.visualize_video(bin)
