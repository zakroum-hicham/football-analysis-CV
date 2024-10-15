import supervision as sv
import cv2

# get the total number of frames 
def get_number_of_frames(VIDEO_SRC):
    cap = cv2.VideoCapture(VIDEO_SRC)
    if not cap.isOpened():
        return -1
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return total_frames,fps

def get_frames(video_src,stride=1,start=0,end=None):
    frame_generator = sv.get_video_frames_generator(source_path=video_src, stride=stride,start=start,end=end)
    return frame_generator