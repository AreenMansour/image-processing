import numpy as np
import cv2


def main(video_path, video_type):
    """
    Main entry point for exercise 1.

    Parameters:
    - video_path: Path to the video file.
    - video_type: Category of the video (either 1 or 2).

    Returns:
    - A tuple of integers representing the frame numbers for which
      the scene cut was detected (i.e., the last frame index of the
      first scene and the first frame index of the second scene).
    """
    dis, counter = 0, 0
    lst_fr, scene = [], []
    return1 = True
    capture = cv2.VideoCapture(video_path)
    while capture.isOpened() and return1:
        return1, frame = capture.read()
        if not return1:
            break
        lst_fr.append(frame)
        counter += 1
    capture.release()
    for i in range(1, counter):
        gray_frame_prev = cv2.cvtColor(lst_fr[i - 1], cv2.COLOR_BGR2GRAY)
        prev = cv2.calcHist([gray_frame_prev], [0], None,
                                 [256], [0, 256]).flatten()
        gray_frame_curr = cv2.cvtColor(lst_fr[i], cv2.COLOR_BGR2GRAY)
        curr = cv2.calcHist([gray_frame_curr], [0], None,
                                 [256], [0, 256]).flatten()
        if video_type == 2:
            cum_prev = np.cumsum(prev)
            cum_curr = np.cumsum(curr)
            cumulative = np.sum(np.abs(cum_prev - cum_curr))
            dis = cumulative
        elif video_type == 1:
            dis = np.sum(np.abs(prev - curr))
        scene.append(dis)
    index = np.argmax(scene)
    return (index, index + 1)



