import cv2
import pyttsx3
from time import time, sleep
import numpy as np


def voice_warning(text: str) -> None:
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)
    engine.setProperty('volume', 1.5)
    engine.say(text)
    engine.runAndWait()


def detect_move(video_capture, fps,
                base_frame=None,
                warned_list: list = [False]
                ):

    background_list = [0]
    previous_frame = None

    while True:
        detected = False
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (17, 17), 2)

        base_frame = gray_blurred if base_frame is None else base_frame
        diff = cv2.absdiff(base_frame, gray_blurred)
        cleaned_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        diff_previous = cv2.absdiff(previous_frame, gray_blurred) if previous_frame is not None else 0
        cleaned_diff_previous = cv2.threshold(diff_previous, 25, 255, cv2.THRESH_BINARY)[1]
        background_list.append(0 if cleaned_diff_previous.sum() == 0 else 1)
        previous_frame = gray_blurred

        (contours, _) = cv2.findContours(cleaned_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valuable_change = frame.shape[0] * frame.shape[1] * 0.01
        for contour in contours:
            if cv2.contourArea(contour) < valuable_change:
                continue
            detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        warned_list.append(detected)
        if warned_list[-1] and not warned_list[-2]:
            voice_warning('Somebody in the place, alarm.')

        warned_list = warned_list[-2:]
        SECOND = 2
        len_ = int(fps*SECOND) if fps*SECOND <= len(background_list)-1 else len(background_list)

        if sum(background_list[-len_:]) == 0:
            base_frame = gray_blurred
            background_list = [0]

        cv2.imshow('Motion_detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    video_capture = cv2.VideoCapture(0)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    detect_move(video_capture, fps)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
