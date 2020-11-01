import cv2
import pyttsx3


def voice_warning(text: str) -> None:
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)
    engine.setProperty('volume', 1.5)
    engine.say(text)
    engine.runAndWait()


def main():
    video_capture = cv2.VideoCapture(0)
    base_frame = None
    warned_list = [False]

    while True:
        detected = False
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (17, 17), 2)

        base_frame = gray_blurred if base_frame is None else gray_blurred
        diff = cv2.absdiff(base_frame, gray_blurred)
        cleaned_diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]

        (contours, _) = cv2.findContours(cleaned_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 5_000:
                continue
            detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        warned_list.append(detected)
        if warned_list[-1] and not warned_list[-2]:
            voice_warning('Somebody in the place, alarm.')
            warned_list = warned_list[-2:]

        cv2.imshow('Motion_detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
