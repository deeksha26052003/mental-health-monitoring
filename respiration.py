import numpy as np
import cv2
from matplotlib import pyplot as plt

# Constants for bounding box adjustments
x1, x2 = 0.4, 0.6
y1, y2 = 0.1, 0.25
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def getFaceROI(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        img = cv2.rectangle(img, (faces[0][0] + int(x1 * faces[0][2]), faces[0][1] + int(y1 * faces[0][3])),
                            (faces[0][0] + int(x2 * faces[0][2]), faces[0][1] + int(y2 * faces[0][3])), (255, 0, 0), 2)
        return [faces[0][0] + int(x1 * faces[0][2]), faces[0][1] + int(y1 * faces[0][3]),
                faces[0][0] + int(x2 * faces[0][2]), faces[0][1] + int(y2 * faces[0][3])]
    else:
        return [0, 0, 0, 0]


cv2.namedWindow("tracking")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 10)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
step = int(1000 / fps) if fps > 0 else 33


def getColorAverage(frame, color_id):
    return frame[:, :, color_id].sum() * 1.0 / (frame.shape[0] * frame.shape[1])


ok, frame = cap.read()
if not ok:
    print('Failed to read video from camera')
    exit()

idf = 0
gsums = []
# Plotting the green channel intensity over time
plt.close()
fig = plt.figure()
plt.grid(True)
plt.ion()  # Interactive mode on

plt.xlabel('Time (Frames)', fontsize=12)
plt.ylabel('Green Channel Intensity', fontsize=12)
plt.title('Heart Rate and Breath Detection', fontsize=14)

bbox = [0, 0, 10, 10]
m_diff = 0

flag = True
while flag:
    ret, frame = cap.read()

    if idf > 0:
        df = cv2.absdiff(frame, previous_frame)
        m_diff = 1.0 * df.sum() / (df.shape[0] * df.shape[1])

        if m_diff > 15:
            droi = getFaceROI(frame)
            if droi[3] > 0:
                bbox = droi

    roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    frame = cv2.rectangle(
        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    green = getColorAverage(roi, 1)  # Get green channel intensity
    gsums.append(green)

    plt.plot(gsums)
    plt.xlim(left=0, right=min(len(gsums), 300))
    plt.ylim(0, max(gsums) + 100)
    plt.pause(0.01)

    previous_frame = frame.copy()
    idf += 1
    cv2.imshow('tracking', frame)

    if cv2.waitKey(1) == 27:  # Exit on 'ESC' key
        break

cap.release()
cv2.destroyAllWindows()
