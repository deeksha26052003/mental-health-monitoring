import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load OpenCV face and smile cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')


def capture_and_analyze():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture a photo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Webcam - Press C to capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            filename = 'photo.jpg'
            cv2.imwrite(filename, frame)
            print(f'Photo saved to {filename}')
            cap.release()
            cv2.destroyAllWindows()
            analyze_photo(filename)
            return

    cap.release()
    cv2.destroyAllWindows()


def analyze_photo(filename):
    img = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_rgb = img_rgb[y:y+h, x:x+w]

        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        # Simple rule-based emotion
        if len(smiles) > 0:
            emotion = "Happy 😊"
            color = (0, 255, 0)
        elif len(eyes) == 0:
            emotion = "Tired / Eyes Closed 😴"
            color = (255, 165, 0)
        else:
            emotion = "Neutral 😐"
            color = (255, 255, 0)

        results.append({
            'box': (x, y, w, h),
            'emotion': emotion,
            'smiles': len(smiles),
            'eyes': len(eyes)
        })

        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_rgb, emotion, (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if results:
        print(f"\nDetected {len(results)} face(s):")
        for i, r in enumerate(results):
            print(
                f"  Face {i+1}: {r['emotion']} | Eyes: {r['eyes']} | Smiles: {r['smiles']}")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img_rgb)
        ax1.set_title(
            f"Facial Expression Analysis\n{results[0]['emotion']}", fontsize=13)
        ax1.axis('off')

        categories = ['Faces Detected', 'Eyes Detected', 'Smiles Detected']
        values = [len(results), results[0]['eyes'], results[0]['smiles']]
        ax2.bar(categories, values, color=['steelblue', 'green', 'orange'])
        ax2.set_title("Detection Summary", fontsize=13)
        ax2.set_ylabel("Count")
        plt.tight_layout()
        plt.show()
    else:
        print("No face detected.")
        plt.imshow(img_rgb)
        plt.title("No Face Detected")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    capture_and_analyze()
