import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf

# Try to import project constants from train_model if available
try:
    from train_model import EMOTIONS, IMG_SIZE
except Exception:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    IMG_SIZE = 48


def load_model(model_path='emotion_recognition_model.h5'):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training first or point --model to the .h5 file.")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image_gray(image, img_size=IMG_SIZE):
    # Accepts grayscale or color images (BGR), returns shape (1, H, W, 1)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image


def predict_emotion(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image {img_path}")
        return
    x = preprocess_image_gray(img)
    pred = model.predict(x)
    emotion_index = int(np.argmax(pred, axis=1)[0])
    emotion = EMOTIONS[emotion_index]
    print(f"Predicted Emotion: {emotion}")
    return emotion, pred


def webcam_emotion_recognition(model, camera_index=0, cascade_path=None):
    # Try to open camera with common Windows backends for better reliability
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    cap = None
    for b in backends:
        if b is None:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, b)
        if cap is not None and cap.isOpened():
            print(f"Opened camera index={camera_index} with backend={b}")
            break

    if cap is None or not cap.isOpened():
        print(f"Failed to open camera {camera_index} with tried backends.\n" \
              "- Ensure camera is connected, not used by another app, and Windows privacy settings allow access.\n" \
              "- Try other camera indices (1,2) or run the tools/camera_test.py script for diagnostics.")
        return

    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Could not load cascade at {cascade_path}")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            x_in = preprocess_image_gray(roi)
            pred = model.predict(x_in)
            emotion_index = int(np.argmax(pred, axis=1)[0])
            emotion = EMOTIONS[emotion_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Live demo: single image or webcam emotion recognition')
    parser.add_argument('--model', '-m', default='emotion_recognition_model.h5', help='Path to saved .h5 model')
    parser.add_argument('--image', '-i', help='Path to an image to predict')
    parser.add_argument('--webcam', '-w', action='store_true', help='Run webcam emotion recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for webcam')
    parser.add_argument('--cascade', help='Path to Haar cascade xml (optional)')
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except Exception as e:
        print('Error loading model:', e)
        sys.exit(1)

    if args.image:
        predict_emotion(model, args.image)

    if args.webcam:
        webcam_emotion_recognition(model, camera_index=args.camera, cascade_path=args.cascade)

    if not args.image and not args.webcam:
        print('No action requested. Use --image IMAGE or --webcam')


if __name__ == '__main__':
    main()
