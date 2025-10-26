import os
import cv2
import numpy as np
import tensorflow as tf
from time import time

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

MODEL_CANDIDATES = ['transfer_model_finetuned.keras', 'transfer_model_finetuned_2.keras', 'best_model.h5', 'emotion_recognition_model.h5']
OUT_DIR = 'tools/webcam_output'
ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
ensure_dir(OUT_DIR)


def load_model():
    for m in MODEL_CANDIDATES:
        if os.path.isfile(m):
            try:
                model = tf.keras.models.load_model(m)
                print('Loaded model:', m)
                return model
            except Exception as e:
                print('Failed loading', m, e)
    raise FileNotFoundError('No model file found. Train a model first or pass a model path.')


def preprocess_gray(img, img_size=IMG_SIZE):
    # img: grayscale or BGR; returns (1, H, W, 1)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


def try_open_camera(camera_index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    for b in backends:
        try:
            if b is None:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, b)
            if cap is not None and cap.isOpened():
                print(f'Opened camera index={camera_index} backend={b}')
                return cap
            else:
                if cap:
                    cap.release()
        except Exception as e:
            print('Backend', b, 'failed:', e)
    return None


def main(frames_to_process=50, camera_index=0):
    model = load_model()

    cap = try_open_camera(camera_index)
    if cap is None:
        print('Failed to open camera. Run tools/camera_test.py for diagnostics.')
        return

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if cascade.empty():
        print('Failed to load Haar cascade')
        return

    frame_count = 0
    saved = 0
    start = time()
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print('Failed reading frame')
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        preds = []
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            x_in = preprocess_gray(roi)
            p = model.predict(x_in)
            idx = int(np.argmax(p, axis=1)[0])
            label = EMOTIONS[idx]
            prob = float(p[0, idx])
            preds.append((label, prob, (x,y,w,h)))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f'{label} {prob:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # print summary for this frame
        print(f'Frame {frame_count+1}/{frames_to_process}, faces={len(faces)}')
        for p in preds:
            print('  ', p[0], f'{p[1]:.4f}', 'box=', p[2])

        # save annotated frames every 10 frames or when faces found
        if saved < 10 and (len(faces) > 0 or frame_count % 10 == 0):
            out_path = os.path.join(OUT_DIR, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_count += 1

    cap.release()
    total = time() - start
    print(f'Done. Processed {frame_count} frames in {total:.1f}s. Saved {saved} annotated frames in {OUT_DIR}')

if __name__ == '__main__':
    main()
