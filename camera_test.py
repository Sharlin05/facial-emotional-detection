import cv2, os, sys

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print('OpenCV version:', cv2.__version__)
print('Cascade path:', cascade_path, 'exists:', os.path.isfile(cascade_path))

backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
for backend in backends:
    print('Testing backend:', backend)
    for idx in range(3):
        try:
            if backend is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, backend)
        except Exception as e:
            print('Error opening camera', idx, 'backend', backend, ':', e)
            continue
        print('  index', idx, 'opened:', cap.isOpened())
        if cap.isOpened():
            ret, frame = cap.read()
            print('    read frame ok:', ret, 'shape:', None if frame is None else frame.shape)
            cap.release()

print('Done')
