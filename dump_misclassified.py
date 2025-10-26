import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Try to import data loader from train_model
try:
    from train_model import load_data, EMOTIONS, IMG_SIZE
except Exception:
    # Fallbacks if train_model isn't importable
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    IMG_SIZE = 48
    def load_data(data_dir=None, emotions=EMOTIONS, img_size=IMG_SIZE):
        raise RuntimeError('load_data not available from train_model; please run this script from the project root where train_model.py exists')


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def save_image(arr, path):
    # arr expected in [0,1] float or uint8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255.0).clip(0,255).astype(np.uint8)
    cv2.imwrite(path, arr)


def main():
    parser = argparse.ArgumentParser(description='Dump misclassified validation images for manual inspection')
    parser.add_argument('--model', '-m', default='best_model.h5', help='Path to saved Keras model (.h5 or .keras)')
    parser.add_argument('--out', '-o', default='tools/misclassified', help='Output folder to save misclassified images')
    parser.add_argument('--data_dir', '-d', default=None, help='Dataset root (optional)')
    parser.add_argument('--max_per_class', type=int, default=200, help='Maximum misclassified images to save per true class')
    args = parser.parse_args()

    model_path = args.model
    out_dir = args.out
    ensure_dir(out_dir)

    if not os.path.isfile(model_path):
        print(f'Model file not found: {model_path}')
        return

    print('Loading model:', model_path)
    model = tf.keras.models.load_model(model_path)

    print('Loading dataset... (this will load the same data used for training)')
    # If the user didn't provide a data_dir, call load_data() without arguments so it
    # uses the default DATA_DIR defined in train_model.py. Passing None caused a
    # TypeError inside train_model.load_data previously.
    if args.data_dir:
        X, y = load_data(data_dir=args.data_dir)
    else:
        X, y = load_data()
    if X.size == 0:
        print('No data loaded. Aborting.')
        return

    # same split as training script
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_true = np.argmax(y_val, axis=1)

    print('Predicting on validation set...')
    preds = model.predict(X_val, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    mis_idx = np.where(y_pred != y_true)[0]
    print(f'Total validation samples: {len(y_true)}, misclassified: {len(mis_idx)}')

    # Create per-class counters
    counters = {i:0 for i in range(len(EMOTIONS))}

    report_lines = []

    for idx in mis_idx:
        true = int(y_true[idx])
        pred = int(y_pred[idx])
        if counters[true] >= args.max_per_class:
            continue
        img = X_val[idx]
        # remove channel dim if present
        if img.ndim == 3 and img.shape[-1] == 1:
            img2 = img[:, :, 0]
        elif img.ndim == 3 and img.shape[-1] == 3:
            img2 = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        else:
            img2 = img
        filename = f"idx{idx:06d}_true_{EMOTIONS[true]}_pred_{EMOTIONS[pred]}.png"
        out_path = os.path.join(out_dir, filename)
        # save grayscale as single-channel PNG
        if img2.dtype != 'uint8':
            img_save = (img2 * 255.0).clip(0,255).astype('uint8')
        else:
            img_save = img2
        try:
            # if shape is (H,W) write as is; if (H,W,3) ensure BGR
            if img_save.ndim == 2:
                cv2.imwrite(out_path, img_save)
            elif img_save.ndim == 3 and img_save.shape[2] == 3:
                cv2.imwrite(out_path, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
            else:
                # squeeze last dim
                cv2.imwrite(out_path, img_save.squeeze())
            counters[true] += 1
            prob = float(preds[idx, pred])
            report_lines.append(f"{filename},{EMOTIONS[true]},{EMOTIONS[pred]},{prob:.4f}\n")
        except Exception as e:
            print('Failed saving', out_path, e)

    # Save report
    report_path = os.path.join(out_dir, 'misclassified_report.csv')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('filename,true,pred,prob\n')
        f.writelines(report_lines)

    print('Saved misclassified images and report to', out_dir)

if __name__ == '__main__':
    main()
