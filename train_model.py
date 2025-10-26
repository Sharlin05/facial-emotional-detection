import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset folder (adjust if necessary)
# Prefer the 'train' subfolder which exists in this repository layout
DATA_DIR = r"D:\facial recognization\facial emotion dataset\train"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

def load_data(data_dir=DATA_DIR, emotions=EMOTIONS, img_size=IMG_SIZE):
    X, y = [], []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')
    try:
        for idx, emotion in enumerate(emotions):
            folder = os.path.join(data_dir, emotion)
            if not os.path.isdir(folder):
                print(f"Warning: folder not found: {folder}")
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
            if not files:
                print(f"Warning: no image files found in {folder}")
                continue
            for img_name in tqdm(files, desc=f'Loading {emotion}', unit='img'):
                img_path = os.path.join(folder, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # unreadable or not an image file
                        continue
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    y.append(idx)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                    continue
    except KeyboardInterrupt:
        print('\nLoading interrupted by user.')

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.expand_dims(X, -1)
    y = to_categorical(y, num_classes=len(emotions))

    # Print per-class counts to help debugging
    try:
        counts = np.bincount(np.argmax(y, axis=1), minlength=len(emotions))
        for emo, c in zip(emotions, counts):
            print(f"{emo}: {c}")
    except Exception:
        pass

    return X, y

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # If dataset is organized under a 'train' subfolder (common layout), prefer that
    data_dir = DATA_DIR
    train_sub = os.path.join(DATA_DIR, 'train')
    if os.path.isdir(train_sub):
        data_dir = train_sub

    X, y = load_data(data_dir=data_dir)
    if X.size == 0:
        print('No data loaded. Check DATA_DIR and folder names.')
        return
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Data shapes:', X_train.shape, X_val.shape)

    model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.summary()

    # Add callbacks to prevent overfitting and save the best model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Compute class weights to help with class imbalance
    num_classes = y_train.shape[1]
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_integers)
    class_weights = dict(enumerate(class_weights))

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    batch_size = 64
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    steps_per_epoch = max(1, len(X_train) // batch_size)

    # Train the model (adjust epochs as needed)
    epochs = 1  # short smoke run; increase to 25-50 for full training
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Save final model
    model.save('emotion_recognition_model.h5')
    print('Model training complete and saved to emotion_recognition_model.h5')

    # --- Post-training evaluation ---
    # Load best model if available
    best_path = 'best_model.h5'
    if os.path.isfile(best_path):
        eval_model = tf.keras.models.load_model(best_path)
        print(f"Loaded best model from {best_path} for evaluation")
    else:
        eval_model = model

    # Evaluate on validation set
    val_loss, val_acc = eval_model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc*100:.2f}%  (loss: {val_loss:.4f})")

    # Confusion matrix and classification report
    y_pred = eval_model.predict(X_val)
    y_true = np.argmax(y_val, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print('\nClassification report:')
    print(classification_report(y_true, y_pred_labels, target_names=EMOTIONS, digits=4))

    cm = confusion_matrix(y_true, y_pred_labels)
    print('Confusion matrix:\n', cm)

    # Plot training history (loss and accuracy)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print('Saved training history to training_history.png')


if __name__ == "__main__":
    main()  # Train and save the model

def predict_emotion(model, img_path):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image {img_path}")
        return
    # Resize and normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)   # batch dimension
    img = np.expand_dims(img, axis=-1)  # channel dimension

    # Predict
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = EMOTIONS[emotion_index]
    print(f"Predicted Emotion: {emotion}")

# Example usage (load model first):
# model = tf.keras.models.load_model('emotion_recognition_model.h5')
# predict_emotion(model, 'test_image.jpg')

def webcam_emotion_recognition(model):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)
            emotion_index = np.argmax(prediction)
            emotion = EMOTIONS[emotion_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# To run webcam or evaluation, load the saved model and call functions explicitly.
# Example:
# model = tf.keras.models.load_model('emotion_recognition_model.h5')
# webcam_emotion_recognition(model)
# val_loss, val_acc = model.evaluate(X_val, y_val)
# print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ImageDataGenerator is imported at the top and used in training
