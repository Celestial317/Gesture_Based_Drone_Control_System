import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Drone command mapping
gesture_mapping = {
    0: "Throttle Up (Palm Open)",
    1: "Throttle Down (Fist)",
    2: "Forward (Index Finger Up)",
    3: "Backward (Victory Sign)",
    4: "Yaw Left (3 Fingers Up)",
    5: "Yaw Right (4 Fingers Up)",
    6: "Rotate (Index and Little Finger Up)",
    7: "Hover (Thumbs Up)",
    8: "Unknown Gesture (Hover)"
}

default_command = "Hover (Thumbs Up)"

# Loads dataset from the mentioned path and then preprocessing is done on it
def load_dataset(dataset_path):
    images, labels, hand_types = [], [], []
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) / 255.0  # Normalization of each pixel of the imagae data
            label = int(file.split("_")[1])  
            hand_type = 0 if "left" in file else 1  # 0 = Left, 1 = Right
            images.append(img)
            labels.append(label)
            hand_types.append(hand_type)
    return np.array(images), np.array(labels), np.array(hand_types)


dataset_path = "/content/datasets/hand-keypoints/data.yaml"
images, labels, hand_types = load_dataset(dataset_path)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(gesture_mapping))

# Split data into 80-20 ratio
X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
    images, labels, hand_types, test_size=0.2, random_state=42
)

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


gesture_model = build_cnn((128, 128, 3), len(gesture_mapping))
gesture_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)


gesture_model.save("gesture_recognition_model.h5")
gesture_model = tf.keras.models.load_model("gesture_recognition_model.h5")


def recognize_gesture(frame):
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = gesture_model.predict(img)
    gesture_id = np.argmax(predictions)
    return gesture_mapping.get(gesture_id, default_command)


def control_drones(left_hand_gesture, right_hand_gesture):
    print(f"Left Drone Command: {left_hand_gesture}")
    print(f"Right Drone Command: {right_hand_gesture}")

cap = cv2.VideoCapture(0)

print("Starting real-time hand gesture recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    left_hand_frame = frame[:, :frame.shape[1]//2] 
    right_hand_frame = frame[:, frame.shape[1]//2:]

   
    left_hand_gesture = recognize_gesture(left_hand_frame)
    right_hand_gesture = recognize_gesture(right_hand_frame)

    control_drones(left_hand_gesture, right_hand_gesture)


    cv2.putText(frame, f"Left: {left_hand_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Right: {right_hand_gesture}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
