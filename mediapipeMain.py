import cv2
import mediapipe as mp
import json
from collections import Counter
import speech_recognition as sr


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

gesture_map = {
    "00000": {"gesture": "fist", "action": "Throttle Down", "movement": "DOWN"},
    "11111": {"gesture": "hand_palm", "action": "Throttle Up", "movement": "UP"},
    "01000": {"gesture": "index_finger_up", "action": "Move Forward", "movement": "FORWARD"},
    "01100": {"gesture": "victory", "action": "Move Backward", "movement": "BACKWARD"},
    "01110": {"gesture": "three_fingers_up", "action": "Yaw Left", "movement": "LEFT"},
    "01111": {"gesture": "four_fingers_up", "action": "Yaw Right", "movement": "RIGHT"},
    "01001": {"gesture": "index_and_little_finger_up", "action": "Rotate", "movement": "ROTATE"},
    "10000": {"gesture": "thumbs_up", "action": "Hover", "movement": "HOVER"}
}

default_action = {"gesture": "unknown", "action": "Hover", "movement": "HOVER", "fingers": [0, 0, 0, 0, 0]}

def interpret_gesture(hand_landmarks, hand_label):
    landmarks = hand_landmarks.landmark
    fingers = [0, 0, 0, 0, 0]

    if hand_label == "Left":
        if landmarks[4].x > landmarks[3].x:
            fingers[0] = 1
    else:
        if landmarks[4].x < landmarks[3].x:
            fingers[0] = 1

    if landmarks[8].y < landmarks[6].y:
        fingers[1] = 1

    if landmarks[12].y < landmarks[10].y:
        fingers[2] = 1

    if landmarks[16].y < landmarks[14].y:
        fingers[3] = 1

    if landmarks[20].y < landmarks[18].y:
        fingers[4] = 1

    finger_state = ''.join(map(str, fingers))
    return fingers, gesture_map.get(finger_state, default_action)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_count = 0
    gesture_window_left = []
    gesture_window_right = []
    current_gesture_output_left = default_action
    current_gesture_output_right = default_action

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_label.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers, gesture = interpret_gesture(hand_landmarks, hand_label)
                hand = 0 if hand_label == "Left" else 1
                gesture_output = {
                    "Hand": hand,
                    "fingers": fingers,
                    **gesture
                }

                if hand_label == "Left":
                    gesture_window_left.append(gesture_output)
                else:
                    gesture_window_right.append(gesture_output)

        frame_count += 1

        if frame_count == 10:
            if gesture_window_left:
                gesture_counts_left = Counter([json.dumps(g) for g in gesture_window_left])
                most_common_gesture_left, count_left = gesture_counts_left.most_common(1)[0]
                if count_left > 8:
                    current_gesture_output_left = json.loads(most_common_gesture_left)
                else:
                    current_gesture_output_left = default_action
                gesture_window_left = []

            if gesture_window_right:
                gesture_counts_right = Counter([json.dumps(g) for g in gesture_window_right])
                most_common_gesture_right, count_right = gesture_counts_right.most_common(1)[0]
                if count_right > 8:
                    current_gesture_output_right = json.loads(most_common_gesture_right)
                else:
                    current_gesture_output_right = default_action
                gesture_window_right = []

            frame_count = 0
        
        if current_gesture_output_left:
            cv2.putText(frame, f"Action1: {current_gesture_output_left['action']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            print(f"Action1: {current_gesture_output_left['action']}, Fingers: {current_gesture_output_left['fingers']}")
        if current_gesture_output_right:
            cv2.putText(frame, f"Action2: {current_gesture_output_right['action']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Action2: {current_gesture_output_right['action']}, Fingers: {current_gesture_output_right['fingers']}")

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
