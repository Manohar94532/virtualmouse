import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Streamlit App Title
st.title("AI Virtual Mouse with Hand Gestures")

# Sidebar Settings
st.sidebar.title("Settings")
sensitivity = st.sidebar.slider("Mouse Sensitivity", 1, 10, 5)

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Frame Placeholder
FRAME_WINDOW = st.image([])

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Screen Dimensions
screen_width, screen_height = pyautogui.size()

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    return results

def get_coordinates(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x = int(index_finger_tip.x * screen_width)
    y = int(index_finger_tip.y * screen_height)
    return x, y

def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access the camera!")
        break

    # Flip the frame for natural interaction
    frame = cv2.flip(frame, 1)

    # Process the frame for hand tracking
    results = process_frame(frame)

    # Draw hand landmarks
    draw_landmarks(frame, results)

    # Control the mouse using index finger
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x, y = get_coordinates(hand_landmarks)

            # Move the mouse with sensitivity scaling
            scaled_x = np.clip(x, 0, screen_width)
            scaled_y = np.clip(y, 0, screen_height)
            pyautogui.moveTo(scaled_x, scaled_y, duration=0.01 * (11 - sensitivity))

    # Display the frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
