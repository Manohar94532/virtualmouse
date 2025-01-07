import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

class VirtualMouse:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)
    
    def get_coordinates(self, hand_landmarks, frame_width, frame_height):
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_finger_tip.x * frame_width)
        y = int(index_finger_tip.y * frame_height)
        return x, y
    
    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

def main():
    st.title("AI Virtual Mouse with Hand Gestures")
    
    # Sidebar settings
    st.sidebar.title("Controls")
    sensitivity = st.sidebar.slider("Tracking Sensitivity", 1, 10, 5)
    show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", True)
    
    # Initialize virtual mouse and webcam
    vm = VirtualMouse()
    cap = cv2.VideoCapture(0)
    
    # Create placeholder for frame
    frame_placeholder = st.empty()
    
    # Create coordinate display
    coord_placeholder = st.empty()
    
    # Create stop button
    stop_button = st.button("Stop")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera!")
            break
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Process frame
        results = vm.process_frame(frame)
        
        # Draw landmarks if enabled
        if show_landmarks:
            vm.draw_landmarks(frame, results)
        
        # Track hand movement
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y = vm.get_coordinates(hand_landmarks, frame_width, frame_height)
                
                # Display coordinates
                coord_placeholder.text(f"Hand Position - X: {x}, Y: {y}")
                
                # Draw cursor
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        
        # Display frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
    
    cap.release()

if __name__ == "__main__":
    main()
