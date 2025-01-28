import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

# Create a blank canvas for drawing
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Colors and parameters
draw_color = (0, 0, 255)  # Red for drawing
eraser_color = (0, 0, 0)  # Black for erasing
thickness = 10  # Line thickness
eraser_thickness = 40
drawing = False
prev_x, prev_y = None, None

# Webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame (optional for debugging)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the coordinates of the index fingertip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_index, y_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Check if thumb and index finger are close (pinch gesture)
            distance = np.sqrt((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2)

            # Eraser mode if pinch gesture is detected
            if distance < 40:  # Adjust the threshold as needed
                drawing = True
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), eraser_color, eraser_thickness)
            else:
                # Drawing mode when only the index finger is extended
                drawing = True
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), draw_color, thickness)

            # Update the previous points
            prev_x, prev_y = x_index, y_index
    else:
        # Reset the previous points if no hand is detected
        prev_x, prev_y = None, None

    # Merge the canvas with the webcam feed
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display instructions
    cv2.putText(frame, "Press 'c' to clear screen", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Use index finger to draw", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Pinch (thumb + index) to erase", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the output
    cv2.imshow("Virtual Writing", frame)

    # Key bindings
    key = cv2.waitKey(1)
    if key == ord('c'):  # Clear the canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    elif key == 27:  # Exit on 'Esc'
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
