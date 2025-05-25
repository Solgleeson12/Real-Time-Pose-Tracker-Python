import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose solution and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose tracker with detection and tracking confidence thresholds
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
prev_time = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and detect pose
        results = pose.process(image_rgb)

        # Revert image to BGR for rendering
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image_bgr,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Example: print right wrist coordinates
            right_hand = results.pose_landmarks.landmark[16]
            print(f"Right hand → x: {right_hand.x:.2f}, y: {right_hand.y:.2f}, z: {right_hand.z:.2f}")

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(
            image_bgr,
            f'FPS: {int(fps)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Show output
        cv2.imshow('🧍 Real-Time Pose Tracker', image_bgr)

        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
