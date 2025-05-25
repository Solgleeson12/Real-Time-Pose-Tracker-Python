import cv2
import mediapipe as mp
import time

# Initialize MediaPipe pose and drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)
p_time = 0  # For FPS calculation

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Make detection
        results = pose.process(image_rgb)

        # Draw the pose annotation on the image
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            # Example: Print coordinates of the right hand (landmark 16)
            right_hand = results.pose_landmarks.landmark[16]
            print(f"Right hand coordinates: x={{right_hand.x:.2f}}, y={{right_hand.y:.2f}}, z={{right_hand.z:.2f}}")

        # Calculate and show FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time else 0
        p_time = c_time
        cv2.putText(image_bgr, f'FPS: {{int(fps)}}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the image
        cv2.imshow('MediaPipe Pose Tracker', image_bgr)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
