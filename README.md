#  Real-Time Pose Tracker with MediaPipe

Track human body joints in real time using your webcam and Google's [MediaPipe](https://google.github.io/mediapipe/solutions/pose). Perfect for pose analysis, fitness applications, gesture recognition, and more.

---

##  Features

-  Real-time body joint detection via webcam
-  Live skeleton overlay with FPS counter
-  Access to 33 body landmarks (arms, legs, torso, face)
- 🖥 Clean and simple OpenCV GUI
-  Modular code that can be extended for:
  - Pose classification
  - Exercise form feedback
  - Data logging for ML

---

##  Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Packages used:
- `opencv-python`
- `mediapipe`

---

##  Getting Started

Run the tracker using:

```bash
python pose_tracker.py
```

 **Tip:** Press `ESC` to exit the live feed window.

---

##  Example Use Cases

-  Fitness coaching with pose feedback
-  Motion analysis and research
-  Record and annotate exercise form
-  Training data collection for ML models

---

## 🛠 Planned Enhancements

> Help make this project better by contributing!

- [ ] Save pose data to CSV
- [ ] Angle calculation between joints
- [ ] Activity classification (e.g. squats, pushups)
- [ ] Streamlit-based web dashboard
- [ ] Multi-person tracking

---

##  Author

Developed with ❤️ using Python, OpenCV, and MediaPipe.

Feel free to fork, star, and contribute!

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
