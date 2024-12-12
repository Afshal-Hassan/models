import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open('./model_3.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Unable to capture frame")
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            
            print(f"Prediction: {prediction}")

        else :
            print("No hand detected")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
