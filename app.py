from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json

app = FastAPI()

# Load the model
model_dict = pickle.load(open('./model_3.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels and thresholds
labels_dict = {0: 'A', 1: 'B', 2: 'L'}
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence

# API's
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the image frame from the frontend
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            img_base64 = frame_data.get("image")

            # Decode the base64 image
            if not img_base64:
                await websocket.send_json({"error": "No image received"})
                continue

            image_bytes = base64.b64decode(img_base64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Process landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Check if the detected hand is the right hand
                    if hand_class.classification[0].label != "Right":
                        await websocket.send_json({"message": "Only predictions for the right hand are allowed"})
                        continue

                    x_, y_, data_aux = [], [], []
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    if len(data_aux) == 42:
                        prediction = model.predict([np.asarray(data_aux)])
                        probabilities = model.predict_proba([np.asarray(data_aux)])
                        confidence = max(probabilities[0])

                        if confidence >= CONFIDENCE_THRESHOLD:
                            predicted_label = labels_dict.get(prediction[0], "Unknown")
                            await websocket.send_json({
                                "prediction": prediction[0],
                                "confidence": round(confidence, 2)
                            })
                        else:
                            await websocket.send_json({
                                "message": "Prediction confidence too low",
                                "confidence": round(confidence, 2)
                            })
            else:
                await websocket.send_json({"message": "No hand detected"})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
