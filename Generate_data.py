import cv2
import mediapipe as mp
import numpy as np

#  0 : 'rock', 1 : 'paper', 2 : 'scissors', 3 : 'FY'

font = cv2.FONT_HERSHEY_DUPLEX
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
file = []

cap = cv2.VideoCapture(0)

def save():
    global data, file
    if len(file) == 0:
        file = data
    else:
        file = np.vstack((file, data))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img,1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_RGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:

            joint = np.zeros((21,3)) # hand landmark는 21개의 point가 존재하고, (x,y,z)좌표로 표현된다.

            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            start_idx = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,5,9,13]
            end_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,9,13,17] # 23개의 edge가 존재
            v_start = joint[start_idx,:]
            v_end = joint[end_idx,:]

            v = v_end - v_start
            norm = np.linalg.norm(v, axis=1)[:, np.newaxis]
            v = v / norm

            dot = np.einsum('nt, nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,4,5,8,9,8,9,12,13,12,13,16,17],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,20,20,20,20,21,21,21,21,22,22,22,22],:]) # (0번 edge와 1번 edge), (1번 edge와 2번 edge)...의 dot product

            angle = np.arccos(dot) 
            angle = np.degrees(angle)
            data = np.array([angle], dtype = np.float32)
            data = np.append(data, 3)
            mp_drawing.draw_landmarks(img, 
                            res, 
                            mp_hands.HAND_CONNECTIONS,           
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                            )

    if cv2.waitKey(1) == ord('s'):
        save()

    cv2.putText(img, "Number of data : {}".format(len(file)) , (20, 90), font, 2, (100,155,155), 2, cv2.LINE_AA)

    cv2.imshow('Generate Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

np.savetxt('data/gesture_train_FY.csv', file, delimiter=',')