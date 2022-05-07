import cv2
import mediapipe as mp
import numpy as np
import os

#    0 : 'rock', 1 : 'paper', 2 : 'scissors', 3 : 'FY'

font = cv2.FONT_HERSHEY_DUPLEX
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

if not os.path.isfile('data/gesture_train.csv'):
    file = np.genfromtxt('data/gesture_train_R.csv', delimiter=',')
    file_P = np.genfromtxt('data/gesture_train_P.csv', delimiter=',')
    file_S = np.genfromtxt('data/gesture_train_S.csv', delimiter=',')
    file_FY = np.genfromtxt('data/gesture_train_FY.csv', delimiter=',')

    file = np.vstack((file, file_P))
    file = np.vstack((file, file_S))
    file = np.vstack((file, file_FY))

    np.savetxt('data/gesture_train.csv', file, delimiter=',')

else:
    file = np.genfromtxt('data/gesture_train.csv', delimiter=',')

angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img,1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)

    if results.multi_hand_landmarks is not None:
        labels = []
        joints = []
        confidence = []
        for result in results.multi_hand_landmarks:
            joint = np.zeros((21,3)) # hand landmark는 21개의 point가 존재하고, (x,y,z)좌표로 표현된다.

            for j, lm in enumerate(result.landmark):
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
            ret, output, neighbor, distance = knn.findNearest(data, 3)
            idx = int(output[0][0])
            score = 0
            if idx == 3:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.02, fy=0.02, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img
                outputs = neighbor[0]
                score = sum(outputs==3) / len(outputs)
                
            labels.append(idx)
            joints.append(joint)
            confidence.append(score)

        if len(labels) == 2:
            cv2.putText(img, "RPS GAME".format(len(file)) , (400, 90), font, 3, (0,75,155), 3, cv2.LINE_AA)
            x1, y1 = tuple((joints[0].min(axis=0)[:2] * [img.shape[1], img.shape[0]]).astype(int))
            x2, y2 = tuple((joints[0].max(axis=0)[:2] * [img.shape[1], img.shape[0]]).astype(int))

            x3, y3 = tuple((joints[1].min(axis=0)[:2] * [img.shape[1], img.shape[0]]).astype(int))
            x4, y4 = tuple((joints[1].max(axis=0)[:2] * [img.shape[1], img.shape[0]]).astype(int))            

            alpha1, beta1 = x1 + (x2 - x1) // 2, y1
            alpha2, beta2 = x3 + (x4 - x3) // 2, y3

            if labels[0] == 3:
                cv2.putText(img, "LANGUAGE!", (alpha1, beta1), font, 2, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(img, "Confidence : {}".format(confidence[0]), (alpha1+200, beta1+20), font, 0.5, (0,0,200), 1, cv2.LINE_AA)
            if labels[1] == 3:
                cv2.putText(img, "LANGUAGE!", (alpha2, beta2), font, 2, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(img, "Confidence : {}".format(confidence[1]), (alpha2+200, beta2+20), font, 0.5, (0,0,200), 1, cv2.LINE_AA)
            elif labels[1] == 0: # R
                if labels[0] == 0: # R
                    cv2.putText(img, "DRAW!", (alpha1, beta1), font, 2, (0,155,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "DRAW!", (alpha2, beta2), font, 2, (0,155,155), 2, cv2.LINE_AA)
                if labels[0] == 1: # P
                    cv2.putText(img, "WIN!" , (alpha1, beta1), font, 2, (0,155,0), 2, cv2.LINE_AA)
                    cv2.putText(img, "LOSE!", (alpha2, beta2), font, 2, (0,0,155), 2, cv2.LINE_AA)                
                if labels[0] == 2: # S
                    cv2.putText(img, "LOSE!", (alpha1, beta1), font, 2, (0,0,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "WIN!" , (alpha2, beta2), font, 2, (0,155,0), 2, cv2.LINE_AA)     
            elif labels[1] == 1: # P
                if labels[0] == 0: # R
                    cv2.putText(img, "LOSE!", (alpha1, beta1), font, 2, (0,0,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "WIN!" , (alpha2, beta2), font, 2, (0,155,0), 2, cv2.LINE_AA)   
                if labels[0] == 1: # P
                    cv2.putText(img, "DRAW!", (alpha1, beta1), font, 2, (0,155,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "DRAW!", (alpha2, beta2), font, 2, (0,155,155), 2, cv2.LINE_AA)
                if labels[0] == 2: # S
                    cv2.putText(img, "WIN!" , (alpha1, beta1), font, 2, (0,155,0), 2, cv2.LINE_AA)
                    cv2.putText(img, "LOSE!", (alpha2, beta2), font, 2, (0,0,155), 2, cv2.LINE_AA)     
            elif labels[1] == 2: # S
                if labels[0] == 0: # R
                    cv2.putText(img, "WIN!" , (alpha1, beta1), font, 2, (0,155,0), 2, cv2.LINE_AA)
                    cv2.putText(img, "LOSE!", (alpha2, beta2), font, 2, (0,0,155), 2, cv2.LINE_AA)     
                if labels[0] == 1: # P
                    cv2.putText(img, "LOSE!", (alpha1, beta1), font, 2, (0,0,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "WIN!" , (alpha2, beta2), font, 2, (0,155,0), 2, cv2.LINE_AA)   
                if labels[0] == 2: # S
                    cv2.putText(img, "DRAW!", (alpha1, beta1), font, 2, (0,155,155), 2, cv2.LINE_AA)
                    cv2.putText(img, "DRAW!", (alpha2, beta2), font, 2, (0,155,155), 2, cv2.LINE_AA)

    cv2.imshow('RPS', img)
    if cv2.waitKey(1) == ord('q'):
        break
