import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('smnist.h5')
mphands = mp.solutions.hands
hands = mphands.Hands()
cap = cv2.VideoCapture(0)
s, frame = cap.read()
h, w, c = frame.shape
x_max = 0
y_max = 0
x_min = w
y_min = h
while True:
    s, frame = cap.read()
    analysis_frame=frame
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 15
            y_max += 15
            x_min -= 15
            x_max += 15
            if(y_max-y_min>x_max-x_min):
                cv2.rectangle(frame, (int(x_min-(y_max-y_min)/2), int(y_min)), (int(x_min+(y_max-y_min)/2), int(y_max)), (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (int(x_min), int(y_min-(x_max-x_min)/2)), (x_max, int(y_min+(x_max-x_min)/2)), (255, 255, 255), 1)
    cv2.imshow("Frame", frame)
    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    if k%256==32:
        #space pressed
        #analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        if(y_max-y_min>x_max-x_min):
            analysis_frame=analysis_frame[y_min:y_max,x_min:x_min+(y_max-y_min)]
        else:
            analysis_frame=analysis_frame[y_min:y_min+x_max-x_min,x_min:x_max]
        resized = cv2.resize(analysis_frame, (28, 28 ))
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        nlist = []
        rows,cols= image.shape
        for i in range(rows):
            for j in range(cols):
                k = image[i,j]
                nlist.append(int(k))
                print(type(k))
        print(image)
        # testlist=[1,2,3,6,7,8,10,11,15]
        # testdata=pd.DataFrame(testlist).T
        # print(type(testlist[2]))
        datan = pd.DataFrame(nlist).T
        print(datan)
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname
        pixeldata = datan.values
        # pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1,28,28,1)
        
        image = image.reshape(-1,28,28,1)
        prediction = model.predict(pixeldata)
        predarray = np.array(prediction[0])
        print(predarray)
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100*value)
            elif value==high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100*value)
            elif value==high3:
                print("Predicted Character 3: ", key)
                print('Confidence 3: ', 100*value)
        plt.imshow(pixeldata[0],cmap='gray')
        plt.show()

cap.release()
cv2.destroyAllWindows()