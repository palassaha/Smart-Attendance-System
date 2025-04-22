import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 6400)
cap.set(4, 480)

imgBG = cv2.imread('Resources/background.png')

folderpath = 'Resources/Modes'
pathlist = os.listdir(folderpath)
imgList = []
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderpath, path)))
# print(len(imgList))

# load encoded file
print("File is being Loaded")
file = open("EncodedFile.p", 'rb')
encodedList_ID = pickle.load(file)
file.close()
encodedList, studentID = encodedList_ID
# print(studentID)
print("File Loading Complete")


while True:
    success, img = cap.read()
    
    image_s = cv2.resize(img, (0,0), None, 0.25, 0.25)
    image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
    
    face_curr_frame = face_recognition.face_locations(image_s)
    encode_curr_frame = face_recognition.face_encodings(image_s, face_curr_frame)
    
    # Overlaying Webcam with bg image
    imgBG[162:162+480, 55:55+640] = img
    imgBG[44:44+633, 808:808+414] = imgList[0]
    
    
    for encode_face, face_loc in zip(encode_curr_frame, face_curr_frame):
        matches = face_recognition.compare_faces(encodedList, encode_face)
        face_dist = face_recognition.face_distance(encodedList, encode_face)
        # print('matches', matches)
        # print('face dist', face_dist)
        
        match_index = np.argmin(face_dist)
        # print("Match Index", match_index)
        
        if matches[match_index]:
            # print("Known face Detected")
            # print(studentID[match_index])
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            imgBG = cvzone.cornerRect(imgBG, bbox, rt=0)
    
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBG)
    cv2.waitKey(1)



