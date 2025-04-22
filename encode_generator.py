import cv2
import face_recognition
import pickle
import os

folderpath = 'images'
pathlist = os.listdir(folderpath)
imgList = []
studentID = []
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderpath, path)))
    studentID.append(os.path.splitext(path)[0])
# print(len(imgList))
# print(studentID)


def gen_encoding(imagesList):
    encodeList = []
    for img in imagesList:
        # opencv->BGR, face-recognition->RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

print("Encoding Started")
encodedList = gen_encoding(imgList)
# print(encodedList)
encodedList_ID = [encodedList, studentID]
print("Encoding Complete")

file = open("EncodedFile.p", 'wb')
pickle.dump(encodedList_ID, file)
file.close()
print("Encoded File Saved")



