



import face_recognition as fr # To recognize faces
import numpy as np # To handle all lists/arrays
import cv2 # To capture webcam footage
import os # To handle all matters relating to folders, paths, image/file names, etc.

#F:\innovation skills\project\fr\opencv\opencv\face_recognition
faces_path = "F:\\innovation skills\\project\\fr\\opencv\\opencv\\face_recognition\\image"
# print(os.listdir(faces_path))
def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []
    for i, name in enumerate(face_names):
         face = fr.load_image_file(f"{faces_path}\\{name}")
         face_encodings.append(fr.face_encodings(face)[0])
         face_names[i] = name.split(".")[0]
    return face_encodings,face_names

face_encodings, face_names = get_face_encodings()

# Reference to webcam

video = cv2.VideoCapture(0)

while True:
    success, image = video.read()
    print(image)
    resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
    # Converting current frame to RGB, since that's what the face recognition module uses
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_image)
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    for face_encoding_test, face_location in zip(unknown_encodings, face_locations):
        result = fr.compare_faces(face_encodings,face_encoding_test,0.4)
        if True in result:
            name = face_names[result.index(True)]
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left*2, top*2), (right*2, bottom*2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left*2, bottom*2 + 20), font, 0.8, (255, 255, 255), 1)
        else:
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left*2, top*2), (right*2, bottom*2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, "unknown", (left*2, bottom*2 + 20), font, 0.8, (255, 255, 255), 1)
    cv2.imshow("frame", image)
    cv2.waitKey(1)



