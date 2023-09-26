import face_recognition as fr  # To recognize faces
import cv2  # To read and display images
import os  # To handle file paths

# Path to the image you want to perform face recognition on
image_path = "F:\\innovation skills\\project\\fr\\opencv\\opencv\\face_recognition\\image\\sagor.jpg"

# Path to the folder containing the face images for comparison
faces_folder = "F:\\innovation skills\\project\\fr\\opencv\\opencv\\face_recognition\\image"

def get_face_encodings():
    face_names = os.listdir(faces_folder)
    face_encodings = []
    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_folder}/{name}")
        face_encodings.append(fr.face_encodings(face)[0])
        face_names[i] = name.split(".")[0]
    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

# Load the image for face recognition
image = fr.load_image_file(image_path)

# Convert the image to RGB, as face_recognition library requires RGB format
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find face locations in the image
face_locations = fr.face_locations(rgb_image)

# Encode the faces found in the image
unknown_encodings = fr.face_encodings(rgb_image, face_locations)

for face_encoding_test, face_location in zip(unknown_encodings, face_locations):
    result = fr.compare_faces(face_encodings, face_encoding_test, 0.4)
    if True in result:
        name = face_names[result.index(True)]
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left, bottom + 20), font, 0.8, (255, 255, 255), 1)
    else:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, "unknown", (left, bottom + 20), font, 0.8, (255, 255, 255), 1)

# Display the image with face recognition results
cv2.imshow("Image with Face Recognition", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
