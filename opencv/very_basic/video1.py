import cv2

cam = cv2.VideoCapture(0)
i=1
while True:
    check, frame = cam.read()
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    cv2.imwrite('Video'+str(i)+'.jpg',frame)
    i=i+1
cam.release()
cv2.destroyAllWindows()