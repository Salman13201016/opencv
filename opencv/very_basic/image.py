


import cv2
image = cv2.imread('1.jpg',-1)
height = image.shape[0]*2
width  = image.shape[1]*3
half = cv2.resize(image, (int(width),int(height)))

print(image.shape)
print(half.shape)

# count_pix = cv2.countNonZero(image)
cv2.imshow("python half ", half)

cv2.imshow("python Original ", image)
# print(image.shape)


# cv2.imshow("Python Image",image)

cv2.waitKey(0)