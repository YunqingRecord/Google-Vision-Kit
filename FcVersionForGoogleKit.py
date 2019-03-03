import cv2
import numpy as np
from keras.models import load_model


def digit_recognition(imgfile):

    global ans
    # load the trained model
    model = load_model('Cnn_Mnist.h5')
    print(model.summary())

    # read the img from the file path
    imgData = cv2.imread(imgfile, 0)
    ret, img1 = cv2.threshold(imgData, 120, 255, cv2.THRESH_BINARY_INV)   # binary thresh
    print(img1)   # np array of the img1

    # resize to the input size of the network
    img = cv2.resize(img1, (28, 28), 0, 0)
    arr = np.asarray(img, dtype="float32")
    print(arr.shape)
    arr = np.reshape(arr, (1, 28, 28, 1))
    print(arr.shape)

    # show the detail of the cv
    cv2.namedWindow("Image_Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Image_Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # output the one-hot result
    results = model.predict(arr)
    for r in results:
        r = r.tolist()  # ndarray to list
        ans = r.index(max(r))  # out put the largest value

    return ans





# load the trained model
model = load_model('Cnn_Mnist.h5')
print(model.summary())
imgfile = 'C:\\Users\\Yunqing\\Desktop\\Google-Vision-Kit\\1.png'
# read the img from the file path
imgData = cv2.imread(imgfile, 0)
ret, img1 = cv2.threshold(imgData, 100, 255, cv2.THRESH_BINARY)   # binary thresh
print(img1)   # np array of the img1

# resize to the input size of the network
img = cv2.resize(img1, (28, 28), 0, 0)
arr = np.asarray(img, dtype="float32")
print(arr.shape)
arr = np.reshape(arr, (1, 28, 28, 1))
print(arr.shape)

# show the detail of the cv
cv2.namedWindow("Image_Test", cv2.WINDOW_NORMAL)
cv2.imshow("Image_Test", img)
cv2.waitKey(0)
# cv2.imwrite("0.png", img)
cv2.destroyAllWindows()

# output the one-hot result
results = model.predict(arr)
for r in results:
    r = r.tolist()  # ndarray to list
    ans = r.index(max(r))  # out put the largest value

print(ans)
