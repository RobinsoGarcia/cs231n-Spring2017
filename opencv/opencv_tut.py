import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

cap = cv2.VideoCapture('/home/alien/Documents/opencv/videos/pipeinsp.avi')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),10)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5))  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    '''

    img = cv2.Canny(img,140,200)
    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
