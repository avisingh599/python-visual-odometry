import numpy as np
import sys
sys.path.insert(0, '/home/avisingh/Downloads/libs/opencv-2.4.11/build/lib/')
import cv2


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 5000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


I_c_l = cv2.imread('./sample_data/image_2/000000.png')
I_l = cv2.cvtColor(I_c_l, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(I_l, mask = None, **feature_params)

I_c_r = cv2.imread('./sample_data/image_3/000000.png')
I_r = cv2.cvtColor(I_c_r, cv2.COLOR_BGR2GRAY)

p1, st, err = cv2.calcOpticalFlowPyrLK(I_l, I_r, p0, None, **lk_params)

p_r = p1[st==1]
p_l = p0[st==1]
#print p_r
#print p_l
projMatr1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 4.538225000000e+01], [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, -1.130887000000e-01], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 3.779761000000e-03]])
projMatr2 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.372877000000e+02], [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 2.369057000000e+00], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.915215000000e-03]])
points_4D = cv2.triangulatePoints(projMatr1, projMatr2, np.transpose(p_l), np.transpose(p_r))
print np.transpose(points_4D)

#cv2.imshow('sample', I_c_r)
#cv2.waitKey(0)
#cv2.destroyAllWindows()