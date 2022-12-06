import numpy as np
import math
import cv2

def getProjectionMatrix(A, H):
	# A is the intrinsic mat, and H is the homography estimated
	H = np.float64(H) 
	A = np.float64(A)
	R_12_T = np.linalg.inv(A).dot(H)

	r1 = np.float64(R_12_T[:, 0]) #col1
	r2 = np.float64(R_12_T[:, 1]) #col2
	T = R_12_T[:, 2] #translation
	
	#ideally |r1| and |r2| should be same
	#since there is always some error we take square_root(|r1||r2|) as the normalization factor
	norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))
	
	r3 = np.cross(r1,r2)/(norm)
	R_T = np.zeros((3, 4))
	R_T[:, 0] = r1
	R_T[:, 1] = r2 
	R_T[:, 2] = r3 
	R_T[:, 3] = T
	return A.dot(R_T)


def blank_region(frame,contour,color):
    cv2.drawContours(frame,[contour],-1,(color),thickness=-1)
    return frame

def warp(H,src,h,w):
    # create indices of the destination image and linearize them
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1] 
    map_x = map_x.reshape(h,w).astype(np.float32)
    map_y = map_y.reshape(h,w).astype(np.float32)

    # generate new image
    new_img = np.zeros((h,w,3),dtype="uint8")

    map_x[map_x>=src.shape[1]] = -1
    map_x[map_x<0] = -1
    map_y[map_y>=src.shape[0]] = -1
    map_x[map_y<0] = -1

    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y,new_x])
            y = int(map_y[new_y,new_x])

            if x == -1 or y == -1:
                pass
            else:
                new_img[new_y,new_x] = src[y,x]

    return new_img