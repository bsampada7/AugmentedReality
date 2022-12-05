import argparse

import cv2 as cv
import numpy as np
import math

MIN_MATCH_COUNT = 10

DRAW_RECTANGLE = True

def main():
    """
    This functions loads the target surface image,
    """
    # random value for camera intrinsic matrix
    camera_intrinsic_mat = np.array(
        [[553.70386558, 0.0, 306.31746871],
         [0.0, 551.3634195, 211.01781909],
         [  0.0, 0.0, 1.0]
        ]
        )

    tag = cv.imread('data/tag.jpg',cv.IMREAD_GRAYSCALE) 

    sift = cv.SIFT_create()
    kp_tag, des_tag = sift.detectAndCompute(tag,None)
    bf = cv.BFMatcher()

    cam = cv.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cam.read()
        if not ret:
            print("Unable to capture video")
            return 
        # find and draw the keypoints of the frame
        kp_frame, des_frame = sift.detectAndCompute(frame,None)
        # match frame descriptors with model descriptors
        matches = bf.knnMatch(des_tag,des_frame,k=2)

        # Apply ratio test
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                goodMatches.append([m])

        # sort them in the order of their distance
        # the lower the distance, the better the match
        goodMatches = sorted(goodMatches, key=lambda x: x[0].distance)

        # compute Homography if enough matches are found
        if len(goodMatches) > MIN_MATCH_COUNT:
            print("matches found - %d" % (len(goodMatches)))
            # differenciate between source points and destination points
            src_pts = np.float32([kp_tag[m[0].queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m[0].trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            # compute Homography
            h_mat, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if DRAW_RECTANGLE:
                # Draw a rectangle that marks the found tag in the frame
                h, w = tag.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv.perspectiveTransform(pts, h_mat)
                # connect them with lines  
                frame = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)  
            # # if a valid homography matrix was found render cube on the tag plane
            # if h_mat is not None:
            #     try:
            #         # obtain 3D projection matrix from homography matrix and camera parameters
            #         projection = getProjectionMatrix(camera_intrinsic_mat, h_mat)  
            #         # project cube or model
            #         frame = render(frame, obj, projection, model, False)
            #         #frame = render(frame, model, projection)
            #     except:
            #         pass
            # draw first 10 matches.
            # frame = cv.drawMatches(tag, kp_tag, frame, kp_frame, matches[:10], 0, flags=2)
            frame = cv.drawMatchesKnn(tag,kp_tag,frame,kp_frame,goodMatches[:MIN_MATCH_COUNT],None,flags=2)
            # show result
            
        else:
            print("Not enough matches found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
        cv.imshow('frame', frame)
        key = cv.waitKey(20)
        if key == 27:
            break

    cam.release()
    cv.destroyAllWindows()
    return 0

def getProjectionMatrix(A, H):
	#finds r3 and appends
	# A is the intrinsic mat, and H is the homography estimated
	H = np.float64(H) #for better precision
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

# def drawCube:


main()