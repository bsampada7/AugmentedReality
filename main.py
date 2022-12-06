import cv2 as cv
import numpy as np
from objLoader import *
from cube import *
from sift import sift_class
from utils import *
import time

# FLAGS
MIN_MATCH_COUNT = 5
DRAW_RECTANGLE = True
RENDER_CUBE = True
RENDER_OBJ = False

# CONSTANTS
# Calibrated instrinsic matrix value for the camera
camera_intrinsic_mat = np.array(
    [[553.70386558, 0.0, 306.31746871],
     [0.0, 551.3634195, 211.01781909],
     [  0.0, 0.0, 1.0]
    ]
    )

def main():
    # Read the reference tag image for performing ar on it
    tag = cv.imread('data/tag-miami.jpg',cv.IMREAD_GRAYSCALE) 
    tagH, tagW = tag.shape

    # Image to render onto the face of the cube
    img = cv2.imread('data/me.jpg')
    img = cv2.resize(img, (260,470), interpolation = cv2.INTER_CUBIC )

    # Initialize the SIFT class for feature matching and description
    sift = sift_class(tag)
    
    # Load the model to render 
    if(RENDER_OBJ):
        obj = three_d_object('models/miami-inner.obj', True)

    # Start webcam
    cam = cv.VideoCapture(0)

    while True:
        # Read the current frame
        ret, frame = cam.read()
        if not ret:
            print("Unable to capture video")
            return 
        matches = sift.getMatches(frame)

        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCH_COUNT:
            # differenciate between source points and destination points
            src_pts = np.float32([sift.kp_tag[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([sift.kp_frame[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # compute Homography
            h_mat, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if DRAW_RECTANGLE:
                # Draw a rectangle that marks the found tag in the frame
                h, w = tag.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                try:
                    # project corners into frame
                    dst = cv.perspectiveTransform(pts, h_mat)
                except:
                    continue
                
                # connect them with lines  
                frame = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)  

            # if a valid homography matrix was found render cube on the tag plane
            if h_mat is not None:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = getProjectionMatrix(camera_intrinsic_mat, h_mat) 
                
                if(RENDER_OBJ):
                    # project cube or model and flipped for better control
                    frame = np.flip(augment(frame, obj, projection, tag), axis = 1) 
                elif(RENDER_CUBE):
                    # draw the cube onto the frame
                    corner_points = []
                    corner_points.append([int(dst[0][0][0]),int(dst[0][0][1])])
                    corner_points.append([int(dst[1][0][0]),int(dst[1][0][1])])
                    corner_points.append([int(dst[2][0][0]),int(dst[2][0][1])])
                    corner_points.append([int(dst[3][0][0]),int(dst[3][0][1])])
                    
                    src = np.array([[[0,0]], [[tagH, 0]], [[tagH, tagW]], [[0, tagW]]])

                    h_mat, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)

                    new_corners=cubePoints(corner_points, h_mat, projection, 10)
                    frame=drawCube(corner_points, new_corners,frame,(255,255,255),(0,0,0),False)

                    dst = np.array([[new_corners[0]], [new_corners[1]], [new_corners[2]], [new_corners[3]]])

                    h_mat, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)

                    frame1 = warp(np.linalg.inv(h_mat),img, frame.shape[0], frame.shape[1])

                    frame2 = blank_region(frame,np.int32(dst),0)
                    frame = cv2.bitwise_or(frame1,frame2)

            
        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCH_COUNT))

        # Show the result
        cv.imshow('frame', frame)

        # Terminate on Esc
        key = cv.waitKey(20)
        if key == 27:
            break

    cam.release()
    
    cv.destroyAllWindows()
    return 0


main()