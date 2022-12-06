import numpy as np
import cv2

def cubePoints(corners, H, P, height):
    new_corners=[]
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
        
    H_c = np.stack((np.array(x),np.array(y),np.ones(len(x))))
    
    sH_w=np.dot(H,H_c)
    
    H_w=sH_w/sH_w[2]
    
    P_w=np.stack((H_c[0],H_c[1],np.full(4,height),np.ones(4)),axis=0)
    
    sP_c=np.dot(P,P_w)
    
    P_c=sP_c/(sP_c[2])
    
    for i in range(4):
        new_corners.append([int(P_c[0][i]),int(P_c[1][i])])
        
    return new_corners

def drawCube(tagcorners, new_corners,frame,face_color,edge_color,flag):
    thickness=5
    if not flag:
        contours = makeContours(tagcorners,new_corners)
        for contour in contours:
            cv2.drawContours(frame,[contour],-1,face_color,thickness=-1)
            
    for i, point in enumerate(tagcorners):
        cv2.line(frame, tuple(np.int32(point)), tuple(np.int32(new_corners[i])), edge_color, thickness)
        
    for i in range (4):
        if i==3:
            cv2.line(frame,tuple(np.int32(tagcorners[i])),tuple(np.int32(tagcorners[0])),edge_color,thickness)
            cv2.line(frame,tuple(np.int32(new_corners[i])),tuple(np.int32(new_corners[0])),edge_color,thickness)
        else:
            cv2.line(frame,tuple(np.int32(tagcorners[i])),tuple(np.int32(tagcorners[i+1])),edge_color,thickness)
            cv2.line(frame,tuple(np.int32(new_corners[i])),tuple(np.int32(new_corners[i+1])),edge_color,thickness)
            
    return frame


def makeContours(corners1,corners2):
	contours = []
	for i in range(len(corners1)):
		if i==3:
			p1 = corners1[i]
			p2 = corners1[0]
			p3 = corners2[0]
			p4 = corners2[i]
		else:
			p1 = corners1[i]
			p2 = corners1[i+1]
			p3 = corners2[i+1]
			p4 = corners2[i]
		contours.append(np.array([p1,p2,p3,p4], dtype=np.int32))
	contours.append(np.array([corners1[0],corners1[1],corners1[2],corners1[3]], dtype=np.int32))
	contours.append(np.array([corners2[0],corners2[1],corners2[2],corners2[3]], dtype=np.int32))

	return contours