import numpy as np
import math

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

def getProjectionMatrix1(K,H):
	H = np.linalg.inv(H)
	h1=H[:,0]
	h2=H[:,1]

	K=np.transpose(K)

	K_inv=np.linalg.inv(K)
	a=np.dot(K_inv,h1)
	c=np.dot(K_inv,h2)
	lamda=1/((np.linalg.norm(a)+np.linalg.norm(c))/2)

	Bhat=np.dot(K_inv,H)

	if np.linalg.det(Bhat)>0:
		B=1*Bhat
	else:
		B=-1*Bhat

	b1=B[:,0]
	b2=B[:,1]
	b3=B[:,2]
	r1=lamda*b1
	r2=lamda*b2
	r3=np.cross(r1,r2)
	t=lamda*b3

	P=np.dot(K,(np.stack((r1,r2,r3,t), axis=1)))

	return P