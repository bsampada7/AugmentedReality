import cv2 as cv

class sift_class:
    def __init__(self, tag):
        # Initialize the SIFT operator for feature matching and description
        self.sift = cv.SIFT_create()
        self.kp_tag, self.des_tag = self.sift.detectAndCompute(tag,None)
        self.bf = cv.BFMatcher()
    
    def getMatches(self, frame, ratioFactor = 0.5):
        # Find and draw the keypoints of the current frame
        self.kp_frame, self.des_frame = self.sift.detectAndCompute(frame,None)
        try:
            # Match frame descriptors with model descriptors
            matches = self.bf.knnMatch(self.des_tag,self.des_frame,k=2)
        except:
            return []

        # Apply ratio test
        goodMatches = []
        for m,n in matches:
            if m.distance < ratioFactor*n.distance:
                goodMatches.append([m])

        # sort them in the order of their distance
        # the lower the distance, the better the match
        goodMatches = sorted(goodMatches, key=lambda x: x[0].distance)
        return goodMatches
