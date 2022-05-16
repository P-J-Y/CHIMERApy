import numpy as np
import cv2

def trackCH(lastContours,contour,s):
    match = False
    p = np.zeros(s)
    cv2.drawContours(p, [contour, ], -1, 255, -1)
    a = np.where(p == 255)[0].reshape(-1, 1)
    b = np.where(p == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([a, b], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    for i in range(len(lastContours)):
        lastContour = lastContours[i]
        p = np.zeros(s)
        cv2.drawContours(p,[lastContour,],-1,255,-1)
        a = np.where(p==255)[0].reshape(-1,1)
        b = np.where(p==255)[1].reshape(-1,1)
        coordinate = np.concatenate([a,b],axis=1).tolist()
        insidei = [tuple(x) for x in coordinate]
        overlap = [k for k in insidei if k in inside]
        if len(overlap)/len(insidei) > 0.8:
            match = True
            return i,match
    return -1,match
