import cv2
import numpy as np


def draw_yolo_results(image,results):

    if results is None:
        return image

    if len(results.shape) == 1:
        results = results[None]

    total = len(results)
    if total:
        for i in range(total):
            xmin,ymin,xmax,ymax,conf,label = results[i]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            label = int(label)
            if label == 0:
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(image,f"{label}-{conf:.2f}",(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    return image


if __name__ == '__main__':
    t1 = np.array([1,2,3,4,5,6])
    t2 = np.array([[1,2,3,4,5,6],[9,8,7,6,5,4]])
    print(len(t1.shape),len(t2.shape))

    if len(t1.shape) == 1:
        t1 = t1[None]
    print(len(t1.shape), len(t2.shape))
    print(t1)