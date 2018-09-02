import tensorflow as tf
import numpy as np

def createColorMap(L=1):
    m = np.zeros((L,3))
    for i in range(0,L,1):
        fourValue = 4. * (i+1) / L
        m[i,0] = max(min(fourValue - 1.5, -fourValue + 4.5, 1),0)
        m[i,1] = max(min(fourValue -  .5, -fourValue + 3.5, 1),0)
        m[i,2] = max(min(fourValue +  .5, -fourValue + 2.5, 1),0)
    return m

def colorNormalize(img,mtd=1):
    img = (img - img.min()) / (img.max() - img.min())
    if mtd == 1:
        return img
    if mtd == 255:
        return img * 255 - 127.5
    if mtd == 2:
        return img * 2 - 1
    return img