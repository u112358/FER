import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
fp = open('../data/indoor_003.pts','r')
print fp.readline()
print fp.readline()
print fp.readline()

xy=[]
for i in range(68):
    [x_t,y_t] = fp.readline().split(' ')
    xy.append((float(x_t),float(y_t)))


im = Image.open('../data/indoor_003.png')


# put a red dot, size 40, at 2 locations:
rec = (xy[0][0]-100, xy[0][1]-100, xy[0][0]+100,xy[0][1]+100)
cropped_im = im.crop(rec)
cropped_im.show()

draw = ImageDraw.Draw(im)
for i in range(68):
    draw.point(xy)
im.show()
print xy