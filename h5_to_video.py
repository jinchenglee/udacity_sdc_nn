#!/usr/bin/env python3

import numpy as np
import tables
import cv2
import argparse

parser = argparse.ArgumentParser(description="Convert DeepDrive HDF5 file into video file.")
parser.add_argument("h5_file", help="Input HDF5 file from DeepDrive dataset.")
 
args = parser.parse_args()

f = tables.open_file(args.h5_file, 'r')

# Open input image array from HDF5 file
image = np.array(f.root.images)
# Swap indice to be [img_cnt, img_channel, height, width]. It was stored as [img_cnt, h, w, ch]
image = image.swapaxes(1,2) 
image = image.swapaxes(2,3) 
# Image heigh, width, channel
cnt,h,w,ch = image.shape
# Create video file
fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
video = cv2.VideoWriter(args.h5_file+".avi",fourcc,15,(w,h),True)
# Record video
for i in range(cnt):
    img_i = np.uint8(image[i,:,:,:])
    video.write(img_i)
    #cv2.imwrite("./image_"+str(i).zfill(4)+".png",img_i)

# Clean up the scene
f.close()
cv2.destroyAllWindows()
video.release

