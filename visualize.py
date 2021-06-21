import cv2
import glob
import numpy as np
import pandas as pd
import csv
import matplotlib.image as img 
import matplotlib.pyplot as plt

with open('data/driving_log.csv') as csv_f:
    reader = csv.DictReader(csv_f)
    for frame_info in reader:
        center_imgs_file = frame_info['center'].split('/')[-1]
        right_imgs_file = frame_info['right'].split('/')[-1]
        left_imgs_file = frame_info['left'].split('/')[-1]
     
        center_imgs = cv2.imread('data/IMG/'+center_imgs_file)
        right_imgs = cv2.imread('data/IMG/'+right_imgs_file)
        left_imgs = cv2.imread('data/IMG/'+left_imgs_file)
        
        fig = plt.figure()
       
    cv2.imwrite('result/cneter.png' ,center_imgs)
    cv2.imwrite('result/right.png' ,right_imgs)
    cv2.imwrite('result/left.png' ,left_imgs)
        

    
    
   