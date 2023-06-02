#!/usr/bin/python3

import glob
import cv2
import numpy as np
import os
import time
import tf.transformations as tf
from skimage.metrics import structural_similarity as ssim

class ORB_Matcher:
    def __init__(self):
        self.map_list_ = glob.glob("/home/hgnaseel/data/global_map/*.jpg")
        self.map_list_.sort()
        self.matcher_ = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kaze_matcher_ = cv2.FlannBasedMatcher_create()

        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.5,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

        self.kaze = cv2.KAZE_create()
        
        # self.orb = cv2.ORB_create(
        #     nfeatures=500,
        #     scaleFactor=1.5,
        #     nlevels=8,
        #     edgeThreshold=31,
        #     firstLevel=0,
        #     WTA_K=3,
        #     scoreType=cv2.NORM_HAMMING2,
        #     patchSize=31,
        #     fastThreshold=20,
        # )
        self.idx = 0

        self.package_path = ""
        self.dataset_path = ""
        self.img_path_list = []
        return
    def getImgList(self):
        self.img_path_list = glob.glob(f'{self.dataset_path}'+os.sep+"*")
        self.img_path_list.sort()
        self.img_path_list = self.img_path_list[:-1]
        # print(self.img_path_list)
        self.img_path_list.sort(key = lambda x: int(x.split(os.sep)[-1][:-4]))   # get rid of .jpg, .meta

    def save_nearest_img(self, img_path):

        input_img = cv2.imread(img_path)
        print(input_img.shape)
        # cv2.imwrite("/home/hgnaseel/HERE.jpg",input_img[0:240,:,:])
        kp_input, des_input = self.orb.detectAndCompute(input_img, None)
        kp_input_half, des_input_half = self.orb.detectAndCompute(input_img[0:240,:,:], None)
        kp_input_kaze, des_input_kaze = self.kaze.detectAndCompute(cv2.cvtColor(input_img[0:240,:,:], cv2.COLOR_BGR2GRAY), None)
        dist_list = []
        dist_list_half = []
        dist_list_kaze = []
        for idx, img_path in enumerate(self.img_path_list):
            if idx%3!=0:
                pass
            # print(f"PATH {img_path}")
            img = cv2.imread(img_path)

            kp_db, des_db = self.orb.detectAndCompute(img, None)
            kp_db_half, des_db_half = self.orb.detectAndCompute(img[0:240,:,:], None)
            kp_db_kaze, des_db_kaze = self.kaze.detectAndCompute(cv2.cvtColor(img[0:240,:,:], cv2.COLOR_BGR2GRAY), None)
            match = self.matcher_.match(des_input, des_db)
            # match = sorted(match, key = lambda x:x.distance)

            match_half = self.matcher_.match(des_input_half, des_db_half)
            
            match_kaze = self.kaze_matcher_.knnMatch(des_input_kaze, des_db_kaze, k=2)
            # match = sorted(match, key = lambda x:x.distance)

            good_match_num = 0
            good_match_num_half = 0
            good_match_num_kaze = 0

            for item in match:
                if item.distance<50:
                    good_match_num+=1

            for item in match_half:
                if item.distance<50:
                    good_match_num_half+=1

            for m, n in match_kaze:
                if m.distance < 0.7 * n.distance:
                    good_match_num_kaze+=1
                    
                # print(f"DIST = {item.distance}")
            dist_list.append([idx, good_match_num])
            dist_list_half.append([idx, good_match_num_half])
            dist_list_kaze.append([idx, good_match_num_kaze])
        dist_list.sort(key=lambda x : x[1], reverse=True)
        dist_list_half.sort(key=lambda x : x[1], reverse=True)
        dist_list_kaze.sort(key=lambda x : x[1], reverse=True)
        print(f" -------------  DIST -------------------")
        print(dist_list[:10])
        print(f" -------------  HALF -------------------")
        print(dist_list_half[:10])
        print(f" -------------  KAZE -------------------")
        print(dist_list_kaze[:10])
if __name__=="__main__":
    print("HELLO WORLD!")
