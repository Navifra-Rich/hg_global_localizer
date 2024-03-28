#!/usr/bin/python3

import glob
import cv2
import os

class KAZE_Matcher:
    def __init__(self):
        self.matcher_ = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kaze_matcher_ = cv2.FlannBasedMatcher_create()
        self.kaze = cv2.KAZE_create()
        self.dataset_path = ""
        self.img_path_list = []

        return

    def getImgList(self):
        self.img_path_list = glob.glob(f'{self.dataset_path}'+os.sep+"*")
        self.img_path_list.sort()
        self.img_path_list = self.img_path_list[:-1]
        # print(self.img_path_list)
        self.img_path_list.sort(key = lambda x: int(x.split(os.sep)[-1][:-4]))   # get rid of .jpg, .meta

    def save_nearest_img(self, input_img):
        print(f"Image Shape {input_img.shape}")
        kp_input_kaze, des_input_kaze = self.kaze.detectAndCompute(cv2.cvtColor(input_img[:,:,:], cv2.COLOR_BGR2GRAY), None)
        dist_list = []
        dist_list_kaze = []
        for idx, path in enumerate(self.img_path_list):
            if idx%6!=0:
                continue
            # print(f"PATH {img_path}")
            img = cv2.imread(path)

            kp_db_kaze, des_db_kaze = self.kaze.detectAndCompute(cv2.cvtColor(img[:,:,:], cv2.COLOR_BGR2GRAY), None)
            match_kaze = self.kaze_matcher_.knnMatch(des_input_kaze, des_db_kaze, k=2)
            good_match_num = 0
            good_match_num_kaze = 0

            for m, n in match_kaze:
                if m.distance < 0.5 * n.distance:
                    good_match_num_kaze+=1
            if True:
            # if good_match_num_kaze>10:
                dist_list.append([idx, good_match_num])
                dist_list_kaze.append([idx, good_match_num_kaze])
        dist_list_kaze.sort(key=lambda x : x[1], reverse=True)
        print(f"KAZE DIST {dist_list_kaze}")
        return [sublist[0] for sublist in dist_list_kaze]
    


if __name__=="__main__":
    print("HELLO WORLD!")
