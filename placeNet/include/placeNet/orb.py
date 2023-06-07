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
        print(f"Input image shape {input_img.shape}")
        kp_input, des_input = self.orb.detectAndCompute(input_img, None)
        kp_input_half, des_input_half = self.orb.detectAndCompute(input_img[0:240,:,:], None)
        kp_input_kaze, des_input_kaze = self.kaze.detectAndCompute(cv2.cvtColor(input_img[0:240,:,:], cv2.COLOR_BGR2GRAY), None)
        dist_list = []
        dist_list_half = []
        dist_list_kaze = []
        start = time.time()
        for idx, path in enumerate(self.img_path_list):
            if idx%6!=0:
                continue
            # print(f"PATH {img_path}")
            img = cv2.imread(path)

            # kp_db, des_db = self.orb.detectAndCompute(img, None)
            # kp_db_half, des_db_half = self.orb.detectAndCompute(img[0:240,:,:], None)
            fe_start = time.time()
            kp_db_kaze, des_db_kaze = self.kaze.detectAndCompute(cv2.cvtColor(img[0:240,:,:], cv2.COLOR_BGR2GRAY), None)
            # match = self.matcher_.match(des_input, des_db)
            # match_half = self.matcher_.match(des_input_half, des_db_half)
            ma_start = time.time()
            match_kaze = self.kaze_matcher_.knnMatch(des_input_kaze, des_db_kaze, k=2)
            print(f" TIME DIFF {ma_start - fe_start} {time.time()-ma_start}")
            good_match_num = 0
            good_match_num_half = 0
            good_match_num_kaze = 0

            # for item in match:
            #     if item.distance<50:
            #         good_match_num+=1

            # for item in match_half:
            #     if item.distance<50:
            #         good_match_num_half+=1

            for m, n in match_kaze:
                if m.distance < 0.7 * n.distance:
                    good_match_num_kaze+=1
                    
            dist_list.append([idx, good_match_num])
            dist_list_half.append([idx, good_match_num_half])
            dist_list_kaze.append([idx, good_match_num_kaze])
        print(f"TIME  {time.time() - start}")
        # dist_list.sort(key=lambda x : x[1], reverse=True)
        # dist_list_half.sort(key=lambda x : x[1], reverse=True)
        dist_list_kaze.sort(key=lambda x : x[1], reverse=True)
        # print(f" -------------  DIST -------------------")
        # print(dist_list[:10])
        # print(f" -------------  HALF -------------------")
        # print(dist_list_half[:10])
        print(f" -------------  KAZE -------------------")
        print(dist_list_kaze[:10])
        img_path_list = []
        for idx in dist_list_kaze[:4]:
            img_path_list.append(os.path.join(self.dataset_path, f"{idx[0]}.jpg"))
        self.publish_imgs(img_path, img_path_list)
        return [dist_list_kaze[:4]]
    
    def publish_imgs(self, origin, near_list):
        origin_img = cv2.imread(origin)
        near_img_list = []
        for idx in range(0, len(near_list),2):
            print(idx)
            img1 = cv2.imread(near_list[idx])
            img2 = cv2.imread(near_list[idx+1])
            img1 = cv2.putText(img1, str(idx+1), (350, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
            img2 = cv2.putText(img2, str(idx+2), (350, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
            v_combined = np.vstack([img1,img2])
            near_img_list.append(v_combined)
        near_combined = np.hstack(near_img_list)

        cv2.imwrite(os.path.join(self.package_path, 'result', 'Combined_Images.jpg'), near_combined)
        cv2.imwrite(os.path.join(self.package_path, 'result', 'Origin_Images.jpg'), origin_img)

if __name__=="__main__":
    print("HELLO WORLD!")
