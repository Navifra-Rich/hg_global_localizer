#!/usr/bin/python3

import rospy

import glob
import cv2
import random
from sensor_msgs.msg import Image
import numpy as np
import os
import time
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tf
from skimage.metrics import structural_similarity as ssim

class ORB_Matcher:
    def __init__(self):
        self.map_list_ = glob.glob("/home/hgnaseel/data/global_map/*.jpg")
        self.map_list_.sort()
        self.matcher_ = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(
            nfeatures=150,
            scaleFactor=1.5,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        self.idx = 0
        return
    
    def img_callback(self, msg):
        input_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        size = len(self.map_list_)
        time_sum = 0

        idx = -1
        min_idx = -1
        min_val = 999
        for item in self.map_list_:
            idx +=1
            if idx%2!=0:
                continue

            start_time = time.time()

            map_img = cv2.imread(item)

            kp1, des1 = self.orb.detectAndCompute(input_img, None)
            kp2, des2 = self.orb.detectAndCompute(map_img, None)

            match = self.matcher_.match(des1, des2)
            match = sorted(match, key = lambda x:x.distance)
            # print(match1[0])
            end_time = time.time()
            time_sum += (end_time-start_time)
            # print(f'Time = {end_time-start_time}')
            # result1 = cv2.drawMatches(input_img, kp1, map_img, kp2, match1, input_img, flags=2)
            result1 = cv2.drawMatches(input_img, kp1, map_img, kp2, match[:30], input_img, flags=2)


            src_points = np.float32([kp1[m.queryIdx].pt for m in match[:30]]).reshape(-1, 1, 2)
            dst_points = np.float32([kp2[m.trainIdx].pt for m in match[:30]]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            K = np.array([[469.18043387 ,  0.    ,         259.40407955],
                 [  0.         ,454.89333646    , 218.33055988],
                 [  0.         ,  0.    ,           1.        ]])
            xx, R, T, xxx = cv2.decomposeHomographyMat(M, K)
            # print('-------------------------------------')
            # print(f"R = {R}")
            # print(f"T = {T}")
            # print(f"R_rpy = {tf.euler_from_matrix(R[0], axes='sxyz')}")
            # print(f"R_rpy = {tf.euler_from_matrix(R[1], axes='sxyz')}")
            # print(f"R_rpy = {tf.euler_from_matrix(R[2], axes='sxyz')}")
            # print(f"R_rpy = {tf.euler_from_matrix(R[3], axes='sxyz')}")
            # cv2.imshow(f"match {self.idx} {idx}", result1)
            # cv2.waitKey(0)

            r, p, y = tf.euler_from_matrix(R[0], axes='sxyz')
            sum = abs(r) + abs(p)
            if sum<min_val:
                min_idx = idx
                min_val = sum



        map_img = cv2.imread(self.map_list_[min_idx])
        kp1, des1 = self.orb.detectAndCompute(input_img, None)
        kp2, des2 = self.orb.detectAndCompute(map_img, None)
        match = self.matcher_.match(des1, des2)
        match = sorted(match, key = lambda x:x.distance)
        # print(match1[0])
        end_time = time.time()
        time_sum += (end_time-start_time)
        # print(f'Time = {end_time-start_time}')
        # result1 = cv2.drawMatches(input_img, kp1, map_img, kp2, match1, input_img, flags=2)
        result1 = cv2.drawMatches(input_img, kp1, map_img, kp2, match[:30], input_img, flags=2)

        # cv2.imshow(f"match {self.idx} {min_idx}", result1)
        cv2.imwrite(f"/home/hgnaseel/loam_ws/src/map_db_generator/result/match_{self.idx}_{min_idx}.jpg", result1)

        self.idx +=1
        # cv2.waitKey(0)
        print(f'    Time sum = {time_sum}')
        return

    def compare_images_ssim(self, image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (_, ssim_map) = ssim(gray1, gray2, full=True)
        ssim_index = np.mean(ssim_map)
        return ssim_index
     
    def img_callback_ssim(self, msg):
        input_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        ssim_list = []
        for idx, item in enumerate(self.map_list_):
            map_img = cv2.imread(item)
            ssim_idx = self.compare_images_ssim(input_img, map_img)
            ssim_list.append(ssim_idx)

        ssim_list.sort()
        path = f"/home/hgnaseel/loam_ws/src/map_db_generator/result"
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f"{path}/origin.jpg", input_img)
        for i in range(5):
            cv2.imwrite(f"{path}/match_{ssim_list[i]}.jpg", self.map_list_[ssim_list[i]])
        self.idx +=1
        return
    
orb = ORB_Matcher()
if __name__=="__main__":
    print("HELLO WORLD!")
    rospy.init_node("localizer")
    # mylist = glob.glob("/home/hgnaseel/Downloads/0601-20230522T072858Z-001/0601/*")
    # img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, orb.img_callback, queue_size=1)
    img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, orb.img_callback_ssim, queue_size=1)



    while not rospy.is_shutdown():
        # print("LOOP")
        rospy.Rate(0.2).sleep()



    # for pck in mylist:
    #     img_list


