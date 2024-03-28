#!/usr/bin/env python3

import os
import sys
import rospy
import cv2

import random
from kaze import KAZE_Matcher
from std_msgs.msg import Int64
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import json
import glob
from threading import Lock
REALTIME_MODE = rospy.get_param("realtime_mode")
PACKAGE_PATH = rospy.get_param("dir_package")
DATASET_PATH = rospy.get_param("dir_dataset")
DATASET_TEST_PATH = rospy.get_param("dir_dataset_test")

LEFT_IMAGE_TOPIC_NAME = rospy.get_param("left_image_topic_name")
RIGHT_IMAGE_TOPIC_NAME = rospy.get_param("right_image_topic_name")

bridge = CvBridge()
combined_image = None
image_left_ = None
image_right_ = None
lock=Lock()

sys.path.append(os.path.join(PACKAGE_PATH, "include", "VPS_localizer"))

class Kidnapper:
    def __init__(self):
        self.kaze = KAZE_Matcher()
        self.dataset_path = os.path.join(DATASET_PATH, "meta.json")
        self.dataset_json = json.loads(open(self.dataset_path).read())
        self.testset_path = os.path.join(DATASET_TEST_PATH, "meta.json")
        self.testset_json = json.loads(open(self.testset_path).read())


        self.originMap_image_path = os.path.join(PACKAGE_PATH, "data", "magok_superstart.png") # 결과물 출력용 지도 원본 이미지
        self.saveMap_image_path = os.path.join(PACKAGE_PATH, "result", "resultMap.png")

        self.output_save_path = os.path.join(PACKAGE_PATH,"result")
        os.makedirs(self.output_save_path, exist_ok=True)

        self.setKAZE()

        self.mean_dist = 0

    def setPubSub(self):
        print(f"Image Topic {LEFT_IMAGE_TOPIC_NAME} {RIGHT_IMAGE_TOPIC_NAME}")
        self.left_img_sub     = rospy.Subscriber(LEFT_IMAGE_TOPIC_NAME, CompressedImage, self.left_image_callback)
        self.right_img_sub     = rospy.Subscriber(RIGHT_IMAGE_TOPIC_NAME, CompressedImage, self.right_image_callback)

    def setKAZE(self):
        self.kaze.package_path = PACKAGE_PATH
        self.kaze.dataset_path = DATASET_PATH
        self.kaze.getImgList()

    def getNearestByKAZE(self, image):
        places = kidnapper.kaze.save_nearest_img(image)
        return places


    # 이미지 콜백 함수 1
    def left_image_callback(self, data):
        
        global image_left_
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        image_left = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        # image_left = cv2.flip(image_left, 0)
        target_width = int(1920/4)  # 변경하려는 너비
        target_height = int(1080/4)  # 변경하려는 높이
        with lock:
            image_left_ = cv2.resize(image_left, (target_width, target_height))


    # 이미지 콜백 함수 2
    def right_image_callback(self, data):
        global image_right_
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        image_right = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        image_right = cv2.flip(image_right,0)
        image_right = cv2.flip(image_right,1)
        target_width = int(1920/4)  # 변경하려는 너비
        target_height = int(1080/4)  # 변경하려는 높이
        with lock:
            image_right_ = cv2.resize(image_right, (target_width, target_height))

    def save_combined_image(self, para):
        global image_left_, image_right_

        # 두 이미지가 모두 수신되었을 때만 실행
        if image_left_ is not None and image_right_ is not None:
            # 이미지 합치기
            combined_image = cv2.hconcat([image_left_, image_right_])
            self.heartbeat(combined_image)
            # 이미지 저장
            print("Combined image saved.")

    def heartbeat(self, img):


        # ------------------ Detect Feature & Estimate Pose -------------------------------------
        kaze_result = kidnapper.getNearestByKAZE(img)
        print(f"KAZE RESULT {kaze_result}")
        if len(kaze_result) ==0:
            return




        # testset_pose = kidnapper.testset_json[msg.data]["odom"]
        estimated_pose = self.dataset_json[kaze_result[0]]["odom"]


        # ------------------ Print Output -------------------------------------
        self.printImage(kaze_result)


        # print(f"Input  Pose  {testset_pose}")
        print(f"Estimated Pose {estimated_pose}")
        print()

    def EstimateDatasets(self, img_idx):
        # print(f"Image path {DATASET_TEST_PATH} ")

        print(f"Test Index {img_idx}")
        img_path = os.path.join(DATASET_TEST_PATH, f"{img_idx}.jpg")
        print(f"Image path {img_path}")
        cv_image = cv2.imread(img_path)
        print(f"Image shape {cv_image.shape}")
        kaze_result = kidnapper.getNearestByKAZE(cv_image)
        print(f"KAZE RESULT {kaze_result}")
        if len(kaze_result) ==0:
            return
        img_input = cv2.imread(os.path.join(DATASET_TEST_PATH, str(img_idx)+".jpg"))
        img_kaze1 = cv2.imread(os.path.join(DATASET_PATH, str(kaze_result[0])+".jpg"))

        # Save Test Image
        cv2.imwrite(os.path.join(self.output_save_path,f"origin.jpg"), img_input)

        # Save Result Images
        cv2.imwrite(os.path.join(self.output_save_path,f"kaze1.jpg"), img_kaze1)


        testset_pose = kidnapper.testset_json[img_idx]["odom"]
        estimated_pose = kidnapper.dataset_json[kaze_result[0]]["odom"]


        # ------------------ Print Output -------------------------------------
        self.printImage(kaze_result, testset_pose)

        print(f"RESULT IDX {img_idx} {kaze_result[0]}")
        print(f"Input  Pose  {testset_pose}")
        print(f"Output Pose1 {estimated_pose}")

        print()

    def printImage(self, kaze_result, testset_pose = None):
        
        # ------------------ Print Output -------------------------------------
        img_map = cv2.imread(self.originMap_image_path)
        if testset_pose is not None:
            cv2.circle(img_map, (int(testset_pose[0]*100 + img_map.shape[1]/2), int( -testset_pose[1]*100 + img_map.shape[0]/2)) , 10, (0, 0, 255) , -1)    
        for idx, result in enumerate(kaze_result):
            if idx==1:
                break
            estimated_pose_x = kidnapper.dataset_json[result]["odom"][0]
            estimated_pose_y = kidnapper.dataset_json[result]["odom"][1]
            print(f"X = {estimated_pose_x} Y={estimated_pose_y}")
            cv2.circle(img_map, (int(estimated_pose_x*100 + img_map.shape[1]/2), int( -estimated_pose_y*100 + img_map.shape[0]/2)) , 10, (255, 0, 0) , -1)    
        cv2.imwrite(self.saveMap_image_path, img_map)
        return
kidnapper = Kidnapper()
def main():
    rospy.init_node('kidnapper', anonymous=True)

    kidnapper.setPubSub()


    test_idx = 0
    # ROS TOPIC MODE
    if REALTIME_MODE:
        rospy.Timer(rospy.Duration(1.0), kidnapper.save_combined_image)
        rospy.spin()
    else:
        # TESTSET MODE
        testsetNum = len(glob.glob(DATASET_TEST_PATH + '/*')) -1
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            print("LOOP")

            # ---- Random Mode
            test_idx = random.randint(0, testsetNum)
            # ---- ALLSearch Mode
            # test_idx+=1
            if test_idx >=testsetNum:
                break
            kidnapper.EstimateDatasets(test_idx)
            rate.sleep()
    

    return

if __name__ == '__main__':

    main()
