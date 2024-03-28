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

PACKAGE_PATH = rospy.get_param("dir_package")
DATASET_PATH = rospy.get_param("dir_dataset")
DATASET_TEST_PATH = rospy.get_param("dir_dataset_test")
IMAGE_TOPIC_NAME = rospy.get_param("image_topic_name")
bridge = CvBridge()

sys.path.append(os.path.join(PACKAGE_PATH, "include", "VPS_localizer"))

class Kidnapper:
    def __init__(self):
        self.kaze = KAZE_Matcher()
        self.dataset_path = os.path.join(DATASET_PATH, "meta.json")
        self.dataset_json = json.loads(open(self.dataset_path).read())
        self.testset_path = os.path.join(DATASET_TEST_PATH, "meta.json")
        self.testset_json = json.loads(open(self.testset_path).read())


        self.originMap_image_path = os.path.join(PACKAGE_PATH, "data", "magok_superstart.png") # 결과물 출력용 지도 원본 이미지

        self.setKAZE()

    def setPubSub(self):
        print(f"Image Topic {IMAGE_TOPIC_NAME}")
        self.img_sub     = rospy.Subscriber(IMAGE_TOPIC_NAME, CompressedImage, self.img_callback)

    def setKAZE(self):
        self.kaze.package_path = PACKAGE_PATH
        self.kaze.dataset_path = DATASET_PATH
        self.kaze.getImgList()

    def getNearestByKAZE(self, image):
        places = kidnapper.kaze.save_nearest_img(image)
        return places
    
    def getNearestByKAZE_TestSet(self, idx):
        img_path = os.path.join(DATASET_TEST_PATH, f"{idx}.jpg")
        places = kidnapper.kaze.save_nearest_img(cv2.imread(img_path))
        return places
    
    def img_callback(self, msg):

        # ------------------ Image Preprocess -------------------------------------
        # image Msg에서 OpenCVImage로 변경
        img = bridge.compressed_imgmsg_to_cv2(msg)
        img = cv2.flip(img, 0)          # 원본 이미지 뒤집어져있는 이슈
        target_width = int(1920/4)      # 다운샘플링
        target_height = int(1080/4)     # 다운샘플링
        print(f"Resize{target_width} {target_height}")
        img = cv2.resize(img, (target_width, target_height))
        # print(f"Image path {DATASET_TEST_PATH} ")

        # ------------------ Detect Feature & Estimate Pose -------------------------------------
        kaze_result = kidnapper.getNearestByKAZE(img)
        print(f"KAZE RESULT {kaze_result}")

        ## PARAMETER
        output_save_path = os.path.join(PACKAGE_PATH,"result")
        os.makedirs(output_save_path, exist_ok=True)

        saveMap_image_path = os.path.join(PACKAGE_PATH, "result", "resultMap.png")


        # groundtruth_pose = kidnapper.testset_json[msg.data]["odom"]
        estimated_pose = self.dataset_json[kaze_result[0]]["odom"]


        # ------------------ Print Output -------------------------------------
        img_map = cv2.imread(self.originMap_image_path)
        # cv2.circle(img_map, (int(groundtruth_pose[0]*100 + img_map.shape[1]/2), int( -groundtruth_pose[1]*100 + img_map.shape[0]/2)) , 10, (0, 0, 255) , -1)    
        cv2.circle(img_map, (int(estimated_pose[0]*100 + img_map.shape[1]/2), int( -estimated_pose[1]*100 + img_map.shape[0]/2)) , 10, (255, 0, 0) , -1)    
        cv2.imwrite(saveMap_image_path, img_map)

        # print(f"Input  Pose  {groundtruth_pose}")
        print(f"Estimated Pose {estimated_pose}")
        print()

    def EstimateDatasets(self, msg):
        print(f"Image path {DATASET_TEST_PATH} ")

        print(f"Test Index {msg.data}")
        kaze_result = kidnapper.getNearestByKAZE(msg.data)
        print(f"KAZE RESULT {kaze_result}")

        img_input = cv2.imread(os.path.join(DATASET_TEST_PATH, str(msg.data)+".jpg"))
        img_kaze1 = cv2.imread(os.path.join(DATASET_PATH, str(kaze_result[0])+".jpg"))

        ## PARAMETER
        output_save_path = os.path.join(PACKAGE_PATH,"result")
        os.makedirs(output_save_path, exist_ok=True)


        # Save Test Image
        cv2.imwrite(os.path.join(output_save_path,f"origin.jpg"), img_input)

        # Save Result Images
        cv2.imwrite(os.path.join(output_save_path,f"kaze1.jpg"), img_kaze1)

        saveMap_image_path = os.path.join(PACKAGE_PATH, "result", "resultMap.png")


        groundtruth_pose = kidnapper.testset_json[msg.data]["odom"]
        estimated_pose = kidnapper.dataset_json[kaze_result[0]]["odom"]


        img_map = cv2.imread(self.originMap_image_path)
        cv2.circle(img_map, (int(groundtruth_pose[0]*100 + img_map.shape[1]/2), int( -groundtruth_pose[1]*100 + img_map.shape[0]/2)) , 10, (0, 0, 255) , -1)    
        cv2.circle(img_map, (int(estimated_pose[0]*100 + img_map.shape[1]/2), int( -estimated_pose[1]*100 + img_map.shape[0]/2)) , 10, (255, 0, 0) , -1)    

        cv2.imwrite(saveMap_image_path, img_map)

        print(f"Input  Pose  {groundtruth_pose}")
        print(f"Output Pose1 {estimated_pose}")
        print()


kidnapper = Kidnapper()
def main():
    rospy.init_node('kidnapper', anonymous=True)

    kidnapper.setPubSub()

    rospy.spin()



    # rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    #     print("LOOP")
    #     random_idx = random.randint(0, 240)
    #     EstimateDatasets(Int64(random_idx))
    #     rate.sleep()

    return

if __name__ == '__main__':

    main()
