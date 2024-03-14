#!/usr/bin/env python3

import os
import sys
import rospy
import cv2

import random
from kaze import KAZE_Matcher
from std_msgs.msg import Int64
import json

PACKAGE_PATH = rospy.get_param("dir_package")
DATASET_PATH = rospy.get_param("dir_dataset")
DATASET_TEST_PATH = rospy.get_param("dir_dataset_test")

sys.path.append(os.path.join(PACKAGE_PATH, "include", "VPS_localizer"))

class Kidnapper:
    def __init__(self):
        self.kaze = KAZE_Matcher()
        self.dataset_path = os.path.join(DATASET_PATH, "meta.json")
        self.dataset_json = json.loads(open(self.dataset_path).read())
        self.testset_path = os.path.join(DATASET_TEST_PATH, "meta.json")
        self.testset_json = json.loads(open(self.testset_path).read())
        self.setKAZE()

    def setKAZE(self):
        self.kaze.package_path = PACKAGE_PATH
        self.kaze.dataset_path = DATASET_PATH
        self.kaze.getImgList()

    def getNearestByKAZE(self, idx):
        img_path = os.path.join(DATASET_TEST_PATH, f"{idx}.jpg")
        places = kidnapper.kaze.save_nearest_img(img_path)
        return places
    
def image_idx_callback(msg):
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

    originMap_image_path = os.path.join(PACKAGE_PATH, "data", "magok_superstart.png")
    saveMap_image_path = os.path.join(PACKAGE_PATH, "result", "resultMap.png")


    groundtruth_pose = kidnapper.testset_json[msg.data]["odom"]
    estimated_pose = kidnapper.dataset_json[kaze_result[0]]["odom"]


    img_map = cv2.imread(originMap_image_path)
    cv2.circle(img_map, (int(groundtruth_pose[0]*100 + img_map.shape[1]/2), int( -groundtruth_pose[1]*100 + img_map.shape[0]/2)) , 10, (0, 0, 255) , -1)    
    cv2.circle(img_map, (int(estimated_pose[0]*100 + img_map.shape[1]/2), int( -estimated_pose[1]*100 + img_map.shape[0]/2)) , 10, (255, 0, 0) , -1)    

    cv2.imwrite(saveMap_image_path, img_map)

    print(f"Input  Pose  {groundtruth_pose}")
    print(f"Output Pose1 {estimated_pose}")
    print()
    print()

kidnapper = Kidnapper()

def main():
    rospy.init_node('kidnapper', anonymous=True)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        print("LOOP")
        random_idx = random.randint(0, 240)
        image_idx_callback(Int64(random_idx))
        rate.sleep()

    return

if __name__ == '__main__':

    main()
