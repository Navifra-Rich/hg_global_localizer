#!/usr/bin/env python3

import os
import sys
import rospy

PACKAGE_PATH = rospy.get_param("dir_package")
sys.path.append(os.path.join(PACKAGE_PATH, "include", "placeNet"))

from knn import KNN
from orb import ORB_Matcher
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int64
import numpy as np
import json


TRAIN_MODE = rospy.get_param("train_mode")
DATASET_PATH = rospy.get_param("dir_dataset")
DATASET_TEST_PATH = rospy.get_param("dir_dataset_test")
MODEL_NAME = rospy.get_param("model_name")

class Kidnapper:
    def __init__(self):
        self.package_path = PACKAGE_PATH
        self.pub = rospy.Publisher("global_map", PointCloud2, queue_size=10)
        self.pcd_msg = None
        self.knn = KNN()
        self.orb = ORB_Matcher()

        self.meta_path = os.path.join(DATASET_PATH, "meta.json")
        self.meta_data = json.loads(open(self.meta_path).read())


        self.markers = MarkerArray()
        self.cur_idx = -1
        return
    def setKNN(self):
        self.knn.package_path = PACKAGE_PATH
        self.knn.dataset_path = DATASET_PATH

        self.orb.package_path = PACKAGE_PATH
        self.orb.dataset_path = DATASET_PATH
        self.orb.getImgList()

        # -------------- Spe ----------------------
        # self.knn.model_path = os.path.join(PACKAGE_PATH, "data", "myModel_my.pkl")
        # self.knn.json_path = os.path.join(PACKAGE_PATH, "data", "train_my.json")
        # ------------------------------------------
        
        return

    def getNearestByORB(self, idx):
        if self.cur_idx == idx:
            return

        img_path = os.path.join(DATASET_TEST_PATH, f"{idx}.jpg")
        places = kidnapper.orb.save_nearest_img(img_path)[0]

        for idx, item in enumerate(places):
            pose = kidnapper.meta_data[item[0]]["odom"]
            marker = Marker()
            marker.id = idx
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(15)
            marker.type = Marker.CUBE
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            # marker.pose.position.z = 5.5
            marker.scale.x = 1.7
            marker.scale.y = 1.7
            marker.scale.z = 1.7
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.markers.markers.append(marker)
            return
    
    def getNearestByKNN(self, idx):
        if self.cur_idx == idx:
            return
        img_path = os.path.join(DATASET_TEST_PATH, f"{idx}.jpg")
        places = kidnapper.knn.save_nearest_img(img_path)
        print(places)
        return
    
        self.markers = MarkerArray()
        for idx, item in enumerate(places):
            pose = kidnapper.meta_data[item]["odom"]
            marker = Marker()
            marker.id = idx
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(0.1)
            marker.type = Marker.CUBE
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            # marker.pose.position.z = 5.5
            marker.scale.x = 1.7
            marker.scale.y = 1.7
            marker.scale.z = 1.7
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.markers.markers.append(marker)
        # print(self.markers)
def image_idx_callback(msg):
    kidnapper.getNearestByKNN(msg.data)
    kidnapper.getNearestByORB(msg.data)
    kidnapper.cur_idx = msg.data


kidnapper = Kidnapper()

def main():
    rospy.init_node('kidnapper', anonymous=True)

    kidnapper.setKNN()
    kidnapper.knn.load_model(MODEL_NAME)

    sub_img_idx = rospy.Subscriber('/place_idx', Int64, image_idx_callback)
    pub_markers = rospy.Publisher('/place_marker', MarkerArray, queue_size=1)

    rate = rospy.Rate(10) # 10hz
    temp_idx = 0
    while not rospy.is_shutdown():
        # print("LOOP")
        # print(kidnapper.markers)
        image_idx_callback(Int64(temp_idx))
        rate.sleep()
        pub_markers.publish(kidnapper.markers)
        temp_idx+=15

def train():
    kidnapper.setKNN()
    kidnapper.knn.train_knn(MODEL_NAME)
    kidnapper.knn.save_model(MODEL_NAME)

    return
if __name__ == '__main__':

    if TRAIN_MODE:
        train()
    else:
        main()
