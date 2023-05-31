#!/usr/bin/env python3

import os
import sys
import rospy

PACKAGE_PATH = rospy.get_param("dir_package")
sys.path.append(os.path.join(PACKAGE_PATH, "include", "placeNet"))

from knn import KNN
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int64
import numpy as np
import json

TEMP_INPUT_IMG = "/home/hgnaseel/data/global_map"

class Kidnapper:
    def __init__(self):
        self.package_path = PACKAGE_PATH
        self.pub = rospy.Publisher("global_map", PointCloud2, queue_size=10)
        self.pcd_msg = None
        self.knn = KNN()

        self.meta_path = os.path.join(PACKAGE_PATH, "data", "meta.json")
        self.meta_data = json.loads(open(self.meta_path).read())


        self.markers = MarkerArray()
        self.cur_idx = -1
        return
    def setKNN(self):
        self.knn.package_path = PACKAGE_PATH
        self.knn.setPaths()
        self.knn.dataset_path="/home/hgnaseel/data/global_map"
        # -------------- Spe ----------------------
        # self.knn.model_path = os.path.join(PACKAGE_PATH, "data", "myModel_my.pkl")
        # self.knn.json_path = os.path.join(PACKAGE_PATH, "data", "train_my.json")
        # ------------------------------------------
        self.knn.load_model()
        
        return

    def getKNNbyIdx(self, idx):
        if self.cur_idx == idx:
            return
        self.cur_idx = idx
        img_path = os.path.join(TEMP_INPUT_IMG, f"{idx}.jpg")
        places = kidnapper.knn.save_nearest_img(img_path)

        self.markers = MarkerArray()

        for idx, item in enumerate(places):
            pose = kidnapper.meta_data[item[0]]["odom"]
            marker = Marker()
            marker.id = idx
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(0.1)
            marker.type = Marker.CUBE
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            # marker.pose.position.z = 5.5
            marker.scale.x = 1.2
            marker.scale.y = 1.2
            marker.scale.z = 1.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            self.markers.markers.append(marker)


def image_idx_callback(msg):
    # print(f"HERE!!!! {msg.data}")
    kidnapper.getKNNbyIdx(msg.data)
kidnapper = Kidnapper()

def main():
    rospy.init_node('kidnapper', anonymous=True)

    kidnapper.setKNN()

    sub_img_idx = rospy.Subscriber('/place_idx', Int64, image_idx_callback)
    pub_markers = rospy.Publisher('/place_marker', MarkerArray, queue_size=1)




        # return
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # print("LOOP")
        # print(kidnapper.markers)
        rate.sleep()
        pub_markers.publish(kidnapper.markers)


if __name__ == '__main__':
    main()
