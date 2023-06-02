#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
import json
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import os
import math

bridge = CvBridge()
pos_x, pos_y, pos_z = 0,0,0
firstRun = True
img = None
data_list=[]
scene_idx = 0

IMAGE_ENCODING = rospy.get_param("image_encoding")
POINT_TOPIC_NAME = rospy.get_param("point_topic_name") 
ODOM_TOPIC_NAME = rospy.get_param("odom_topic_name")
IMAGE_TOPIC_NAME = rospy.get_param("image_topic_name")
SAVE_TERM = rospy.get_param("save_term")
PACKAGE_DIR = rospy.get_param("package_dir")

def img_callback(msg):
    global img
    # print("Received an image!")
    img = bridge.imgmsg_to_cv2(msg, IMAGE_ENCODING)
    heartbeat(Odometry())

def point_callback(msg):
    global firstRun
    # print("Received an pose!")
    odom = Odometry()
    odom.pose.pose.position.x = msg.point.x
    odom.pose.pose.position.y = msg.point.y
    odom.pose.pose.position.z = msg.point.z
    odom.pose.pose.orientation.w = 1
    heartbeat(odom)

def odom_callback(msg):
    global firstRun
    print("Received an odom!")
    heartbeat(msg)

def firstRunCheck(new_x, new_y):
    global firstRun
    if firstRun:
        firstRun = False
        return
    
def getDist(pos_x, pos_y, pos_z, new_x, new_y, new_z):
    dist_x = abs(pos_x - new_x)
    dist_y = abs(pos_y - new_y)
    dist_z = abs(pos_z - new_z)
    return math.sqrt(dist_x*dist_x + dist_y*dist_y) + dist_z*dist_z

def save_db(odom):
    global scene_idx, img, data_list
    # save img
    img_path = f'{PACKAGE_DIR}/result/{scene_idx}.jpg'  # 저장할 이미지 파일 경로 및 이름을 지정합니다
    cv2.imwrite(img_path, img)

    px = odom.pose.pose.position.x
    py = odom.pose.pose.position.y
    pz = odom.pose.pose.position.z
    
    ox = odom.pose.pose.orientation.x
    oy = odom.pose.pose.orientation.y
    oz = odom.pose.pose.orientation.z
    ow = odom.pose.pose.orientation.w
    data = {"id" : scene_idx, "odom" : [px, py, pz, ox, oy, oz, ow]}
    
    data_list.append(data)
    
    json_str = json.dumps(data_list, indent=2)
    file_path = f'{PACKAGE_DIR}/result/meta.json'  # 저장할 JSON 파일 경로 및 이름을 지정합니다.
    with open(file_path, 'w') as file:
        file.write(json_str)

    scene_idx +=1

save_idx = 0
def heartbeat(odom):
    global pos_x, pos_y, pos_z, save_idx
    new_x, new_y, new_z = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
    
    os.system("clear")
    print(f"    POSE {new_x} {new_y} {new_z}")

    if firstRunCheck(new_x, new_y):
        pos_x , pos_y, pos_z = new_x, new_y, new_z
        return
    
    dist = getDist(pos_x, pos_y, pos_z, new_x, new_y, new_z)
    print(f"    DIST {dist}")

    if dist>SAVE_TERM or save_idx % 35 == 0:
        save_db(odom)
        pos_x , pos_y, pos_z = new_x, new_y, new_z
    save_idx += 1


if __name__ == '__main__':
    rospy.init_node("map_generator")
    print("hello world")

    # ------------------------- Subscribers ----------------------------------
    odom_sub    = rospy.Subscriber(ODOM_TOPIC_NAME, Odometry, odom_callback)
    point_sub   = rospy.Subscriber(POINT_TOPIC_NAME, PointStamped, point_callback)
    img_sub     = rospy.Subscriber(IMAGE_TOPIC_NAME, Image, img_callback)
    print(POINT_TOPIC_NAME)
    print(POINT_TOPIC_NAME)
    print(POINT_TOPIC_NAME)
    # ------------------------- Publishers -----------------------------------


    os.makedirs(os.path.join(PACKAGE_DIR, "result"), exist_ok=True)

    r = rospy.Rate(30)
    sim_time = 0.0


    node_pub_idx = 0
    while not rospy.is_shutdown():
        # print(f" Sim Time = {sim_time}")
        r.sleep()



# CONV net simple image retrieval
# https://towardsdatascience.com/a-hands-on-introduction-to-image-retrieval-in-deep-learning-with-pytorch-651cd6dba61e

# 