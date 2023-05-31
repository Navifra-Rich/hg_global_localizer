#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import json
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()
pos_x, pos_y = 0,0
odom = Odometry()
firstRun = True
img = None
data_list=[]
scene_idx = 0

def img_callback(msg):
    global img
    print("Received an image!")
    img = bridge.imgmsg_to_cv2(msg, "bgr8")

def odom_callback(msg):
    global odom, firstRun
    new_x, new_y = msg.pose.pose.position.x, msg.pose.pose.position.y

    if firstRun:
        pos_x , pos_y = new_x, new_y
        odom = msg
        firstRun = False
        return

    dist_x = abs(odom.pose.pose.position.x - new_x)
    dist_y = abs(odom.pose.pose.position.y - new_y)
    dist = dist_x*dist_x + dist_y*dist_y

    if dist>3:
        save_db()
        pos_x , pos_y = new_x, new_y
        odom = msg
    return

def save_db():
    global scene_idx, img, odom, data_list
    # save img
    img_path = f'/home/hgnaseel/data/global_map/{scene_idx}.jpg'  # 저장할 이미지 파일 경로 및 이름을 지정합니다
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
    # dic[""]
    
    json_str = json.dumps(data_list, indent=2)
    file_path = f'/home/hgnaseel/data/global_map/meta.json'  # 저장할 JSON 파일 경로 및 이름을 지정합니다.
    with open(file_path, 'w') as file:
        file.write(json_str)

    scene_idx +=1
    #   with coord
    return
if __name__ == '__main__':
    rospy.init_node("map_generator")
    print("hello world")

    # ------------------------- Subscribers ----------------------------------
    odom_sub = rospy.Subscriber("/odom", Odometry, odom_callback)
    img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, img_callback)

    # ------------------------- Publishers -----------------------------------




    r = rospy.Rate(30)
    sim_time = 0.0


    node_pub_idx = 0
    while not rospy.is_shutdown():
        # print(f" Sim Time = {sim_time}")
        r.sleep()



# CONV net simple image retrieval
# https://towardsdatascience.com/a-hands-on-introduction-to-image-retrieval-in-deep-learning-with-pytorch-651cd6dba61e

# 