#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
import json
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import os
import math
from threading import Lock
import numpy as np
lock = Lock()
kaze = cv2.KAZE_create()

odom_ = Odometry()
bridge = CvBridge()
pos_x, pos_y, pos_z = 0,0,0
firstRun = True
data_list=[]
scene_idx = 0


combined_image = None
image_left_ = None
image_right_ = None

IMAGE_ENCODING = rospy.get_param("image_encoding")
POINT_TOPIC_NAME = rospy.get_param("point_topic_name") 
LEFT_IMAGE_TOPIC_NAME = rospy.get_param("left_image_topic_name")
RIGHT_IMAGE_TOPIC_NAME = rospy.get_param("right_image_topic_name")
SAVE_TERM = rospy.get_param("save_term")
PACKAGE_DIR = rospy.get_param("package_dir")
RESULT_FOLDER = rospy.get_param("result_folder")

# 이미지 콜백 함수 1
def left_image_callback(data):
    global image_left_
    # ROS 이미지 메시지를 OpenCV 이미지로 변환
    image_left = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    # print(f"Left {image_left.shape}")
    # image_left = cv2.flip(image_left, 0)
    target_width = int(1920/4)  # 변경하려는 너비
    target_height = int(1080/4)  # 변경하려는 높이
    with lock:
        image_left_ = cv2.resize(image_left, (target_width, target_height))


# 이미지 콜백 함수 2
def right_image_callback(data):
    global image_right_
    # ROS 이미지 메시지를 OpenCV 이미지로 변환
    image_right = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    # print(f"Right {image_right.shape}")

    image_right = cv2.flip(image_right, 0)
    image_right = cv2.flip(image_right, 1)
    target_width = int(1920/4)  # 변경하려는 너비
    target_height = int(1080/4)  # 변경하려는 높이
    with lock:
        image_right_ = cv2.resize(image_right, (target_width, target_height))


def save_combined_image(para):
    global image_left_, image_right_, odom_

    # 두 이미지가 모두 수신되었을 때만 실행
    if image_left_ is not None and image_right_ is not None:
        # 이미지 합치기
        print(f"SHAPE {image_left_.shape} {image_right_.shape}")
        combined_image = cv2.hconcat([image_left_, image_right_])
        heartbeat(odom_, combined_image)
        # 이미지 저장
        print("Combined image saved.")


# def img_callback(msg):
#     global img, odom_
#     print("Received an image!")
#     img = bridge.compressed_imgmsg_to_cv2(msg)
#     img = cv2.flip(img, 0)
#     target_width = int(1920/4)  # 변경하려는 너비
#     target_height = int(1080/4)  # 변경하려는 높이
#     print(f"Resize{target_width} {target_height}")
#     img = cv2.resize(img, (target_width, target_height))
#     heartbeat(odom_)

def point_callback(msg):
    global firstRun, odom_
    # print("Received an pose!")
    odom_.pose = msg.pose



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

def save_db(odom, img):
    global scene_idx, data_list
    # save img
    img_path = f'{PACKAGE_DIR}/{RESULT_FOLDER}/{scene_idx}.jpg'  # 저장할 이미지 파일 경로 및 이름을 지정합니다
    disc_path = f'{PACKAGE_DIR}/{RESULT_FOLDER}/{scene_idx}.npy'  # 저장할 이미지 파일 경로 및 이름을 지정합니다




    kp_input_kaze, des_input_kaze = kaze.detectAndCompute(cv2.cvtColor(img[:,:,:], cv2.COLOR_BGR2GRAY), None)
    np.save(disc_path, des_input_kaze)
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
    file_path = f'{PACKAGE_DIR}/{RESULT_FOLDER}/meta.json'  # 저장할 JSON 파일 경로 및 이름을 지정합니다.
    with open(file_path, 'w') as file:
        file.write(json_str)
    scene_idx +=1

save_idx = 0
def heartbeat(odom, img):
    global pos_x, pos_y, pos_z, save_idx
    new_x, new_y, new_z = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
    
    os.system("clear")
    print(f"    POSE {new_x} {new_y} {new_z}")

    if firstRunCheck(new_x, new_y):
        pos_x , pos_y, pos_z = new_x, new_y, new_z
        return
    
    dist = getDist(pos_x, pos_y, pos_z, new_x, new_y, new_z)
    print(f"    DIST {dist}")

    if dist>SAVE_TERM:
    # if dist>SAVE_TERM or save_idx % 35 == 0:
        save_db(odom, img)
        pos_x , pos_y, pos_z = new_x, new_y, new_z
    save_idx += 1


if __name__ == '__main__':
    rospy.init_node("map_generator")
    print("hello world")
    # ------------------------- Subscribers ----------------------------------
    point_sub   = rospy.Subscriber(POINT_TOPIC_NAME, PoseWithCovarianceStamped, point_callback)
    left_img_sub     = rospy.Subscriber(LEFT_IMAGE_TOPIC_NAME, CompressedImage, left_image_callback)
    right_img_sub     = rospy.Subscriber(RIGHT_IMAGE_TOPIC_NAME, CompressedImage, right_image_callback)
    print(POINT_TOPIC_NAME)
    print(POINT_TOPIC_NAME)
    print(POINT_TOPIC_NAME)
    # ------------------------- Publishers -----------------------------------


    os.makedirs(os.path.join(PACKAGE_DIR, RESULT_FOLDER), exist_ok=True)

    r = rospy.Rate(30)
    sim_time = 0.0

    rospy.Timer(rospy.Duration(1.0), save_combined_image)

    node_pub_idx = 0
    while not rospy.is_shutdown():
        # print(f" Sim Time = {sim_time}")
        r.sleep()



# CONV net simple image retrieval
# https://towardsdatascience.com/a-hands-on-introduction-to-image-retrieval-in-deep-learning-with-pytorch-651cd6dba61e

# 