import glob
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import cv2
import os

# 이미지를 1차원 벡터로 변환
def image_to_vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2])


class KNN():
    def __init__(self):
        self.knn = None
        self.package_path = ""
        self.dataset_path = ""
        return
    
    def setPaths(self):
        model_path = os.path.join(self.package_path, "data", "myModel_my.pkl")
        return
    
    def save_model(self, model_name="train_temp.pkl"):
        save_path = os.path.join(self.package_path, "data", f"{model_name}.pkl")
        joblib.dump(self.knn, save_path)
    
    def load_model(self, path="myModel_my.pkl"):
        model_path = os.path.join(self.package_path, "data", f"{path}.pkl")
        print(f" MODEL PATH {model_path}")
        self.knn = joblib.load(model_path)
        
    
    def train_knn(self, model_name):
        images = []
        labels = []
        files = glob.glob(f'{self.dataset_path}'+os.sep+"*")
        files.sort()
        files = files[:-1]
        files.sort(key = lambda x: int(x.split(os.sep)[-1][:-4]))   # get rid of .jpg, .meta

        class_idx = -1
        img_idx = -1

        for idx, file in enumerate(files):
            print(idx)
            print(file)
            print(file.split(os.sep)[-1][-3:])
            if idx%20==0:
                class_idx +=1
            img_idx +=1
            img = cv2.imread(file)
            vec = image_to_vector(img)
            print(f"PATH {class_idx}  {file}")
            images.append(vec)
            labels.append(class_idx)

        images = np.array(images)
        labels = np.array(labels)

        # k-NN 분류기 생성 및 학습
        k = class_idx  # 이웃의 수 설정
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(images, labels)
    
    def save_nearest_img(self, img_path):

        img = cv2.imread(img_path)

        new_vector = image_to_vector(img)
        new_vector = np.array([new_vector])  # 2차원 배열로 변환
        distances, indices = self.knn.kneighbors(new_vector, n_neighbors=10)

        print(indices)
        img_path_list = []
        img_idx_list = []
        for idx in indices[0]:
            img_idx_list.append(idx)
            img_path_list.append(os.path.join(self.dataset_path, f"{idx}.jpg"))
        self.publish_imgs(img_path, img_path_list)
    
        return img_idx_list # = [ [idx, weight], ...... ]
    
    def publish_imgs(self, origin, near_list):
        origin_img = cv2.imread(origin)
        near_img_list = []
        for idx in range(0, len(near_list),2):
            img1 = cv2.imread(near_list[idx])
            img2 = cv2.imread(near_list[idx+1])
            v_combined = np.vstack([img1,img2])
            near_img_list.append(v_combined)
        near_combined = np.hstack(near_img_list)

        cv2.imwrite(os.path.join(self.package_path, 'result', 'Combined_Images.jpg'), near_combined)
        cv2.imwrite(os.path.join(self.package_path, 'result', 'Origin_Images.jpg'), origin_img)

    def predict_one(self, img_path):
        img = cv2.imread(img_path)
        new_vector = image_to_vector(img)
        new_vector = np.array([new_vector])  # 2차원 배열로 변환
        predicted_label = self.knn.predict(new_vector)
        print("Predicted label:", predicted_label)

def main():
    knn = KNN()
 
    return
if __name__=="__main__":
    main()
    
# 1. AP access
# name : A200-0733 , pw : bvUNdWJo  
# 2. keygen
# ssh-keygen -f "/home/hgnaseel/.ssh/known_hosts" -R "10.42.0.1"
# 3. ssh
# ssh husky@10.42.0.1, pw : clearpath