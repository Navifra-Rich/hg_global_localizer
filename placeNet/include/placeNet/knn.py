import glob
import time
PATH = "/home/hgnaseel/Downloads/place_data"

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import cv2
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# 이미지를 1차원 벡터로 변환하는 함수
def image_to_vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2])



class KNN():
    def __init__(self):
        self.knn = None
        self.model_save_path = "/home/hgnaseel/Downloads/placeNet"
        self.package_path = ""
        self.json_path = ""
        self.meta_path = ""
        self.model_path = ""
        self.dataset_path = ""
        self.json_data = None
        return
    
    def setPaths(self):

        self.model_path = os.path.join(self.package_path, "data", "myModel_my.pkl")
        self.json_path = os.path.join(self.package_path, "data", "train_my.json")
        # self.meta_path = os.path.join(self.package_path, "data", "meta.json")
        return
    def save_model(self, path="myModel_my.pkl"):
        # 학습된 모델 저장
        save_path = os.path.join(self.model_save_path, path)
        print(save_path)
        print(save_path)
        joblib.dump(self.knn, save_path)
        return
    
    def load_model(self, path="myModel_my.pkl"):
        # self.knn = joblib.load(os.path.join(self.model_save_path, path))
        print(f" MODEL PATH {self.model_path}")
        self.knn = joblib.load(self.model_path)
        self.json_data = json.loads(open(self.json_path).read())
        
    def train_knn(self):
        images = []
        labels = []
        folders = glob.glob(self.dataset_path+os.sep+"*")

        class_idx = -1
        img_idx = -1
        json_list = []
        for forlder in folders:
            folders2 = glob.glob(forlder+os.sep+"*")
            for folder2 in folders2:
                if class_idx == 5:
                    break
                files = glob.glob(folder2+os.sep+"*")
                if len(files)<3:
                    continue
                class_idx +=1
                print(f"Class Idx {class_idx} {folder2.split(os.sep)[-1]}")
                for file in files:
                    img_idx +=1
                    img = cv2.imread(file)
                    json_list.append([img_idx, file])
                    vec = image_to_vector(img)
                    images.append(vec)
                    labels.append(class_idx)
        json_data = json.dumps(json_list, indent=2)

        f = open("./json/train_my.json", 'w')
        f.write(json_data)
        images = np.array(images)
        labels = np.array(labels)

        # k-NN 분류기 생성 및 학습
        k = class_idx  # 이웃의 수 설정
        k = 5
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(images, labels)
        return
    
    def train_knn_my(self):
        images = []
        labels = []
        files = glob.glob(f'/home/hgnaseel/data/global_map'+os.sep+"*")
        files.sort()
        


        class_idx = -1
        img_idx = -1
        json_list = []

        # print(f"Class Idx {class_idx} {folder2.split(os.sep)[-1]}")
        for idx, file in enumerate(files):
            print(idx)
            print(file)
            print(file.split(os.sep)[-1][-3:])
            if file.split(os.sep)[-1][-3:]!="jpg":
                continue
            if idx%10==0:
                class_idx +=1
            img_idx +=1
            # if img_idx%2==0:
            #     continue
            img = cv2.imread(file)
            vec = image_to_vector(img)
            print(f"PATH {class_idx}  {file}")
            images.append(vec)
            labels.append(class_idx)
            json_list.append([class_idx, file])
        json_data = json.dumps(json_list, indent=2)

        f = open("./json/train_my.json", 'w')
        f.write(json_data)
        images = np.array(images)
        labels = np.array(labels)

        # k-NN 분류기 생성 및 학습
        k = class_idx  # 이웃의 수 설정
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(images, labels)
        return
    
    def save_nearest_img(self, img_path):

        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(self.package_path, "result", "origin.jpg"), img)
        # img = cv2.resize(img, (img.shape[0], img.shape[1]))

        new_vector = image_to_vector(img)
        new_vector = np.array([new_vector])  # 2차원 배열로 변환
        distances, indices = self.knn.kneighbors(new_vector, n_neighbors=10)

        # print("Weight :", distances)
        # print("Nearest Images:", indices)

        img_list = []
        near_img_path_list = []
        for i, knn_idx in enumerate(indices[0]):
            print(self.json_data[knn_idx])
            img_idx = [int(self.json_data[knn_idx][1].split(os.sep)[-1][:-4]), distances[0,i]]
            img_list.append(img_idx)
            img = cv2.imread(self.json_data[knn_idx][1])
            near_img_path_list.append(self.json_data[knn_idx][1])
            # cv2.imwrite(os.path.join(self.package_path, "result", f"{self.json_data[knn_idx][0]}.jpg"), img)
        self.publish_imgs(img_path, near_img_path_list)
    
        return img_list # = [ [idx, weight], ...... ]
    def publish_imgs(self, origin, near_list):
        origin_img = cv2.imread(origin)
        near_img_list = []
        for idx in range(0, len(near_list),2):
            img1 = cv2.imread(near_list[idx])
            img2 = cv2.imread(near_list[idx+1])
            v_combined = np.vstack([img1,img2])
            near_img_list.append(v_combined)
        near_combined = np.hstack(near_img_list)

        # cv2.namedWindow('Combined Images', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Combined Images', int(3700/1.5), int(1000/1.5))
        print(f"나는어디? {os.path.join(self.package_path, 'result', 'Combined_Images.jpg')}")
        cv2.imwrite(os.path.join(self.package_path, 'result', 'Combined_Images.jpg'), near_combined)
        cv2.imwrite(os.path.join(self.package_path, 'result', 'Origin_Images.jpg'), origin_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return
    def predict_one(self, img_path):

        img = cv2.imread(img_path)
        # img = cv2.resize(img, (img.shape[0], img.shape[1]))

        print(f"PATH {img_path}")
        print(f"IMG  {img.shape}")
        new_vector = image_to_vector(img)
        new_vector = np.array([new_vector])  # 2차원 배열로 변환
        print("--------------------------")
        print(new_vector)
        predicted_label = self.knn.predict(new_vector)
        print("Predicted label:", predicted_label)
        return

def main():
    knn = KNN()
    # knn.train_knn()
    # knn.train_knn_my()
    # knn.save_model()
    # return

    knn.load_model()
    # knn.save_nearest_img("/home/hgnaseel/Downloads/place_data/t/television_studio/gsun_0d71cc4cb26e9c85df0455b5820f06dc.jpg")
    # knn.save_nearest_img("/home/hgnaseel/Downloads/place_data/r/restaurant/gsun_0bbf131009b417059d0bbe7811ea61d6.jpg")
    # knn.save_nearest_img("/home/hgnaseel/Downloads/place_data/h/highway/gsun_0f8f622b0aa2628c454ed26290ab8057.jpg")
    start = time.time()
    # knn.save_nearest_img("/home/hgnaseel/data/global_map/17.jpg")
    # knn.save_nearest_img("/home/hgnaseel/data/global_map/57.jpg")
    knn.save_nearest_img("/home/hgnaseel/data/global_map/115.jpg")
    # knn.save_nearest_img("/home/hgnaseel/data/global_map/277.jpg")
    # knn.predict_one("/home/hgnaseel/Downloads/place_data/r/restaurant/gsun_0bbf131009b417059d0bbe7811ea61d6.jpg")
    # knn.predict_one("/home/hgnaseel/Downloads/place_data/h/hospital_room/gsun_0b78e6b904e3527a6ba1f653411f30bb.jpg")

    print(f"    TIME = {time.time()-start}")
    return
if __name__=="__main__":
    main()
    