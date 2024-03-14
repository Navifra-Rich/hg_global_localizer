

This package performs image-based global localization



How to use make dataset
---

### 1. Get odometry of robot and images

```
roslaunch map_db_generator dataset_gen.launch
rosbag play magok_superstart_partial_path.bag
```
Just Run launch and Play bagfile.


Acquire location data by your own method (SLAM, gps, visual odometry...etc)
The type of sensor does not matter if you can get the pose of robot.


You have to set TOPIC NAME in launch file.
Output (img and pos json file) will be save in 'result' folder


다 끝나면 map_db_generator/result폴더에 저장되는데, 데이터셋 테스트셋 둘다 만들어야되니까 만들고나서 폴더이름 바꿔주면 좋습니다.


### 2. Execute localizer

```
roslaunch VPS_localizer kidnapper_magok.launch
```

위 런치만 슥 키면 됩니다.
현재는 테스트셋에서 임의로 하나 골라서 DB와 비교하고, 결과 저장하는 식으로 되어있습니다.
결과이미지는 VPS_localizer/result 에 저장됩니다.
origin은 입력한 데이터, kaze1은 입력 데이터와 가장 유사한 DB데이터입니다.

