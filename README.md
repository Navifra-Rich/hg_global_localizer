

This package performs image-based global localization




Quickstart
```
(launch main module)
roslaunch placeNet kidnapper.launch

(rostopic pub (img idx))
rostopic pub /place_idx std_msgs/Int64 "data: 50" -r 10
```


Train dataset
```
blabla~
```



How to use make dataset
---

### 1. Get odometry of robot and images

Acquire location data by your own method (SLAM, gps, visual odometry...etc)
The type of sensor does not matter if you can get the pose of robot.

### 2. Coupling the two types of data

Combine the two type of data(Image + Odom) in the form of rosbag.

```
Odom  -> nav_msgs   ::Odometry
Image -> sensor_msg ::Image
```
you can set the topic name in the launch file.

but you've to modify the source code if you want to change the topic type


### 3. Execute dataset generator

and execute below launch.

```
roslaunch map_db_generator dataset_gen.launch
```

you can earn dataset in 
yours/map_db_generator/result

### 4. Train KNN classifier

train your knn classifier

```

```