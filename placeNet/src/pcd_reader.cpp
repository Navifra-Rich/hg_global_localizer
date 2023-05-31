#include<ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
ros::Publisher map_pub;
std::string param_value;

void readPCD(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (param_value, *cloud) == -1) //* load the file
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    else
      return;
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
}

int main (int argc, char** argv)
{
    ros::init (argc, argv, "pcd_reader");
    ros::NodeHandle nh;
    nh.getParam("dir_pcd", param_value);

    map_pub = nh.advertise<sensor_msgs::PointCloud2> ("global_map", 1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>); 

    readPCD(cloud);
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "map";
    while(true){

      map_pub.publish (output);
      sleep(1);
      ros::spinOnce ();
    }
    // Spin
}
