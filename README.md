# U-VIP-SLAM
Underwater Visual Inertial Pressure SLAM (U-VIP SLAM), a new robust monocular visual-inertial-pressure, real-time state estimator, which includes all of the essential components of a full SLAM system and is equipped with loop closure capabilities tailored to the underwater environment.
We develop the code on top of ORB SLAM

Research Paper

Amarasinghe, C., Rathnaweera, A. & Maithripala, S. U-VIP-SLAM: Underwater Visual-Inertial-Pressure SLAM for Navigation of Turbid and Dynamic Environments. Arab J Sci Eng (2023). https://doi.org/10.1007/s13369-023-07906-6

Youtube Videos
https://www.youtube.com/watch?v=HvrYjkh9WR4
https://www.youtube.com/watch?v=3J8NYGkPuwo
https://www.youtube.com/watch?v=kqdrXbFp8Ns&t=1s

#2. Prerequisites (dependencies)
We tested all in the Ubuntu 18.04.6 LTS

##2.1 Boost

We use the Boost library to launch the different threads of our SLAM system.

sudo apt-get install libboost-all-dev 

##2.2 ROS 
We use ROS to receive images from the camera or from a recorded sequence (rosbag), and for visualization (rviz, image_view). We have tested ORB-SLAM in Ubuntu 18.04.6 LTS with ROS Melodic. If you do not have already installed ROS in your computer, we recommend you to install the Full-Desktop version of ROS melodic (http://wiki.ros.org/melodic/Installation/Ubuntu).

##2.3 
OpenCV We use OpenCV to manipulate images and features. We tested OpenCV 3.4.6 Dowload and install instructions can be found at: http://opencv.org/

##2.4 
g2o (included in Thirdparty folder) We use a modified version of g2o (see original at https://github.com/RainerKuemmerle/g2o) to perform optimizations. In order to compile g2o you will need to have installed Eigen3 (We used eigen 3.3.4).

sudo apt-get install libeigen3-dev

##2.5 
DBoW2 (included in Thirdparty) We make use of some components of the DBoW2 and DLib library (see original at https://github.com/dorian3d/DBoW2) for place recognition and feature matching. There are no additional dependencies to compile DBoW2.

##2.6
pangolin, install pangolin from https://github.com/stevenlovegrove/Pangolin We use version 0.4

#3. Installation

Make sure you have installed ROS and all library dependencies (boost, eigen3, opencv, blas, lapack, cholmod, PCL).

Clone the repository:
git clone https://github.com/chintha/U-VIP-SLAM.git

Add the path where you cloned U-VIP-SLAM to the ROS_PACKAGE_PATH environment variable. To do this, modify your .bashrc and add at the bottom the following line (replace PATH_TO_PARENT_OF_ORB_SLAM):

export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH_TO_PARENT_OF_U-VIP_SLAM

Build g2o. Go into PATH_TO_PARENT_OF_U-VIP_SLAM/Thirdparty/g2o/ and execute:

     mkdir build
     cd build
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make 

    Tip: To achieve the best performance in your computer, set your favorite compilation flags in line 61 and 62 of Thirdparty/g2o/CMakeLists.txt (by default -03 -march=native)

Build DBoW2. Go into PATH_TO_PARENT_OF_U-VIP_SLAM/Thirdparty/DBoW2/ and execute:

     mkdir build
     cd build
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make  

    Tip: Set your favorite compilation flags in line 4 and 5 of Thirdparty/DBoW2/CMakeLists.txt (by default -03 -march=native)

Build U-VIP_SLAM. In the U-VIP_SLAM root execute:

    

     mkdir build
     cd build
     cmake .. -DROS_BUILD_TYPE=Release
     make

    Tip: Set your favorite compilation flags in line 12 and 13 of ./CMakeLists.txt (by default -03 -march=native)

#4. Usage

See section 5 to run the Example Sequence.
    Launch ros : get a terminal and run roscore
    Launch U-VIP-SLAM from the terminal (roscore should have been already executed): navigate to the U-VIP root folder and get a new terminal. run 
    
    rosrun USLAM USLAM PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE
    you can download the Vocabulary file in https://drive.google.com/file/d/15wJsFqKzJYPKV09FJil7l8xKtn7vDFsn/view?usp=sharing
    you can find the setting files in data folder in hte repository, If you are using your own data, make a setting file accordingly

You have to provide the path to the ORB vocabulary and to the settings file. The paths must be absolute.

    The last processed frame is published to the topic /UW_SLAM/Frame. You can visualize it using image_view:

     rosrun image_view image_view image:=/UW_SLAM/Frame _autosize:=true

    The map is published to the topic /USLAM/Map, the current camera pose and global world coordinate origin are sent through /tf in frames /USLAM/Camera and /USLAM/World respectively. Run rviz to visualize the map:
    navigate to the root U-VIP folder and get a new terminal, then run,
    rosrun rviz rviz -d Data/rviz.rviz

U-VIP_SLAM will receive the images from the topic /camera/image_raw. Adjust the rosbag path ('bagfile:') in the setting file. If you have a sequence with individual image files, you will need to generate a bag from them. We provide a tool to do that: https://github.com/raulmur/BagFromImages.


#5. Example Sequence 
We provide the settings and the rosbag of an example sequence from Aqualoc dataset from https://www.lirmm.fr/aqualoc/
    
    Download one of the rosbag file: For testing we recommond downloding harbor_sequence_1.bag, Uncompress the file

    Launch U-VIP_SLAM with the settings for the example sequence. You should have the vocabulary file, and the setting file. Setting files can be found in data folder. for the aqualoc harbor dataset use Settings_VI_Aqualoc_harbor.yaml.  For Aqualoc archiological sequeces, use Settings_VI_Aqualoc_archiological.yaml. 
    **** Important **** Remember to change the rosbag path inside the setting files.

#6. The Settings File

U-VIP_SLAM reads the camera calibration and setting parameters from a YAML file. We provide an example in Data/****.yaml, where you will find all parameters and their description. We use the camera calibration model of OpenCV.

Please make sure you write and call your own settings file for your camera (copy the example file and modify the calibration)

#7. Failure Modes

You should expect to achieve good results in sequences similar to those in which we show results in our paper [1], in terms of camera movement and texture in the environment. In general our Monocular SLAM solution is expected to have a bad time in the following situations:

    No translation at system initialization (or too much rotation).
    Pure rotations in exploration.
    Low texture environments.
    Many (or big) moving objects, especially if they move slowly.

The system is able to initialize from planar and non-planar scenes. In the case of planar scenes, depending on the camera movement relative to the plane, it is possible that the system refuses to initialize, see the paper [1] for details.
About

[1] Amarasinghe, C., Rathnaweera, A. & Maithripala, S. U-VIP-SLAM: Underwater Visual-Inertial-Pressure SLAM for Navigation of Turbid and Dynamic Environments. Arab J Sci Eng (2023). https://doi.org/10.1007/s13369-023-07906-6

