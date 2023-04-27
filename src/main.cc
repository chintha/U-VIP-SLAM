/**
This code is part of the development of Underwater SLAM called U-VIP SLAM. This code was devloped Using ORB SLAM as the base and further used code from VI-ORB SLAM,
Below shows original ORB SLAM informations

* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<fstream>
#include<ros/ros.h>
#include<ros/package.h>
#include<boost/thread.hpp>
#include<opencv2/core/core.hpp>



#include "Tracking.h"
#include "FramePublisher.h"
#include "Map.h"
#include "MapPublisher.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "parse_ros.h"
#include "IMU/configparam.h"


#include "Converter.h"


using namespace std;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "USLAM");
    ros::start();
    cout<<"opencv version "<< CV_VERSION<<endl;
    cout<<"opencv major version "<< CV_MAJOR_VERSION<<endl;

    cout << endl << "This code is part of the development of Underwater SLAM called U-VIP SLAM"<<endl << 
            "This code was devloped Using ORB SLAM as the base and further used code from VI-ORB SLAM"<<endl <<
            "Below shows original ORB SLAM informations"<<endl <<endl <<
            "ORB-SLAM Copyright (C) 2014 Raul Mur-Artal" << endl <<
            "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
            "This is free software, and you are welcome to redistribute it" << endl <<
            "under certain conditions. See LICENSE.txt." << endl;

    if(argc != 3)
    {
        std::cout<<"running with hardcode vocabulary and settings"<<std::endl;
        //cerr << endl << "Usage: rosrun USLAM USLAM path_to_vocabulary path_to_settings (absolute or relative to package directory)" << endl;
        //ros::shutdown();
        //return 1;
    }
    string strSettingsFile;

    if (argc ==3){
    // Load Settings and Check
    strSettingsFile = ros::package::getPath("USLAM")+"/"+argv[2];
    std::cout<<strSettingsFile<<std::endl;
    }
    else{
    //strSettingsFile = "/home/chinthaka/USLAM/Data/Settings.yaml";
    //strSettingsFile = "/home/chinthaka/USLAM/Data/euroc.yaml";
    //strSettingsFile = "/home/chinthaka/USLAM/Data/Settings_VI_Aqualoc_archiological.yaml";
    strSettingsFile = "/home/chinthaka/USLAM/Data/Settings_VI_Aqualoc_harbor.yaml"; // User can give the setting file in here or as an argument with the rosrun
    }

    
    cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        ROS_ERROR("Wrong path to settings file. Path must be absolut or relative to USLAM package directory.");
        ros::shutdown();
        return 1;
    }
    
    //Create Frame Publisher for image_view
    USLAM::FramePublisher FramePub;

    string strVocFile;
    if (argc ==3){
    strVocFile = ros::package::getPath("USLAM")+"/"+argv[1];
    std::cout<<strVocFile<<std::endl;
    }
    else{
    //strVocFile = "/home/chinthaka/USLAM/Data/ORBvoc.txt";  
    strVocFile = "/home/chinthaka/USLAM/Data/ORBvoc.txt";//hard code vocubulary path,
    }
    cout << endl << "Loading ORB Vocabulary. This could take a while." << endl;
    
    USLAM::ORBVocabulary Vocabulary;
    bool bVocLoad = Vocabulary.loadFromTextFile(strVocFile);

    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. Path must be absolut or relative to USLAM package directory." << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        ros::shutdown();
        return 1;
    }


    cout << "Vocabulary loaded!" << endl << endl;

    USLAM::ConfigParam config(fsSettings);

    //Create KeyFrame Database
    USLAM::KeyFrameDatabase Database(Vocabulary);

    //Create the map
    USLAM::Map World;

    FramePub.SetMap(&World);

    //Create Map Publisher for Rviz
    USLAM::MapPublisher MapPub(&World);

    //Initialize the Tracking Thread and launch
    USLAM::Tracking Tracker(&Vocabulary, &FramePub, &MapPub, &World, strSettingsFile,&config);
    boost::thread trackingThread(&USLAM::Tracking::Run,&Tracker);

    Tracker.SetKeyFrameDatabase(&Database);

    //Initialize the Local Mapping Thread and launch
    USLAM::LocalMapping LocalMapper(&World,&config,strSettingsFile);
    boost::thread localMappingThread(&USLAM::LocalMapping::Run,&LocalMapper);

    USLAM::LoopClosing LoopCloser(&World, &Database, &Vocabulary,&config);
    boost::thread loopClosingThread(&USLAM::LoopClosing::Run, &LoopCloser);
    
    //Set pointers between threads
    Tracker.SetLocalMapper(&LocalMapper);
    Tracker.SetLoopClosing(&LoopCloser);

    LocalMapper.SetTracker(&Tracker);
    LocalMapper.SetLoopCloser(&LoopCloser);
    //LocalMapper.SetLoopCloserHaloc(&loop_closing_haloc);


    LoopCloser.SetTracker(&Tracker);
    LoopCloser.SetLocalMapper(&LocalMapper);

    //loop_closing_haloc.SetTracker(&Tracker);
    //loop_closing_haloc.SetLocalMapper(&LocalMapper);

    //This "main" thread will show the current processed frame and publish the map
    float fps = fsSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    ros::Rate r(fps);

    while (ros::ok())
    {
        FramePub.Refresh();
        MapPub.Refresh();
        Tracker.CheckResetByPublishers();
        r.sleep();
    }

    // Save keyframe poses at the end of the execution
    ofstream f;

    vector<USLAM::KeyFrame*> vpKFs = World.GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),USLAM::KeyFrame::lId);

    cout << endl << "Saving Keyframe Trajectory to stamped_traj_estimate.txt" << endl;
    string strFile = ros::package::getPath("USLAM")+"/"+"stamped_traj_estimate.txt";
    f.open(strFile.c_str());
    f << fixed;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        USLAM::KeyFrame* pKF = vpKFs[i];

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = USLAM::Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }
    f.close();

    ros::shutdown();

	return 0;
}
