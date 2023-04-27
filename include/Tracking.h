/**
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

#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<sensor_msgs/Image.h>
#include<sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/FluidPressure.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <unistd.h>  
#include <ctime>    
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <chrono>
#include <thread> 
#include <random> 
#include <iterator>

#include <iomanip>
#include <fstream>                                                                                                                               #include<unistd.h

#include"FramePublisher.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"FrameKTL.h"
#include"ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapPublisher.h"
#include "src/IMU/configparam.h"
#include "src/IMU/imudata.h"
#include "colors.h"

#include<tf/transform_broadcaster.h>


namespace USLAM
{

class FramePublisher;
class Map;
class LocalMapping;
class LoopClosing;

class DepthData
{
public:
    DepthData(){}
    double timestamp;
    double depth;
    double depth_sd;
    
};

class Tracking
{  

public:
    Tracking(ORBVocabulary* pVoc, FramePublisher* pFramePublisher, MapPublisher* pMapPublisher, Map* pMap, string strSettingPath,ConfigParam* pParams);

    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        INITIALIZING=2,
        WORKING=3,
        LOST=4,
        IMU_RELOCALIZATION=5,
        R_INITIALIZING=6,
    };

    enum eMode{
        MONO=0,
        VI=1,
        VIP=2,
    };

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetKeyFrameDatabase(KeyFrameDatabase* pKFDB);
    Eigen::Vector3d z_axis;

    // This is the main function of the Tracking Thread
    void Run();

    void ForceRelocalisation();

    eTrackingState mState;
    eTrackingState mLastProcessedState;  
    eMode mMode;  

    // Current Frame
    FrameKTL mCurrentFrame;

    // Initialization Variables
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<int> mvp3pMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    std::vector<KeyFrame*> mLast20KF;

    
    FrameKTL mInitialFrame;
    FrameKTL mRecoveryFrame;

    vector<FrameKTL> mv20FramesReloc;
    IMUPreintegrator mIMUPreIntInTrack;

    void CheckResetByPublishers();
    bool mbCreateNewKFAfterReloc;
    ConfigParam* mpParams;
    std::vector<IMUData> mvIMUSinceLastKF;
    std::vector<IMUData> mvIMUSinceRecovery;
    void feed_imu_data(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am);
    void feed_depth_data(double timestamp, double depth);
    //static std::vector<IMUData> select_imu_readings(const std::vector<IMUData>& imu_data, double time0, double time1);
    std::vector<IMUData> select_imu_readings(double time0, double time1);
    bool select_depth_readings(double time0, double time1, double& depth,double& depth_time);
    bool select_depth_readings_txt(double time0, double time1, double& depth);
    bool TrackWithIMU(bool bMapUpdated=false);
    void PredictNavStateByIMU(bool bMapUpdated);
    IMUPreintegrator GetIMUPreIntSinceLastKF(FrameKTL* pCurF, KeyFrame* pLastKF, const std::vector<IMUData>& vIMUSInceLastKF);
    IMUPreintegrator GetIMUPreIntSinceLastFrame(FrameKTL* pCurF, FrameKTL* pLastF);
    bool TrackLocalMapWithIMU(bool bMapUpdated=false);
    void RecomputeIMUBiasAndCurrentNavstate(NavState& nscur);
    double ini_depth;
    

    // depth covarience value
    double depth_cov; // 0.002*10 for harbor 0.03 for archio// this is standerad diviation
    int IMU_Re_Count=0;
    int count=0;
    int Con=0;
    bool needKeyFrame;
    
    bool mbRelocBiasPrepare;
        //bool flagforscaleupdate=false;

    static IMUData interpolate_data(const IMUData imu_1, const IMUData imu_2, double timestamp) 
    {
        // time-distance lambda
        double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
        //cout << "lambda - " << lambda << endl;
        // interpolate between the two times
        IMUData data;
        data.timestamp = timestamp;
        data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
        data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
        return data;
    }
    
private:

    cv::Point2f undistort_point(cv::Point2f pt_in);
    void calculate_G();
    

protected:
    void CheckReplacedInLastFrame();
    void GrabImage(const cv::Mat &img,double TimeStamp,vector<IMUData> imu_data,bool has_depth, double depth, double depth_time);
    void FirstInitialization();
    void RecoveryInitialization();
    void Initialize();
    void Recovery_Initialize();
    void CreateInitialMap(cv::Mat &Rcw, cv::Mat &tcw, int &Lim_KP);
    void CreateRecoveryMap(cv::Mat &Rcw, cv::Mat &tcw,int &Lim_KP);

    void Reset();

    bool TrackWithPnP();

    bool RelocalisationRequested();
    bool Relocalisation();
    bool IMU_Relocalisation();    

    void UpdateReference();
    void UpdateReferencePoints();
    void UpdateReferenceKeyFrames();

    bool TrackLocalMap();
    void SearchReferencePointsInFrustum();
    bool TrackwithMotionModel();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame(bool flag);
    void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, cv::Mat &descriptors);
    void perform_matching(const std::vector<cv::Mat>& img0pyr, const std::vector<cv::Mat>& img1pyr,
                                std::vector<cv::KeyPoint>& kpts0, std::vector<cv::KeyPoint>& kpts1, 
                                std::vector<cv::KeyPoint>& kpts0_Un,std::vector<cv::KeyPoint>& kpts1_Un,
                                std::vector<uchar>& mask_out);


    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractor;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    

    //Publishers
    FramePublisher* mpFramePublisher;
    MapPublisher* mpMapPublisher;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;
    int x_lim;
    int y_lim;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    FrameKTL mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Mutex
    boost::mutex mMutexTrack;
    boost::mutex mMutexForceRelocalisation;
    boost::mutex mMutexThresds;

    //Reset
    bool mbPublisherStopped;
    bool mbReseting;
    boost::mutex mMutexReset;

    //Is relocalisation requested by an external thread? (loop closing)
    bool mbForceRelocalisation;

    //Motion Model
    bool mbMotionModel;
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    // Transfor broadcaster (for visualization in rviz)
    tf::TransformBroadcaster mTfBr;
    
    int min_px_dist = 20;// 20 works better for aqualoc
    int num_features;// there is a relation between this and the min_px_dist, look on the detection part
    //int grid_x = 15;
    //int grid_y = 15;
    double time_skip;
    int fastTh;
    int pyr_levels = 5; //5 previous value works better this paramert also defined in framesKTL, both should be equal
    cv::Size win_size = cv::Size(21,21);// 21 works better for aqualoc this paramert also defined in framesKTL.cc line no 34, both should be equal(21,12 perform better)
    std::vector<IMUData> imu_data;
    std::vector<DepthData> depth_data;
    bool image_enhance = true;
    bool Fisheye_Cam = false;
    int count_for_relocalize=0;
    std::vector<double> mdepth_Vec;
    std::vector<double> mdepth_time_Vec;


    // KTL Tracker
    
};
} //namespace USLAM
#endif // TRACKING_H