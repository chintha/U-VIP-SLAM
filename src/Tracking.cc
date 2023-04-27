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

#include"Tracking.h"
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>


#include<opencv2/opencv.hpp>

#include"ORBmatcher.h"
#include"FramePublisher.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<fstream>


using namespace std;

namespace USLAM
{

    Tracking::Tracking(ORBVocabulary* pVoc, FramePublisher *pFramePublisher, MapPublisher *pMapPublisher, Map *pMap, string strSettingPath,ConfigParam* pParams):
        mState(NO_IMAGES_YET), mpORBVocabulary(pVoc), mpFramePublisher(pFramePublisher), mpMapPublisher(pMapPublisher), mpMap(pMap),
        mnLastRelocFrameId(0), mbPublisherStopped(false), mbReseting(false), mbForceRelocalisation(false), mbMotionModel(false)
    {
        mbCreateNewKFAfterReloc = false;
        mpParams = pParams;
        mbRelocBiasPrepare = false;
        needKeyFrame=false;
        
        
        // Load camera parameters from settings file
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        int mode = fSettings["Mode"];
        if(mode ==0){
             mMode = MONO;
        }
        else if(mode==1){
            mMode=VI;

        }
        else if(mode==2){
            mMode = VIP;
        }

        int En = fSettings["Enhance"];
        image_enhance = En;
        int fisheye= fSettings["Camera.Fisheye"];
        Fisheye_Cam = fisheye;
        min_px_dist = fSettings["Px_distance"];
        pyr_levels = fSettings["Pyramid.Level"];
        int win_z = fSettings["Window.Size"];
        win_size = cv::Size(win_z,win_z);
        depth_cov = fSettings["depth.noise"];
       
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        DistCoef.copyTo(mDistCoef);

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        //cv::Size size ={fSettings["Camera.col"],fSettings["Camera.row"]};
        //cv::Mat new_mK;
        //new_mK  = cv::getOptimalNewCameraMatrix(K, mDistCoef, size, 1, size, 0);
        //new_mK.copyTo(mK);
        K.copyTo(mK);

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=20;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 4;
        mMaxFrames = 15;


        cout << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if(mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        num_features=nFeatures;
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        fastTh = fSettings["ORBextractor.fastTh"];    
        int Score = fSettings["ORBextractor.nScoreType"];
        double skip_time = fSettings["test.DiscardTime"];
        int LoopC = fSettings["LoopC"];
        time_skip=skip_time;
        assert(Score==1 || Score==0);

        mpORBextractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,Score,fastTh);

        cout << endl  << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Fast Threshold: " << fastTh << endl;
        cout << "- Loop Closer Enabled: " << LoopC << endl;
        if(Score==0)
            cout << "- Score: HARRIS" << endl;
        else
            cout << "- Score: FAST" << endl;


        // ORB extractor for initialization
        // Initialization uses only points from the finest scale level
        mpIniORBextractor = new ORBextractor(nFeatures*2,1.2,8,Score,fastTh);  
        tf::Transform tfT;
        tfT.setIdentity();
        mTfBr.sendTransform(tf::StampedTransform(tfT,ros::Time::now(), "/USLAM/World", "/USLAM/Camera"));
    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper=pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
    {
        mpLoopClosing=pLoopClosing;
    }

    void Tracking::SetKeyFrameDatabase(KeyFrameDatabase *pKFDB)
    {
        mpKeyFrameDB = pKFDB;
    }

    void Tracking::Run()
    {
        //ros::NodeHandle nodeHandler;
        //ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 1, &Tracking::GrabImage, this);

        //ros::spin();
        double timeStart;//remove this later
        time_t tstart, tend;
        int Frame_Count=0;
        ros::NodeHandle nh;
        std::string path_to_bag;
        double bag_start, bag_durr;
        rosbag::Bag bag;
        rosbag::View view_full;
        rosbag::View view;
        std::string topic_camera,topic_imu,topic_depth;
        //nh.param<std::string>("topic_camera", topic_camera, "/camera/image_raw");
        nh.param<std::string>("topic_camera", topic_camera, mpParams->_imageTopic);// toppic for openvins exmple data
        nh.param<std::string>("topic_imu", topic_imu, mpParams->_imuTopic);
        nh.param<std::string>("topic_depth", topic_depth, mpParams->_depthTopic);

        //nh.param<std::string>("path_bag", path_to_bag, "/home/chinthaka/USLAM/sequence_3.bag");
        nh.param<std::string>("path_bag", path_to_bag, mpParams->_bagfile);
        //nh.param<std::string>("path_bag", path_to_bag, "/home/chinthaka/DataSet/OV_original/V1_01_easy.bag");
        //nh.param<std::string>("path_bag", path_to_bag, "/home/chinthaka/USLAM/Example.bag");
        ROS_INFO("ros bag path is: %s", path_to_bag.c_str());

        nh.param<double>("bag_start", bag_start, 0);
        nh.param<double>("bag_durr", bag_durr, -1);
        
        bag.open(path_to_bag, rosbag::bagmode::Read);
        

        view_full.addQuery(bag);
        ros::Time time_init = view_full.getBeginTime();
        time_init += ros::Duration(bag_start);
        ros::Time time_finish = (bag_durr < 0)? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
        ROS_INFO("time start = %.6f", time_init.toSec());
        ROS_INFO("time end   = %.6f", time_finish.toSec());
        view.addQuery(bag, time_init, time_finish);
        double time_stamp = time_init.toSec();
        double time_stamp_buffer1 = time_init.toSec();
        double time_stamp_buffer2 = time_init.toSec();
        double cutoff_Time = time_stamp + time_skip;;
        double Pre_depth;
        bool first_depth = true;
        bool has_img = false;  
        cv::Mat img0;
        cv::Mat img0_buffer;
        int imgCount=0;
        vector<IMUData> prop_data;

        // Check to make sure we have data to play
        if (view.size() == 0) {
            ROS_ERROR("No messages to play on specified topics.  Exiting.");
            
            ros::shutdown();
            //return EXIT_FAILURE;
        }
        
        cout << "rosbag Loaded!" << endl << endl;
        
        tstart = time(0);
        bool flag=true;
        
        for (const rosbag::MessageInstance& m : view) {
            if (!ros::ok()){
                break;
            }
            
            sensor_msgs::Imu::ConstPtr s1 = m.instantiate<sensor_msgs::Imu>();
            if (s1 != nullptr && m.getTopic() == topic_imu) {
                // convert into correct format
                double timem = (*s1).header.stamp.toSec();
                Eigen::Matrix<double, 3, 1> wm, am;
                
                wm << (*s1).angular_velocity.x, (*s1).angular_velocity.y, (*s1).angular_velocity.z;
                am << (*s1).linear_acceleration.x, (*s1).linear_acceleration.y, (*s1).linear_acceleration.z;
                //cout<<"wm"<<endl<<wm<<endl;
                //cout<<"am"<<endl<<am<<endl;
                /*
                if(wm(0)>1 ||wm(1)>1|| wm(2)>1){
                    cerr<<"gyro measurment out or CRAZE+"<<endl;
                }
                if(am(0)>15 ||am(1)>15|| am(2)>15){
                    cerr<<"acc measurment out or CRAZE+"<<endl;
                }

                if(wm(0)<-1 ||wm(1)<-1|| wm(2)<-1){
                    cerr<<"gyro measurment out or CRAZE-"<<endl;
                }
                if(am(0)<-15 ||am(1)<-15|| am(2)<-15){
                    cerr<<"acc measurment out or CRAZE-"<<endl;
                }
                */
                // send it to our VIO system
                feed_imu_data(timem, wm, am);
                
            }

            sensor_msgs::FluidPressure::ConstPtr s2 = m.instantiate<sensor_msgs::FluidPressure>();
            if (s2 != nullptr && m.getTopic() == topic_depth) {
                
                double timem = (*s2).header.stamp.toSec();
                double depth= (*s2).fluid_pressure;
                if(!first_depth){
                    // convert into correct format
                    if (depth>Pre_depth-2 && depth < Pre_depth+2){
                        feed_depth_data(timem,depth);   
                        Pre_depth=depth;      
                    }
                    else{
                        cerr<<"out of order depth measurment"<<Pre_depth<<" vs "<< depth<<endl;
                        feed_depth_data(timem,Pre_depth);
                    }  
                } 
                else{
                    first_depth=false;
                    Pre_depth=depth;
                    feed_depth_data(timem,depth);  
                }   
            }


            sensor_msgs::Image::ConstPtr s0 = m.instantiate<sensor_msgs::Image>();
            
            if (s0 != nullptr && m.getTopic() == topic_camera){
               
                cv_bridge::CvImageConstPtr cv_ptr;
                try {
                        cv_ptr = cv_bridge::toCvShare(s0, sensor_msgs::image_encodings::MONO8);
                    } 
                    catch (cv_bridge::Exception &e) {
                        ROS_ERROR("cv_bridge exception: %s", e.what());
                        continue;
                    }

                ROS_ASSERT(cv_ptr->image.channels()==3 || cv_ptr->image.channels()==1);

                if(cv_ptr->image.channels()==3)
                {
                    if(mbRGB)
                        cvtColor(cv_ptr->image, img0, CV_RGB2GRAY);
                    else
                        cvtColor(cv_ptr->image, img0, CV_BGR2GRAY);
                }
                else if(cv_ptr->image.channels()==1)
                {
                    cv_ptr->image.copyTo(img0);
                }
                   
                has_img = true;
                img0 = cv_ptr->image.clone();
                time_stamp = cv_ptr->header.stamp.toSec();
                time_stamp = time_stamp + mpParams->GetImageDelayToIMU();
                  
            }
            if(has_img && imgCount<2) {
            has_img = false;
            time_stamp_buffer2 = time_stamp_buffer1;
            time_stamp_buffer1 = time_stamp;
            img0_buffer = img0.clone();
            timeStart=time_stamp;// remove this later
            imgCount++;
            }

            if(has_img){

                double time0 = time_stamp_buffer2;
                double time1 = time_stamp_buffer1;
                double a_depth= 0.0;
                double a_depth_time= 0.0;
                bool has_depth=false;
                if(mMode!=MONO){
                    prop_data = select_imu_readings(time0,time1);
                    if(prop_data.empty()){
                    cerr<<"no imu Data  Check the Mode of operation V-Only ? VI ? VIP ?"<<endl;
                }
                }
                if(mMode == VIP){
                    has_depth = select_depth_readings(time0,time1,a_depth, a_depth_time);
                    //has_depth = select_depth_readings_txt(time0,time1,a_depth); // this function for selecting groundthruth depth for EuRoC

                }
                
                
                //vimunew.insert(vimunew.end(), mvIMUData.begin(), mvIMUData.end());
                //flagforscaleupdate = true;
                
                //Sending Blank Image
                /*if((time_stamp-timeStart)>20 && flag){
                cout<<"sending Balnk image"<<endl;
                img0 = cv::Mat(img0.size(), CV_8UC1, Scalar(100));
                //cv::imshow("Rowresult.png", img0);
                //cv::waitKey(-1);
                flag=false;
                }*/

                if (cutoff_Time<time_stamp_buffer1){
                    //usleep(100000);
                    Tracking::GrabImage(img0_buffer,time_stamp_buffer1,prop_data, has_depth,a_depth, a_depth_time);
                }
                else{
                    cout<<"skipping initial stationaly frames....."<<endl;
                }
                //flagforscaleupdate = false;
                Frame_Count++; 
                time_stamp_buffer2 = time_stamp_buffer1;
                time_stamp_buffer1 = time_stamp;
                img0_buffer = img0.clone();
                has_img = false;
            }

        }

        cout << "End of Datastream.. exciting...." << endl << endl;
        tend = time(0); 
        cout << "It took "<< difftime(tend, tstart) <<" seconds for total number of frame: "<< Frame_Count<<"  FPS = "<<Frame_Count/difftime(tend, tstart)<<endl; 
        ros::shutdown();
        
    }

    void Tracking::GrabImage(const cv::Mat &img,double TimeStamp, vector<IMUData> imu_data, bool has_depth, double depth, double depth_time)
    {
       
        /*
        cv::imshow("Rowresult.png", im);
        cv::waitKey(10);
        cv::imshow("result11.png", dst);
        cv::waitKey(10);
        cv::imshow("result22.png", dst1);
        cv::waitKey(10);
        cv::imshow("calibresult.png", undistort);
        cv::waitKey(-1);

        */
        
        boost::mutex::scoped_lock lock(mpMap->mMutexMapUpdate);
       
        cv::Mat im=img.clone();
        //cv::imshow("row image", im);
        // cv::waitKey(10);
        //cv::equalizeHist(im, im);
        if (image_enhance){
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(4);
            cv::Size tilesize ={12,12};
            clahe->setTilesGridSize(tilesize);
            clahe->apply(im, im);
        }

        mCurrentFrame = FrameKTL(im,TimeStamp,mpORBextractor,mpORBVocabulary,mK,mDistCoef,imu_data,has_depth,depth, depth_time,pyr_levels,win_size);
        //cout<<cv_ptr->header.stamp<<endl;

        // If we didn't have any successful tracks last time, just extract this time
        // This also handles, the tracking initalization on the first call to this extractor                                                                    
        if(mState==NO_IMAGES_YET || mState==NOT_INITIALIZED || mState==LOST || mState==IMU_RELOCALIZATION) {
            // Detect new features
            perform_detection_monocular(mCurrentFrame.imgpyr, mCurrentFrame.mvKeys, mCurrentFrame.mDescriptors);
            mCurrentFrame.SetN((int)mCurrentFrame.mvKeys.size());
            mCurrentFrame.mvKeysUn.resize(mCurrentFrame.GetN());
            mCurrentFrame.mvKeysUn = mCurrentFrame.mvKeys;
            mCurrentFrame.compute_descriptors();
            mvIniMatches.resize(mCurrentFrame.GetN());
            mCurrentFrame.ComputeImageBounds(); // this will set image dimentions 
            mCurrentFrame.mvpMapPoints.resize(mCurrentFrame.GetN());
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
            cv::Point2f pts,pts_Un;
            for(size_t i=0; i<mCurrentFrame.GetN(); i++) {
                pts = mCurrentFrame.mvKeys.at(i).pt;
                pts_Un = undistort_point(pts);
                mCurrentFrame.mvKeysUn.at(i).pt = pts_Un;
                mvIniMatches[i]=i;
            }
            x_lim=mCurrentFrame.im.cols;
            y_lim=mCurrentFrame.im.rows;   
        }
        else{
            //std::cout<<"entering tracking.."<<std::endl;
            std::vector<uchar> mask_ll;
            mCurrentFrame.mvpMapPoints = mLastFrame.mvpMapPoints;
            // Lets track temporally
            if (mState== WORKING){
                perform_detection_monocular(mLastFrame.imgpyr, mLastFrame.mvKeys,mLastFrame.mDescriptors);
                mCurrentFrame.mvpMapPoints.resize(mLastFrame.mvKeys.size(),static_cast<MapPoint*>(NULL));
                mLastFrame.mvpMapPoints.resize(mLastFrame.mvKeys.size(),static_cast<MapPoint*>(NULL));
                mvIniMatches.resize(mLastFrame.mvKeys.size(),2000);// i here is just to make the positive
            }
            
            mCurrentFrame.mvKeys = mLastFrame.mvKeys;
            mCurrentFrame.mvKeysUn.resize(mLastFrame.mvKeys.size());
            mLastFrame.mvKeysUn.resize(mLastFrame.mvKeys.size());
            

            perform_matching(mLastFrame.imgpyr,mCurrentFrame.imgpyr,mLastFrame.mvKeys,mCurrentFrame.mvKeys,
                                mLastFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mask_ll);// improve this function to avoid  fault keypoints

            
            if(mask_ll.empty()) {
                std::cout<<"ransac failed for KTL Tracking.........."<<std::endl;
                //mLastFrame = FrameKTL(mCurrentFrame);
                //printf( "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n RESET");
            }
            
            for(size_t i=0; i<mCurrentFrame.mvKeys.size(); i++) {  
                //std::cout<<"mask_all  "<<+mask_ll[i]<<std::endl;   
                // Ensure we do not have any bad KLT tracks (i.e., points are negative)
                
                if(mCurrentFrame.mvKeys[i].pt.x < 2 || mCurrentFrame.mvKeys[i].pt.y < 2){
                    mvIniMatches[i]= -1;
                    continue;          
                }
                if(mCurrentFrame.mvKeys[i].pt.x > x_lim-2 || mCurrentFrame.mvKeys[i].pt.y > y_lim-2){
                    mvIniMatches[i]= -1;
                    continue;          
                }
                if(!mask_ll[i]){
                    mvIniMatches[i]= -1;
                }
                     
            }
            std::vector<int> des_index;
            int pt_des=-1;
            auto it0 = mvIniMatches.begin();
            auto it1 = mCurrentFrame.mvKeys.begin();
            auto it2 = mCurrentFrame.mvKeysUn.begin();
            auto it3 = mCurrentFrame.mvpMapPoints.begin();
            while(it0 != mvIniMatches.end()) {
                int K = *it0;
                pt_des++;
                if(K<0) {
                    it0 = mvIniMatches.erase(it0);
                    it1 = mCurrentFrame.mvKeys.erase(it1);
                    it2 = mCurrentFrame.mvKeysUn.erase(it2);
                    MapPoint* pMP = *it3;
                    if(pMP){  
                        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                        pMP->mbTrackInView = false;
                    }
                    it3 = mCurrentFrame.mvpMapPoints.erase(it3);
                    continue;  

                }
                it0++;
                it1++;
                it2++;
                it3++;
                des_index.push_back(pt_des);
            }
            mCurrentFrame.mDescriptors.create(des_index.size(), 32, CV_8U);
            cv::Mat temp;
            for(int i=0;i<des_index.size();i++) {
                temp = mLastFrame.mDescriptors.row(des_index[i]).clone();
                temp.copyTo(mCurrentFrame.mDescriptors.row(i));
            }
            
            mCurrentFrame.SetN((int)mCurrentFrame.mvKeys.size());
            mCurrentFrame.compute_descriptors();
            if(mCurrentFrame.mDescriptors.rows != mCurrentFrame.GetN()){
                cout<<"out of Order..... feature points and descriptors dont match..........."<<endl;
                return;      
            } 
        }
        
        //Inserting IMU data
        mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(),mCurrentFrame.mIMUdata.begin(),mCurrentFrame.mIMUdata.end());
        
        if(mState==NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState=mState;
        // there is a mutex in here originally
        bool bMapUpdated = false;
        if(needKeyFrame)
        {
            needKeyFrame=false;
            bMapUpdated = true;
        }
        
        if(mpLocalMapper->GetMapUpdateFlagForTracking())
        {
            bMapUpdated = true;
            boost::mutex::scoped_lock lock(mMutexThresds);// not sure this works or not
            mpLocalMapper->SetMapUpdateFlagInTracking(false);
        }
        if (mpLoopClosing->GetMapUpdateFlagForTracking())
        {
            bMapUpdated = true;
            mpLoopClosing->SetMapUpdateFlagInTracking(false);
        }
        if (mCurrentFrame.mnId == mnLastRelocFrameId + 20)
        {
            bMapUpdated = true;
        }       
        if(mState==NOT_INITIALIZED)
        {
            FirstInitialization(); // save the current frame and detected points

        }

        else if(mState==INITIALIZING)
        {
            Initialize();
            mLastFrame = FrameKTL(mCurrentFrame);
            if (mState == WORKING)
            {
                calculate_G();
            }
        }

        else if(mState==IMU_RELOCALIZATION)
        {
            //pause();
            cout<<"entering IMUpose recovery...."<<endl;
            PredictNavStateByIMU(false);
            RecoveryInitialization();
            if(mMode==VIP && mCurrentFrame.mHas_depth){
                cv::Mat tcw = mCurrentFrame.GetTranslation();
                //cout<<"disp 1 : "<<tcw.t()<<endl;
                cv::Mat Rcw = mCurrentFrame.GetRotation();
                cv::Mat Rwc = Rcw.t();
                cv::Mat twc = -Rwc*tcw;
                twc.at<float>(0,2)= mCurrentFrame.mDepth-ini_depth;// replacing depth measurment
                tcw = -Rcw*twc;
                //cout<<"disp 2 : "<<tcw.t()<<endl;
                tcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));
                mCurrentFrame.UpdatePoseMatrices();
            }
            
        }
        else if(mState==R_INITIALIZING){
            
            PredictNavStateByIMU(false);
            Recovery_Initialize(); 
            
            if(mState==WORKING){
                 mpLocalMapper->SetMapUpdateFlagInTracking(true);
                 //mbRelocBiasPrepare=true;
                 mnLastRelocFrameId = mCurrentFrame.mnId;
                 }
            else{
                Con++;
            }
            if (Con>15){
                mState=IMU_RELOCALIZATION;
                Con=0;
            }
        }
        else
        {
            bool bOK;
            if(mState==WORKING){

            
                // Initial Camera Pose Estimation from Previous Frame (Motion Model or Coarse) or Relocalisation
                //CheckReplacedInLastFrame();
                if(mpLocalMapper->GetVINSInited()){
                    //std::cout<<"Gravity Vector"<<mpLocalMapper->mGravityVec<<std::endl;
                    if(mbRelocBiasPrepare)
                    {
                        bOK = TrackWithPnP();
                    }
                    else
                    {
                        bOK = TrackWithIMU(bMapUpdated);
                        if(!bOK){
                            cout<<"Traking fails...................."<< endl;
                            bOK=true;                                                    
                        }
                                                            
                    }   

                }           
                else {     
                    bOK = TrackWithPnP(); 
                    if (!bOK && !mVelocity.empty())
                            bOK = TrackwithMotionModel();     
                }
            }
            else if(mState ==LOST)
            {   
                cout<<"State LOST call for Relocalization..................."<<endl;
                bOK = Relocalisation();
                if(bOK){
                    cout<<"Pose recoverd....."<<endl<<endl;
                    mState = WORKING;
                }
            }
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            //if(mCurrentFrame.mnId==1250){
                //cout<<"Forced Tracking loss.................................................."<<endl;
                //bOK=false;
            //}
                
            if(!bOK && mState==WORKING && mpLocalMapper->GetVINSInited())
            {
                bOK = IMU_Relocalisation();
                if (bOK){   
                    
                    mState==WORKING;
                    needKeyFrame=true;
                    cout<<"Frist try for relocalization Success....."<<endl;
                }
                else{ 
                cout<<"First try for relocalization fails...."<<endl;
                }
                
            }
            

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(bOK){
                
                if(!mpLocalMapper->GetVINSInited()){
                    bOK = TrackLocalMap();
                    //cout<<"LOCAL map normal...."<<endl;
                }
                else{
                    if(mbRelocBiasPrepare)
                    {
                        // 20 Frames after reloc, track with only vision
                        bOK = TrackLocalMap();
                        needKeyFrame=true;
                    }
                    else
                    {
                        bOK=TrackLocalMapWithIMU(bMapUpdated);
                        
                    }

                }
            }
                
           
            if(bOK)
            {
                mState = WORKING;
                //update motion model
                if(!mLastFrame.mTcw.empty())
                {
                    cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                    mVelocity = mCurrentFrame.mTcw*LastTwc;
                }
                else
                    mVelocity = cv::Mat();

                // Add Frames to re-compute IMU bias after reloc
                
                if(mbRelocBiasPrepare)
                {
                    mv20FramesReloc.push_back(mCurrentFrame);
                    cout<<"adding frame for recomputing for bias......."<<endl;
                    // Before creating new keyframe
                    // Use 20 consecutive frames to re-compute IMU bias
                    if(mCurrentFrame.mnId == mnLastRelocFrameId+10-1)// originally was set to 20
                    {
                        NavState nscur;
                        RecomputeIMUBiasAndCurrentNavstate(nscur);
                        // Update NavState of CurrentFrame
                        mCurrentFrame.SetNavState(nscur);
                        // Clear flag and Frame buffer
                        mbRelocBiasPrepare = false;
                        mv20FramesReloc.clear();

                        // Release LocalMapping. To ensure to insert new keyframe.
                        //mpLocalMapper->Release();
                        // Create new KeyFrame
                        mbCreateNewKFAfterReloc = true;

                        //Debug log
                        cout<<"NavState recomputed."<<endl;
                        cout<<"V:"<<mCurrentFrame.GetNavState().Get_V().transpose()<<endl;
                        cout<<"bg:"<<mCurrentFrame.GetNavState().Get_BiasGyr().transpose()<<endl;
                        cout<<"ba:"<<mCurrentFrame.GetNavState().Get_BiasAcc().transpose()<<endl;
                        cout<<"dbg:"<<mCurrentFrame.GetNavState().Get_dBias_Gyr().transpose()<<endl;
                        cout<<"dba:"<<mCurrentFrame.GetNavState().Get_dBias_Acc().transpose()<<endl;
                    }
                }
            }
            
            else if(!bOK)
            {  
                 
                mState=IMU_RELOCALIZATION;
                if(mMode==MONO){
                    mState=LOST;
                }
                // Clear Frame vectors for reloc bias computation
                if(mv20FramesReloc.size()>0)
                    mv20FramesReloc.clear();
            }
            

            if(bOK)
            {
                //mpMapPublisher->SetCurrentCameraPose(mCurrentFrame.mTcw);
                if (mCurrentFrame.mHas_depth){
                        mdepth_Vec.push_back(mCurrentFrame.mDepth);
                        mdepth_time_Vec.push_back(mCurrentFrame.mDepth_time);
                    }

                if(NeedNewKeyFrame()||mbCreateNewKFAfterReloc||needKeyFrame){
                    cout<<"cont number of frames with in last kyframe : "<<count<<endl;
                    count=0;
                    bool flag=false;
                    CreateNewKeyFrame(flag);
                    mvIMUSinceLastKF.clear();
                    mdepth_Vec.clear();
                    mdepth_time_Vec.clear();
                    if(needKeyFrame){
                        
                        //mpLocalMapper->InterruptBA();
                        //flag=true;
                        cout<<"............................Force Key Frame Creation.............................................."<<endl;
                    }
                    
                    
                }
                else{
                    count++;
                }

                if(mbCreateNewKFAfterReloc){
                mbCreateNewKFAfterReloc = false;
                }
                if(mpLocalMapper->GetFirstVINSInited())
                {
                    mpLocalMapper->SetFirstVINSInited(false);
                }
                
            }

            // Reset if the camera get lost soon after initialization
            /*
            if(mState==LOST)
            {   
                if(!mpLocalMapper->GetVINSInited())
                {
                    cout<<"State LOST call for Relocalization..................."<<endl;
                    Reset();
                    return;
                }
            }*/
                       
            if(!mCurrentFrame.mpReferenceKF){
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            }
                       
        }       

        // Update drawer
         mLastFrame = FrameKTL(mCurrentFrame);
         mpFramePublisher->Update(this);
         

        if(!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Rwc = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3).t();
            cv::Mat twc = -Rwc*mCurrentFrame.mTcw.rowRange(0,3).col(3);
            mpMap->addpose(twc);
            tf::Matrix3x3 M(Rwc.at<float>(0,0),Rwc.at<float>(0,1),Rwc.at<float>(0,2),
                            Rwc.at<float>(1,0),Rwc.at<float>(1,1),Rwc.at<float>(1,2),
                            Rwc.at<float>(2,0),Rwc.at<float>(2,1),Rwc.at<float>(2,2));
            tf::Vector3 V(twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));

            tf::Transform tfTcw(M,V);
            //mTfBr.sendTransform(tf::StampedTransform(tfTcw,mCurrentFrame.mTimeStamp,"USLAM/World","USLAM/Camera"));// turn on this for openvins
            mTfBr.sendTransform(tf::StampedTransform(tfTcw,ros::Time::now(),"USLAM/World","USLAM/Camera"));
            mpMapPublisher->SetCurrentCameraPose(mCurrentFrame.mTcw);
            
            
        }

    }

    bool Tracking::TrackwithMotionModel()
    {
        
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.GetN(); i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        return nmatchesMap >= 10;
    }

    void Tracking::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, cv::Mat &mDescriptors) {

        // Create a 2D occupancy grid for this current image
        // Note that we scale this down, so that each grid point is equal to a set of pixels
        // This means that we will reject points that less then grid_px_size points away then existing features

        int pt_num = pts0.size();
        
        Eigen::MatrixXi grid_2d = Eigen::MatrixXi::Zero((int)(img0pyr.at(0).rows/min_px_dist)+2, (int)(img0pyr.at(0).cols/min_px_dist)+2);
        //Picture size is 480x752
        int x,y;
        KeyPoint kpt;
        for(size_t i=0; i<pt_num; i++){
            kpt=pts0[i];
            x= (int)(kpt.pt.y/min_px_dist);
            y= (int)(kpt.pt.x/min_px_dist);
            grid_2d(x,y)++;
            /*
            if(mLastFrame.mvpMapPoints[i] && mState== WORKING)
            {
                grid_2d(x,y)++;
                if (grid_2d(x,y)>2){
                    MapPoint* pMP = mLastFrame.mvpMapPoints[i];   
                    mLastFrame.mvpMapPoints[i]=NULL;
                    mLastFrame.mvbOutlier[i]=false;
                    mvIniMatches[i]=-1;
                    pMP->mnLastFrameSeen = mLastFrame.mnId;
                    pMP->mbTrackInView = false;    
                }
            }
            else if(mState!= WORKING)
            {
                grid_2d(x,y)++;
            }**/

        }
        
        // First compute how many more features we need to extract from this image
        int num_featsneeded = num_features - (int)pts0.size();

        // If we don't need any features, just return
        if(num_featsneeded <= num_features*0.05)
            return;

        // Extract our features (use fast with griding)
        std::vector<cv::KeyPoint> pts0_ext;
        cv::Mat New_Descriptors;
        //Grider_FAST::perform_griding(img0pyr.at(0), pts0_ext, num_featsneeded, grid_x, grid_y, fastTh, true); // change here if you need HARRIS
        //Grider_HARRIS::perform_griding(img0pyr.at(0), pts0_ext, num_featsneeded, grid_x, grid_y, fastTh, true); // change here if you need HARRIS
        bool FullDetect=true;
        if(mState==WORKING)
            FullDetect=false;

        (*mpORBextractor)(img0pyr.at(0),cv::Mat(),pts0_ext,New_Descriptors, grid_2d,min_px_dist,FullDetect,num_featsneeded);
        
        if(pts0_ext.size()>0 && pt_num>0){
            cv::Mat des_temp = mDescriptors.clone(); 
            mDescriptors.create(pt_num + pts0_ext.size(), 32, CV_8U);
            cv::Mat aux_des1=mDescriptors.colRange(0,32).rowRange(0,pt_num);
            des_temp.copyTo(aux_des1);
            cv::Mat aux_des2=mDescriptors.colRange(0,32).rowRange(pt_num,pt_num+pts0_ext.size());
            New_Descriptors.copyTo(aux_des2);
            
            for(size_t i=0; i<pts0_ext.size(); i++) {
                pts0.push_back(pts0_ext.at(i));
            }
        } 
        if ((pts0_ext.size()>0 && pt_num==0)|| mState==LOST){
            mDescriptors=New_Descriptors;
            pts0=pts0_ext;
            
        } 

    }

    void Tracking::perform_matching(const std::vector<cv::Mat>& img0pyr, const std::vector<cv::Mat>& img1pyr,
                                    std::vector<cv::KeyPoint>& kpts0, std::vector<cv::KeyPoint>& kpts1, 
                                    std::vector<cv::KeyPoint>& kpts0_Un,std::vector<cv::KeyPoint>& kpts1_Un,
                                    std::vector<uchar>& mask_out) {
        
        // We must have equal vectors
        assert(kpts0.size() == kpts1.size());

        // Return if we don't have any points
        if(kpts0.empty() || kpts1.empty())
            return;

        // Convert keypoints into points (stupid opencv stuff)
        std::vector<cv::Point2f> pts0, pts1;
        for(size_t i=0; i<kpts0.size(); i++) {
            pts0.push_back(kpts0.at(i).pt);
            pts1.push_back(kpts1.at(i).pt);
        }
        // Project pror points
        if(!mVelocity.empty()&&!mpLocalMapper->GetFirstVINSInited())
        {
            mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
            const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
            const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);

            const cv::Mat twc = -Rcw.t()*tcw;

            const cv::Mat Rlw = mLastFrame.mTcw.rowRange(0,3).colRange(0,3);
            const cv::Mat tlw = mLastFrame.mTcw.rowRange(0,3).col(3);

            const cv::Mat tlc = Rlw*twc+tlw;

            for(int i=0; i<mLastFrame.GetN(); i++)
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];

                if(pMP) 
                {
                    if(!mLastFrame.mvbOutlier[i])
                    {
                        cv::Mat x3Dw = pMP->GetWorldPos();
                        cv::Mat x3Dc = Rcw*x3Dw+tcw;

                        const float xc = x3Dc.at<float>(0);
                        const float yc = x3Dc.at<float>(1);
                        const float invzc = 1.0/x3Dc.at<float>(2);

                        if(invzc<0)
                            continue;

                        float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
                        float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy; 
                        if(u<mCurrentFrame.mnMinX || u>mCurrentFrame.mnMaxX)
                            continue;
                        if(v<mCurrentFrame.mnMinY || v>mCurrentFrame.mnMaxY)
                            continue;
                        //cout<< pts1.at(i).x <<" Replaced by "<<u<<endl;
                        //cout<< pts1.at(i).y <<" Replaced by "<<v<<endl;
                        pts1.at(i).x = u;
                        pts1.at(i).y = v;
                    }

                }

            }
        }

        // If we don't have enough points for ransac just return empty
        // We set the mask to be all zeros since all points failed RANSAC
        if(pts0.size() < 10) {
            for(size_t i=0; i<pts0.size(); i++)
                mask_out.push_back((uchar)0);
            return;
        }

        // Now do KLT tracking to get the valid new points
        std::vector<uchar> mask_klt;
        std::vector<float> error;
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);// originally this was 25 and 0.01
        cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

        std::vector<cv::Point2f> pts0_Un, pts1_Un;
        for(size_t i=0; i<pts0.size(); i++) {
            pts0_Un.push_back(undistort_point(pts0.at(i)));
            pts1_Un.push_back(undistort_point(pts1.at(i)));
        }
        
        // maka undistrode feature points for the current frame
        // Normalize these points, so we can then do ransac
        // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
        
        
        // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
        std::vector<uchar> mask_rsc;
        cv::findFundamentalMat(pts0_Un, pts1_Un, cv::FM_RANSAC, 1, 0.999, mask_rsc);//originally 1 is used for harbor dataset
        // Loop through and record only ones that are valid
        for(size_t i=0; i<mask_klt.size(); i++) {      
            auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i])? 1 : 0);
            mask_out.push_back(mask);
        }

        // Copy back the updated positions
        for(size_t i=0; i<pts0.size(); i++) {
            kpts0.at(i).pt = pts0.at(i);
            kpts1.at(i).pt = pts1.at(i);
            kpts0_Un[i] = kpts0[i];
            kpts1_Un[i] = kpts1[i];
            kpts0_Un.at(i).pt = pts0_Un.at(i);
            kpts1_Un.at(i).pt = pts1_Un.at(i);
        }

        
    }
    bool Tracking::TrackWithIMU(bool bMapUpdated)
    
    {
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.GetN(); i++){
            if(mCurrentFrame.mvpMapPoints[i]){
                nmatchesMap++;
            }
        }
        int cont = nmatchesMap;

        if(!mpLocalMapper->GetVINSInited()) cerr<<"local mapping VINS not inited. why call TrackWithIMU?"<<endl;
        PredictNavStateByIMU(bMapUpdated);
        cv::Mat p2=mCurrentFrame.mTcw.col(3).t();
        //cout<<"Current pose BO "<<p2<<endl;
        // Pose optimization. false: no need to compute marginalized for current Frame
        if(mpLocalMapper->GetFirstVINSInited() || bMapUpdated)
        {
            nmatchesMap = Optimizer::PoseOptimization(&mCurrentFrame,mpLastKeyFrame,mIMUPreIntInTrack,mpLocalMapper->GetGravityVec(),
            mpLocalMapper->GetGravityRotation(),false,ini_depth,depth_cov);
            //cout<<"1 optimized with last KEYframe  ID : "<<mpLastKeyFrame->mnId<<endl;
        }
        else
        {
            nmatchesMap = Optimizer::PoseOptimization(&mCurrentFrame,&mLastFrame,mIMUPreIntInTrack,mpLocalMapper->GetGravityVec(),
            mpLocalMapper->GetGravityRotation(),false,ini_depth,depth_cov);
            
        }

        cv::Mat p3=mCurrentFrame.mTcw.col(3).t();
        //cout<<"Current pose AO "<<p3<<endl;
        //cout<<"Current pose Diff  "<<cv::norm(p3-p2)<<" from KF "<<bMapUpdated<<endl;
        //cout<<"mapPoint 0 "<<cont<<endl;
        //cout<<"mapPoint 1 "<<nmatchesMap<<endl;

        if(nmatchesMap<5){
            cout<<"Track with IMU fails, Inliers :"<<nmatchesMap<< " bMapUpdated ..? "<<bMapUpdated<<endl;  
            return false;
            }
        //cout<<"mapPoint 0 "<<cont<<endl;
        //cout<<"mapPoint 1 "<<nmatchesMap<<endl;

        // Discard outliers
        
        nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.GetN(); i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {   
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    mvIniMatches[i]=-1;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }
              
        //cout<<" mapPoint 2 "<<nmatchesMap<<endl;
        
        return true;//nmatchesMap>=20;
    }

    void Tracking::PredictNavStateByIMU(bool bMapUpdated)
    {
        
        // Map updated, optimize with last KeyFrame
        if(mpLocalMapper->GetFirstVINSInited() || bMapUpdated)
        {
            if(mpLocalMapper->GetFirstVINSInited() && !bMapUpdated) cerr<<"2-FirstVinsInit, but not bMapUpdated. shouldn't"<<endl;

            // Compute IMU Pre-integration
            mIMUPreIntInTrack = GetIMUPreIntSinceLastKF(&mCurrentFrame, mpLastKeyFrame, mvIMUSinceLastKF);

            // Get initial NavState&pose from Last KeyFrame
            mCurrentFrame.SetInitialNavStateAndBias(mpLastKeyFrame->GetNavState());
            mCurrentFrame.UpdateNavState(mIMUPreIntInTrack,Converter::toVector3d(mpLocalMapper->GetGravityVec()));
            
            //cout<<"pose pnp: "<<mCurrentFrame.mTcw.col(3).t()<<endl;
            mCurrentFrame.UpdatePoseFromNS(ConfigParam::GetMatTbc());
            ///cout<<"pose imu: "<<mCurrentFrame.mTcw.col(3).t()<<endl;


            // Test log
            // Updated KF by Local Mapping. Should be the same as mpLastKeyFrame
            if(mCurrentFrame.GetNavState().Get_dBias_Acc().norm() > 1e-6) cerr<<"PredictNavStateByIMU1 current Frame dBias acc not zero"<<endl;
            if(mCurrentFrame.GetNavState().Get_dBias_Gyr().norm() > 1e-6) cerr<<"PredictNavStateByIMU1 current Frame dBias gyr not zero"<<endl;
            //cout<<"IMU pose predicted from last KEYFrame ID : "<< mpLastKeyFrame->mnId<<endl;
        }
        // Map not updated, optimize with last Frame
        else
        {
            // Compute IMU Pre-integration
            mIMUPreIntInTrack = GetIMUPreIntSinceLastFrame(&mCurrentFrame, &mLastFrame);

            // Get initial pose from Last Frame
            mCurrentFrame.SetInitialNavStateAndBias(mLastFrame.GetNavState());
            mCurrentFrame.UpdateNavState(mIMUPreIntInTrack,Converter::toVector3d(mpLocalMapper->GetGravityVec()));
            
            //cout<<"pose pnp: "<<mCurrentFrame.mTcw.col(3).t()<<endl;
            mCurrentFrame.UpdatePoseFromNS(ConfigParam::GetMatTbc());
            //cout<<"pose imu: "<<mCurrentFrame.mTcw.col(3).t()<<endl;
            

            // Test log
            if(mCurrentFrame.GetNavState().Get_dBias_Acc().norm() > 1e-6) cerr<<"PredictNavStateByIMU2 current Frame dBias acc not zero"<<endl;
            if(mCurrentFrame.GetNavState().Get_dBias_Gyr().norm() > 1e-6) cerr<<"PredictNavStateByIMU2 current Frame dBias gyr not zero"<<endl;
            
        }
    }

    IMUPreintegrator Tracking::GetIMUPreIntSinceLastKF(FrameKTL* pCurF, KeyFrame* pLastKF, const std::vector<IMUData>& vIMUSInceLastKF)
    {
        // Reset pre-integrator first
        IMUPreintegrator IMUPreInt;
        IMUPreInt.reset();

        Vector3d bg = pLastKF->GetNavState().Get_BiasGyr();
        Vector3d ba = pLastKF->GetNavState().Get_BiasAcc();

        // remember to consider the gap between the last KF and the first IMU
        /*
        {
            const IMUData& imu = vIMUSInceLastKF.front();
            double dt = imu.timestamp - pLastKF->mTimeStamp;
            IMUPreInt.update(imu.wm - bg, imu.am - ba, dt);

            // Test log
            if(dt < 0)
            {
                cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this KF vs last imu time: "<<pLastKF->mTimeStamp<<" vs "<<imu.timestamp<<endl;
                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
            }
        }
        */
        // integrate each imu
        for(size_t i=0; i<vIMUSInceLastKF.size(); i++)
        {
            const IMUData& imu = vIMUSInceLastKF[i];
            double nextt;
            if(i==vIMUSInceLastKF.size()-1)
                nextt = pCurF->mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = vIMUSInceLastKF[i+1].timestamp;  // regular condition, next is imu data

            // delta time
            double dt = nextt - imu.timestamp;
            // update pre-integrator
            if (dt>0){
            IMUPreInt.update(imu.wm - bg, imu.am - ba, dt);
            }


            // Test log
            if(dt < 0)
            {
                cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time 1: "<<imu.timestamp<<" vs "<<nextt<<endl;
                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
            }
        }

        return IMUPreInt;
    }

    IMUPreintegrator Tracking::GetIMUPreIntSinceLastFrame(FrameKTL* pCurF, FrameKTL* pLastF)
    {
        // Reset pre-integrator first
        IMUPreintegrator IMUPreInt;
        IMUPreInt.reset();

        pCurF->ComputeIMUPreIntSinceLastFrame(pLastF,IMUPreInt);

        return IMUPreInt;
    }



    cv::Point2f Tracking::undistort_point(cv::Point2f pt_in){
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt_in.x;
        mat.at<float>(0, 1) = pt_in.y;
        mat = mat.reshape(2); // Nx1, 2-channel
        // Undistort it!
        if(Fisheye_Cam){
            cv::fisheye::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);// this needed for aqualoc harbor dataset
        }
        else{
            cv::undistortPoints(mat, mat, mK, mDistCoef,cv::Mat(),mK); // last two parameters for not obtain normalized coordinates
        }
        // Construct our return vector
        cv::Point2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        return pt_out;
    }


    void Tracking::FirstInitialization()
    {
        //We ensure a minimum ORB features to continue, otherwise discard frame
        mvIMUSinceLastKF.clear();
        if(mMode == VIP && !mCurrentFrame.mHas_depth){
            cout<<"No depth measurments..."<<endl;
            return;
        }
        if(mCurrentFrame.GetN()>num_features*0.8 )
        {
            mInitialFrame = FrameKTL(mCurrentFrame);
            mLastFrame = FrameKTL(mCurrentFrame);
            
            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);// this curret frame will be the reference frame


            mState = INITIALIZING;
            ini_depth = mCurrentFrame.mDepth;// CCHANGE TO 0.0 HERE FOR VIP SETUP ...Why ....?   CHHINTHAKA
            if (mMode == VIP){
                //ini_depth =0.0;
            }
            cout<<"First initialization..........................."<<endl; 
            
        }
    }


    void Tracking::Initialize()

    {   
        // Check if current frame has enough keypoints, otherwise reset initialization process
        //std::cout<<"detected point size is   ="<<mCurrentFrame.mvKeys.size()<<std::endl;
        int nmatches = mCurrentFrame.GetN();
        if(nmatches<=num_features*0.9)
        {
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);// mvIniMatches is a int vector which store the index of the match  feature, -1 if no match found
            mState = NOT_INITIALIZED;
            mvIMUSinceLastKF.clear();
            cout<<"initialized failed, Low in Propagated Points"<<endl; 
            return;
        }
        if(mMode == VIP && !mCurrentFrame.mHas_depth){
            cout<<"initialized failed"<<endl; 
            return;// CHANGE HERE FOR VIP SETUP
        }    

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))// mvIni and vBTri gives according to th einitial frame(last frame)
        {   
            mState=WORKING;
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            
            }
            std::vector<int> des_index;
            int pt_des=-1;
            auto it0 = mvIniMatches.begin();
            auto it1 = mCurrentFrame.mvKeys.begin();
            auto it2 = mCurrentFrame.mvKeysUn.begin();
            auto it3 = mvIniP3D.begin();
            auto it4 = mCurrentFrame.mvpMapPoints.begin();
            while(it0 != mvIniMatches.end()) {
                int K = *it0;
                pt_des++;
                
                if(K<0) {
                    it0 = mvIniMatches.erase(it0);
                    it1 = mCurrentFrame.mvKeys.erase(it1);
                    it2 = mCurrentFrame.mvKeysUn.erase(it2);
                    it3 = mvIniP3D.erase(it3);
                    it4 = mCurrentFrame.mvpMapPoints.erase(it4);
                    continue;  

                }
                it0++;
                it1++;
                it2++;
                it3++;
                it4++;
                des_index.push_back(pt_des);
            }
            cv::Mat temp_descriptors = mCurrentFrame.mDescriptors.clone();
            mCurrentFrame.mDescriptors.release();
            //cv::resize(temp_descriptors,mCurrentFrame.mDescriptors,cv::Size(32, des_index.size()),0,0, CV_8U);
            mCurrentFrame.mDescriptors.create(des_index.size(), 32, CV_8U);
            cv::Mat temp;
            for(int i=0;i<des_index.size();i++) {
                temp = temp_descriptors.row(des_index[i]).clone();
                temp.copyTo(mCurrentFrame.mDescriptors.row(i));
            }
            
            int N=mCurrentFrame.GetN();
            //mCurrentFrame.mvbOutlier = vector<bool>(N,false);
            mCurrentFrame.mvKeysUn.resize(N);
            mvIniMatches.resize(N,-1);
            mCurrentFrame.mvpMapPoints.resize(N,static_cast<MapPoint*>(NULL));
            
            CreateInitialMap(Rcw,tcw,N); // save the last frame

            //detecting new Points
            
            for (size_t i=0; i< mvIniMatches.size(); i++){
                mvIniMatches[i]=i;
            }
            cout<<"initialized successs.. #points : "<<nmatches<<endl;
        }
        else{
            //mvIniMatches.resize(mLastFrame.mvKeys.size(),-1);// this is needed
            //for (size_t i=0; i< mvIniMatches.size(); i++){
               // mvIniMatches[i]=i;
            //}
            //cout<<"initialized failed, cant tiangulate enough points"<<endl; 
        }  

    }

     
    void Tracking::CreateInitialMap(cv::Mat &Rcw, cv::Mat &tcw, int &Lim_KP)
    {
        // Set Frame Poses
        mInitialFrame.mTcw = cv::Mat::eye(4,4,CV_32F);
        mCurrentFrame.mTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));
        
        //stopped here

        // Create KeyFrames
        KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB,mInitialFrame.mIMUdata,NULL);
        pKFini->ComputePreInt();
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,mvIMUSinceLastKF,pKFini);
        pKFcur->ComputePreInt();
        mvIMUSinceLastKF.clear();
        pKFcur->mvIniMatches=mvIniMatches;

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();
        pKFcur->ComputeHaloc();
        pKFini->ComputeHaloc();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        pKFcur->IndexforLastKfKp= std::vector<int> (mvIniMatches.begin(),mvIniMatches.begin()+Lim_KP);

        // Create MapPoints and asscoiate to keyframes
        for(size_t i=0; i<mvIniMatches.size();i++)
        {
            if(mvIniMatches[i]<0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

            pKFini->AddMapPoint(pMP,mvIniMatches[i]);
            pKFcur->AddMapPoint(pMP,i);

            pMP->AddObservation(pKFini,mvIniMatches[i]);
            pMP->AddObservation(pKFcur,i);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[i] = pMP; // 3d point(map point storing)
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
            //mLastFrame.mvpMapPoints[mvIniLastMatches[i]] = pMP;// dont need to do this
            //Add to Map
            mpMap->AddMapPoint(pMP);

        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        ROS_INFO("New Map created with %d points",mpMap->MapPointsInMap());

        Optimizer::GlobalBundleAdjustemnt(mpMap,20);


        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f/medianDepth;

        if(medianDepth<0 || pKFcur->TrackedMapPoints(2)<100)
        {
            ROS_INFO("Wrong initialization, reseting...");
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;// 2 was not here originally
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
        {
            if(vpAllMapPoints[iMP])
            {
                MapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);// 2 is not here originally;
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mCurrentFrame.mTcw = pKFcur->GetPose().clone();
        mCurrentFrame.UpdatePoseMatrices();
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        mpLocalMapper->mpLastKeyFrame = pKFcur;
        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapPublisher->SetCurrentCameraPose(pKFcur->GetPose());

       
    }
    
    void Tracking::RecoveryInitialization()
    {
        
        mvIMUSinceRecovery= mvIMUSinceLastKF;
        //mvIMUSinceLastKF.clear();
        cout<<"First detected points : "<<mCurrentFrame.GetN()<<endl;
        if(mMode==VIP && !mCurrentFrame.mHas_depth)
        {
           mState = IMU_RELOCALIZATION;
            cout<<"No depth measurments  Check the Mode of operation V-Only ? VI ? VIP ? "<<endl; 
            return;
        }
        if(mCurrentFrame.GetN()>num_features*0.9)
        {
            mRecoveryFrame = FrameKTL(mCurrentFrame);
            mRecoveryFrame.UpdatePoseMatrices();
            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);// this curret frame will be the reference frame

            mState = R_INITIALIZING;
            //cout<<endl<<"Id : "<<mCurrentFrame.mnId<<" Pose 0 is :" <<mCurrentFrame.GetTranslation().t()<<endl;
            
        }
        else{
            mState = IMU_RELOCALIZATION;
            cout<<"no enough points detected..."<<endl;
        }
    }

    void Tracking::Recovery_Initialize()

    {   
        
        int nmatches = mCurrentFrame.GetN();
        cout<<"Second detected point : "<<nmatches<<endl;
        if(nmatches<=num_features*0.6)
        {
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);// mvIniMatches is a int vector which store the index of the match  feature, -1 if no match found
            mState = IMU_RELOCALIZATION;
            mvIMUSinceLastKF.clear();
            cout<<"no enough points Propagated..."<<endl;
            return;
        }
        if(mMode==VIP && !mCurrentFrame.mHas_depth){//CHANNGE HERE FOR VIP SETUP
            cout<<"Recovery initialization failed"<<endl; 
            return;
        }
        mvIniP3D.clear(); 
        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        bool Tri_success = mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated);  
        if(Tri_success){
            cout<<"original triagulated success....................................................................... "<<endl;
        }
        //vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        
        //float parallax;
        //cout<<endl<<"Id : "<<mCurrentFrame.mnId<<" Pose 1 is :" <<mCurrentFrame.GetTranslation().t()<<endl;
        //const float MedianDepth = mpLastKeyFrame->ComputeSceneMedianDepth(2);
        //int nGood= mpInitializer->Re_CheckRT(mRecoveryFrame,mCurrentFrame,mvIniMatches,mK,mvIniP3D,8.0,vbTriangulated,parallax, MedianDepth);
        //cout<<"Good Points :"<<nGood<<endl;
        //cout<<"parallex :"<<parallax<<endl;
        //if(nGood>100)// && parallax>1)// mvIni and vBTri gives according to th einitial frame(last frame)
        if(Tri_success)
        {   
            
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            
            }
            std::vector<int> des_index;
            int pt_des=-1;
            auto it0 = mvIniMatches.begin();
            auto it1 = mCurrentFrame.mvKeys.begin();
            auto it2 = mCurrentFrame.mvKeysUn.begin();
            auto it3 = mvIniP3D.begin();
            auto it4 = mCurrentFrame.mvpMapPoints.begin();
            while(it0 != mvIniMatches.end()) {
                int K = *it0;
                pt_des++;
                
                if(K<0) {
                    it0 = mvIniMatches.erase(it0);
                    it1 = mCurrentFrame.mvKeys.erase(it1);
                    it2 = mCurrentFrame.mvKeysUn.erase(it2);
                    it3 = mvIniP3D.erase(it3);
                    it4 = mCurrentFrame.mvpMapPoints.erase(it4);
                    continue;  

                }
                it0++;
                it1++;
                it2++;
                it3++;
                it4++;
                des_index.push_back(pt_des);
            }
            cv::Mat temp_descriptors = mCurrentFrame.mDescriptors.clone();
            mCurrentFrame.mDescriptors.release();
            //cv::resize(temp_descriptors,mCurrentFrame.mDescriptors,cv::Size(32, des_index.size()),0,0, CV_8U);
            mCurrentFrame.mDescriptors.create(des_index.size(), 32, CV_8U);
            cv::Mat temp;
            for(int i=0;i<des_index.size();i++) {
                temp = temp_descriptors.row(des_index[i]).clone();
                temp.copyTo(mCurrentFrame.mDescriptors.row(i));
            }
            
            int N=mCurrentFrame.GetN();
            //mCurrentFrame.mvbOutlier = vector<bool>(N,false);
            mCurrentFrame.mvKeysUn.resize(N);
            mvIniMatches.resize(N,-1);
            mCurrentFrame.mvpMapPoints.resize(N,static_cast<MapPoint*>(NULL));
            bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
            if(! bLocalMappingIdle){
                mpLocalMapper->InterruptBA();
            }
            mpLocalMapper->clearLocalKF();
            mState=WORKING;
            CreateRecoveryMap(Rcw,tcw,N); // save the last frame
            
            for (size_t i=0; i< mvIniMatches.size(); i++){
                mvIniMatches[i]=i;
            }
            cout<<"Relocalization with IMU Poses successs.. #points : "<<nmatches<<endl;
            //cout<<endl<<"Id : "<<mCurrentFrame.mnId<<" Pose 2 is :" <<mCurrentFrame.GetTranslation().t()<<endl;
        }
        else{
            //mvIniMatches.resize(mLastFrame.mvKeys.size(),-1);// this is needed
            //for (size_t i=0; i< mvIniMatches.size(); i++){
               // mvIniMatches[i]=i;
            //}
            cout<<"IMU initialized failed........."<<endl; 
        }  

    }

    void Tracking::CreateRecoveryMap(cv::Mat &Rcw, cv::Mat &tcw,int &Lim_KP)
    {
        cv::Mat Tbc = ConfigParam::GetMatTbc();
        cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3);
        cv::Mat pbc = Tbc.rowRange(0,3).col(3);
        cv::Mat Rcb = Rbc.t();
        cv::Mat pcb = -Rcb*pbc;

        vector<float> vDepths;
        for(size_t i=0; i<mvIniMatches.size();i++){
            if(mvIniMatches[i]<0)
                continue;

            //Create MapPoint.
            cv::Point3f pt3D= mvIniP3D[i];
            vDepths.push_back(pt3D.z);
        }
        sort(vDepths.begin(),vDepths.end());
        float d2 = vDepths[(vDepths.size()-1)/2];
            
        // Create KeyFrames
        Eigen::Vector3d dis=mRecoveryFrame.mNavState.Get_P() - mCurrentFrame.mNavState.Get_P();
        cv::Mat dis1 = Converter::toCvMat(dis);
        float d1 =cv::norm(dis1)/cv::norm(tcw);
        //float d1 =mpLastKeyFrame->ComputeSceneMedianDepth()/d2;
        //d1=max(d0,d1);
        
        //cout<<"rations  "<<d0<<"   "<<d1<<endl;
        cv::Mat Rcw0 = mRecoveryFrame.GetRotation();
        cv::Mat tcw0 = mRecoveryFrame.GetTranslation();
        cv::Mat Rcw1 = Rcw*Rcw0;
        cv::Mat tcw1 = tcw*d1 + Rcw*tcw0;  
        cv::Mat twc0 = -Rcw0.t()*tcw0;
        

        Rcw1.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
        tcw1.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));
        mCurrentFrame.UpdatePoseMatrices();
        cout<<"distance : "<<d1<<endl;
        cout<<"first pose : "<<(-Rcw0.t()*tcw0).t()<<" distance : "<<cv::norm(tcw0)<<endl;
        cout<<"Second pose : "<<(-Rcw1.t()*tcw1).t()<<" distance : "<<cv::norm(tcw1)<<endl;


        KeyFrame* pKFini = new KeyFrame(mRecoveryFrame,mpMap,mpKeyFrameDB,mvIMUSinceRecovery,mpLastKeyFrame);
        pKFini->SetInitialNavStateAndBias(mRecoveryFrame.GetNavState());
        pKFini->ComputePreInt();
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,mvIMUSinceLastKF,pKFini);
        pKFcur->SetInitialNavStateAndBias(mCurrentFrame.GetNavState());
        cv::Mat wPc1 = pKFcur->GetPoseInverse().rowRange(0,3).col(3);                   // wPc
        cv::Mat Rwc1 = pKFcur->GetPoseInverse().rowRange(0,3).colRange(0,3);            // Rwc
            // Set position and rotation of navstate
        cv::Mat wPb1 = wPc1 + Rwc1*pcb;
        pKFcur->SetNavStatePos(Converter::toVector3d(wPb1));
        pKFcur->SetNavStateRot(Converter::toMatrix3d(Rwc1*Rcb));

        cv::Mat wPc0 = pKFini->GetPoseInverse().rowRange(0,3).col(3);                   // wPc
        cv::Mat Rwc0 = pKFini->GetPoseInverse().rowRange(0,3).colRange(0,3);            // Rwc
            // Set position and rotation of navstate
        cv::Mat wPb0 = wPc0 + Rwc0*pcb;
        pKFini->SetNavStatePos(Converter::toVector3d(wPb0));
        pKFini->SetNavStateRot(Converter::toMatrix3d(Rwc0*Rcb));

        // compute velocity
        cv::Mat gw = mpLocalMapper->GetGravityVec(); 
        Vector3d gweig = Converter::toVector3d(gw);                                     // deltaTime
        KeyFrame* pKFprev = pKFcur->GetPrevKeyFrame();
        const IMUPreintegrator& imupreint_prev_cur = pKFcur->GetIMUPreInt();
        double dt = imupreint_prev_cur.getDeltaTime();
        Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
        Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
        //
        Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
        Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
        Eigen::Vector3d dbiasa_eig = pKFini->mNavState.Get_BiasAcc();
        Eigen::Vector3d veleig = velpre + gweig*dt + rotpre*( dv + Jvba*dbiasa_eig );
        pKFcur->SetNavStateVel(veleig);

        pKFcur->ComputePreInt();
        mvIMUSinceLastKF.clear();// you have rmoved this ...? why...?
        pKFcur->mvIniMatches=mvIniMatches;

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();
        pKFcur->ComputeHaloc();
        pKFini->ComputeHaloc();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);
        mpLocalMapper->fixedKF_id=pKFini->mnId;// setting  for the fixed keyframe in bundle adjustment

        pKFcur->IndexforLastKfKp= std::vector<int> (mvIniMatches.begin(),mvIniMatches.begin()+Lim_KP);
        mvpLocalMapPoints.clear();
        mvpLocalKeyFrames.clear();

        // Create MapPoints and asscoiate to keyframes
        for(size_t i=0; i<mvIniMatches.size();i++)
        {
            if(mvIniMatches[i]<0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);
            worldPos = twc0+ (Rcw0.t()*worldPos*d1);

            MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

            pKFini->AddMapPoint(pMP,mvIniMatches[i]);
            pKFcur->AddMapPoint(pMP,i);

            pMP->AddObservation(pKFini,mvIniMatches[i]);
            pMP->AddObservation(pKFcur,i);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[i] = pMP; // 3d point(map point storing)
            mCurrentFrame.mvbOutlier[i] = false;
            mLastFrame.mvpMapPoints[mvIniMatches[i]] = pMP;// dont need to do this
            //Add to Map
            mpMap->AddMapPoint(pMP);
            mvpLocalMapPoints.push_back(pMP);

        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        ROS_INFO("Relocalized with %d new points",mvIniMatches.size());

        //Optimizer::GlobalBundleAdjustemnt(mpMap,20);
        Optimizer::RecoveryBundleAdjustemnt(pKFini,pKFcur,20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f/medianDepth;

        if(medianDepth<0 || pKFcur->TrackedMapPoints(2)<50)
        {   cout<<"Lost in createing recovery map...."<<endl;
            mState=LOST;
            return;
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mCurrentFrame.mTcw = pKFcur->GetPose().clone();
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        //mpLocalMapper->mpLastKeyFrame = pKFcur;
        //mvpLocalKeyFrames.clear();
        //mvpLocalMapPoints.clear();
        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        //mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMapPublisher->SetCurrentCameraPose(pKFcur->GetPose());
       
    }


    bool Tracking::TrackWithPnP()
    {   
       
        int N = mCurrentFrame.GetN();
        if(N<10){
            cout<<"PNP Ransac fails...."<< endl;
            return false;
        }
        std::vector<cv::KeyPoint> kpts = mCurrentFrame.mvKeys;
        std::vector<cv::Point2f> pts;
        std::vector<cv::Point3f> mappts;
               
        for(int i=0; i<N; i++) {
            if(mCurrentFrame.mvpMapPoints[i]){
                pts.push_back(kpts.at(i).pt);
                cv::Point3f Wpts(mCurrentFrame.mvpMapPoints[i]->GetWorldPos());
                mappts.push_back(Wpts);
            }
            
        } 
             
        cv::Mat Rvec; // Rotation in axis-angle form
        cv::Mat Tvec;
        cv::Mat Rmat;
        vector<int> mask_pnp;
        bool pnp_OK;
        if (mappts.size()>4){
         pnp_OK = cv::solvePnPRansac(mappts,pts,mK,mDistCoef,Rvec,Tvec,false,300,3,0.99,mask_pnp,cv::SOLVEPNP_EPNP); // mask_pnp gives indices of  inliers, use this to fill outliers and remove the optomozation function later
        }
        else{pnp_OK=false;}
        //cout<<"mask"<<mask_pnp.size()<<endl;
        
        if (!pnp_OK){
            cout<<"PNP Ransac fails...."<< endl;
            return false;
        }
                
        cv::Rodrigues(Rvec,Rmat);

        mCurrentFrame.mTcw = cv::Mat::eye(4,4,CV_32F);
        Rmat.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
        Tvec.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));
        
        //mCurrentFrame.mTcw = mVelocity*mLastFrame.mTcw;

        int nmatches = 0;
        // Optimize pose with all correspondences
        Optimizer::PoseOptimization(&mCurrentFrame);
        // Discard outliers
        for(size_t i =0; i<mCurrentFrame.mvpMapPoints.size(); i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                    mCurrentFrame.mvbOutlier[i]=false;
                    mvIniMatches[i]=-1;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
                else
                nmatches++;
            }
        }

        //cout<<"number of matches..  : "<<nmatches<<endl;
        if (nmatches<20){
        cout<<"maches: "<<nmatches<<endl;
        cout<<"PNP track fails: "<<endl;
        }
        return nmatches>=20;
    }

    bool Tracking::TrackLocalMap()
    {
        // Tracking from previous frame or relocalisation was succesfull and we have an estimation
        // of the camera pose and some map points tracked in the frame.
        // Update Local Map and Track
       
        // Update Local Map
        UpdateReference();// set a local map of keyframes and map points
        
        SearchReferencePointsInFrustum();// search and add map point to mvpMappoint from the local mappoints made in upper step

        // Optimize Pose
        mnMatchesInliers = Optimizer::PoseOptimization(&mCurrentFrame);// you need this because you add more points to this
        // Update MapPoints Statistics
        for(size_t i=0; i<mCurrentFrame.mvpMapPoints.size(); i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(!mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                else{
                    mvIniMatches[i]=-1;
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<10)
            return false;

        cout<<"mapPoint Vonly....................... "<<mnMatchesInliers<<endl;
        if(mnMatchesInliers<10)//Originally this was 20
        {
            cout<<"low inliers in local map, relocalizing # points : "<<mnMatchesInliers<< endl;
            
            return false;
        }
        else
            return true;
    }

    bool Tracking::TrackLocalMapWithIMU(bool bMapUpdated)
    {
        // Tracking from previous frame or relocalisation was succesfull and we have an estimation
        // of the camera pose and some map points tracked in the frame.
        // Update Local Map and Track
        
        
        // Update Local Map
        UpdateReference();// set a local map of keyframes and map points
        // set up a local map of keyframes(mvpLocalKeyframes),setup local map points(mvpLocalMappoints)
        //find the maximum map point observer(keyframe and save as mReferenceKF)
        //pass local map points in the map for display
        int count0=0;
        for(int i =0; i<mCurrentFrame.GetN(); i++){
            if(mCurrentFrame.mvpMapPoints[i]){
                count0++;
            }
        }
        //cout<<"mappoint in Local mapping......"<<count0<<endl;
        // Search Local MapPoints
        SearchReferencePointsInFrustum();// search and add map point to mvpMappoint from the local mappoints made in upper step
        
        int count1=0;
        for(int i =0; i<mCurrentFrame.GetN(); i++){
            if(mCurrentFrame.mvpMapPoints[i]){
                count1++;
            }
        }

        // Optimize Pose
        if(mpLocalMapper->GetFirstVINSInited() || bMapUpdated)
            {
                // Get initial pose from Last KeyFrame
                IMUPreintegrator imupreint = GetIMUPreIntSinceLastKF(&mCurrentFrame, mpLastKeyFrame, mvIMUSinceLastKF);

                // Test log
                if(mpLocalMapper->GetFirstVINSInited() && !bMapUpdated) cerr<<"1-FirstVinsInit, but not bMapUpdated. shouldn't"<<endl;
                if(mCurrentFrame.GetNavState().Get_dBias_Acc().norm() > 1e-6) cerr<<"TrackLocalMapWithIMU current Frame dBias acc not zero"<<endl;
                if(mCurrentFrame.GetNavState().Get_dBias_Gyr().norm() > 1e-6) cerr<<"TrackLocalMapWithIMU current Frame dBias gyr not zero"<<endl;

                //
                mnMatchesInliers=Optimizer::PoseOptimization(&mCurrentFrame,mpLastKeyFrame,imupreint,mpLocalMapper->GetGravityVec(),
                mpLocalMapper->GetGravityRotation(),true,ini_depth,depth_cov);
                //cout<<"2 Optimization with KEYFrame ID : "<<mpLastKeyFrame->mnId<<endl;
            }
        // Map not updated, optimize with last Frame
        else
            {
                // Get initial pose from Last Frame
                IMUPreintegrator imupreint = GetIMUPreIntSinceLastFrame(&mCurrentFrame, &mLastFrame);

                mnMatchesInliers=Optimizer::PoseOptimization(&mCurrentFrame,&mLastFrame,imupreint,mpLocalMapper->GetGravityVec(),
                mpLocalMapper->GetGravityRotation(),true,ini_depth,depth_cov);
                
            }
        // Update MapPoints Statistics
        
        for(size_t i=0; i<mCurrentFrame.mvpMapPoints.size(); i++){

            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(!mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                else{
                    //mCurrentFrame.mvpMapPoints[i]->mbTrackInView = false;
                    //mCurrentFrame.mvpMapPoints[i]->mnLastFrameSeen = mCurrentFrame.mnId;
                    //mvIniMatches[i]=-1;
                    //mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    //mCurrentFrame.mvbOutlier[i]=false;
                   

                }
            }
        }
        

        int count2=0;
        for(int i =0; i<mCurrentFrame.GetN(); i++){
            if(mCurrentFrame.mvpMapPoints[i]){
                count2++;
            }
        }

        //cout<<"mapPoint TLM 0....................... "<<count0<<endl;
        //cout<<"mapPoint TLM 1....................... "<<count1<<endl;
        cout<<"mapPoint TLM 2....................... "<<mnMatchesInliers<<endl;

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<2){
           
            //needKeyFrame=true;
            return false;
        }

        if(mnMatchesInliers<10){
            cout<<"fails in track Local map, # points : "<<mnMatchesInliers<< endl;
            count_for_relocalize++;
            needKeyFrame=true;
        }
        else if(mnMatchesInliers<2)//Originally this was 20
        {
            cout<<"fails in track Local map, # points : "<<mnMatchesInliers<< endl;
            needKeyFrame=true;
            count_for_relocalize=0;
        }
        else
        {count_for_relocalize=0;}
        
        if(count_for_relocalize>30)
        {
            return false;
        }
        else
        {   
            
            return true;
        }
    }


    bool Tracking::NeedNewKeyFrame()
    {
         // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
        
        if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()){
            return false;
        }
        
        /*
        if(mMode==VIP){
            if(!mCurrentFrame.mHas_depth && mpLocalMapper->GetVINSInited()){
                return false;//CHANGE HERE FOR VIP SETUP
            }
        }
        */

        const int nKFs = mpMap->KeyFramesInMap();
        // Not insert keyframes if not enough frames from last relocalisation have passed
        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames){
            return false;
        }

        if(mbRelocBiasPrepare)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        int nTracked= 0;
        for(int i =0; i<mCurrentFrame.mvKeys.size(); i++)
        {   
            if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTracked++;   
        }

        int nMinObs = 3;
        if (nKFs <= 3)
            {nMinObs = 2;}
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
        
        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1 = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c2 = mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle;
        //Condition 1c: tracking is weak
        const bool c3 =  nTracked < 150;///nRefMatches * 0.25;
        // Condition 2: Less than 90% of points than reference keyframe and enough inliers
        const bool c4 = mnMatchesInliers<nRefMatches*0.9 && mnMatchesInliers>15;
        
        if(c1||(c2&&c4)||c3)/// try to implement a parallex function CHINTHAKA
        {
            // If the mapping accepts keyframes insert, otherwise send a signal to interrupt BA, but no
            
            if(bLocalMappingIdle)
            {   
                return true;
            }
            else
            {                 
                mpLocalMapper->InterruptBA();
                if (mMode==MONO)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
        }
        else
            return false;
    }

    void Tracking::CreateNewKeyFrame(bool flag)
    {
        KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,mvIMUSinceLastKF,mpLastKeyFrame);
        pKF->RL_flag=flag;
        pKF->SetInitialNavStateAndBias(mCurrentFrame.GetNavState());
        pKF->ComputePreInt();
        pKF->mvIniMatches = mvIniMatches;
        pKF->Set_Depth(mdepth_Vec,mdepth_time_Vec);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;
 
        pKF->IndexforLastKfKp = mvIniMatches;
        mLast20KF.push_back(pKF);
        if(mLast20KF.size()>30){
            mLast20KF.erase(mLast20KF.begin());
        }
        mpLocalMapper->InsertKeyFrame(pKF);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
        for (size_t i=0; i< mvIniMatches.size(); i++){
                mvIniMatches[i]=i;
        }
        cout<<"Key Frame created...ID: "<<pKF->mnId<<endl;
        

    }

    void Tracking::SearchReferencePointsInFrustum()
    {
        // Do not search map points already matched
        for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(pMP->isBad())
                {
                    *vit = NULL;// step for remove bad points
                }
                else
                {
                    pMP->IncreaseVisible();// increase a counter
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        mCurrentFrame.UpdatePoseMatrices();// updata R and t for the frame

        int nToMatch=0;
        int nmatched=0;

        // Project points in frame and check its visibility
        for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if(pMP->isBad())
                continue;        
            // Project (this fills MapPoint variables for matching)
            if(mCurrentFrame.isInFrustum(pMP,0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }
        }    
        //cout<<"To match points : "<<nToMatch<<endl;

        if(nToMatch>0)
        {
            ORBmatcher matcher(0.8);//originally this was 0.8
            int th = 1;//originally this is 1
            // If the camera has been relocalised recently, perform a coarser search
            if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
                th=5;
                
            nmatched=matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        }
        //cout<<"Matched points : "<<nmatched<<endl;
    }

    void Tracking::UpdateReference()
    {    
        // This is for visualization
       
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);// pass all map points in the map for display

        // Update
        UpdateReferenceKeyFrames();// set up a local map of keyframes, find the maximum map point observer(keyframe and save as mReferenceKF)
        UpdateReferencePoints(); // setup local mappoints and update its reference to current frame ID
    }

    void Tracking::UpdateReferencePoints()
    {
        mvpLocalMapPoints.clear();

        for(vector<KeyFrame*>::iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
        {
            KeyFrame* pKF = *itKF;
            vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

            for(vector<MapPoint*>::iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
            {
                MapPoint* pMP = *itMP;
                if(!pMP)
                    continue;
                if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                    continue;
                if(!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId; // this is updating this parameter on local map points "mnTrackReferenceForFram" 
                }
            }
        }
    }


    void Tracking::UpdateReferenceKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        float limit = mpLocalMapper->fixedKF_id;// this is used in recovery map creation
        map<KeyFrame*,int> keyframeCounter;
        
        for(size_t i=0, iend=mCurrentFrame.mvpMapPoints.size(); i<iend;i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP->isBad())
                {
                    map<KeyFrame*,size_t> observations = pMP->GetObservations(); // this gives keyframe pointer and the index of the point in that keyframe
                    for(map<KeyFrame*,size_t>::iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        if(it->first->mnId>=limit){
                            keyframeCounter[it->first]++;
                        }
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }

        
        int max=0;
        KeyFrame* pKFmax=NULL;

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for(map<KeyFrame*,int>::iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
        {
            KeyFrame* pKF = it->first;

            if(pKF->isBad())
                continue;

            if(it->second>max)
            {
                max=it->second;
                pKFmax=pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for(vector<KeyFrame*>::iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if(mvpLocalKeyFrames.size()>80)
                break;

            KeyFrame* pKF = *itKF;

            vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for(vector<KeyFrame*>::iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
            {
                KeyFrame* pNeighKF = *itNeighKF;
                if(!pNeighKF->isBad())
                {
                    if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;// this is updating this parameter on local key frame "mnTrackReferenceForFram" 
                        break;
                    }
                }
            }

        }

        if(mvpLocalKeyFrames.size()<30 && mMode!=MONO){
            mvpLocalKeyFrames.clear();
            mvpLocalKeyFrames.reserve(30);
           for(vector<KeyFrame*>::iterator itNeighKF=mLast20KF.begin(), itEndNeighKF=mLast20KF.end(); itNeighKF!=itEndNeighKF; itNeighKF++){
               KeyFrame* pNeighKF = *itNeighKF;
                if(!pNeighKF->isBad())
                { 
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;// this is updating this parameter on local key frame "mnTrackReferenceForFrame"  
                }

           } 
        }
        if(pKFmax){
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
        else{
            
            int pp = mvpLocalMapPoints.size();
            cout<<"pKFmax is empty  but mvplocalmappoints are :"<<pp<<endl;
            }
    }

    bool Tracking::Relocalisation()
    {
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        vector<KeyFrame*> vpCandidateKFs;
        vpCandidateKFs= mpKeyFrameDB->DetectRelocalisationCandidates(&mCurrentFrame);      
        if(vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75,true);

        vector<PnPsolver*> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint*> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates=0;

        for(size_t i=0; i<vpCandidateKFs.size(); i++)
        {
            KeyFrame* pKF = vpCandidateKFs[i];
            if(pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
                if(nmatches<15)
                {
                    vbDiscarded[i] = true;
                    continue;
                }
                else
                {
                    PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }        
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9,true);

        while(nCandidates>0 && !bMatch)
        {
            for(size_t i=0; i<vpCandidateKFs.size(); i++)
            {
                if(vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver* pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if(bNoMore)
                {
                    vbDiscarded[i]=true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if(!Tcw.empty())
                {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint*> sFound;

                    for(size_t j=0; j<vbInliers.size(); j++)
                    {
                        if(vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        }
                        else
                            mCurrentFrame.mvpMapPoints[j]=NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if(nGood<10)
                        continue;

                    for(size_t io =0, ioend=mCurrentFrame.mvbOutlier.size(); io<ioend; io++)
                        if(mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io]=NULL;

                    // If few inliers, search by projection in a coarse window and optimize again
                    if(nGood<50)
                    {
                        int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                        if(nadditional+nGood>=50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if(nGood>20 && nGood<50)
                            {
                                sFound.clear();
                                for(size_t ip =0, ipend=mCurrentFrame.mvpMapPoints.size(); ip<ipend; ip++)
                                    if(mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                                // Final optimization
                                if(nGood+nadditional>=30)
                                {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for(size_t io =0; io<mCurrentFrame.mvbOutlier.size(); io++)
                                        if(mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io]=NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if(nGood>=30)
                    {                    
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if(!bMatch)
        {
            return false;
        }
        else
        {   
            if(!mpLocalMapper->GetVINSInited()) cerr<<"VINS not inited? why."<<endl;
            if(mMode!=MONO){
                mbRelocBiasPrepare = true;
            }
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }

    }

    void Tracking::ForceRelocalisation()
    {
        boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
        mbForceRelocalisation = true;
        mnLastRelocFrameId = mCurrentFrame.mnId;
    }

    bool Tracking::RelocalisationRequested()
    {
        boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
        return mbForceRelocalisation;
    }


    void Tracking::Reset()
    {
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            mbPublisherStopped = false;
            mbReseting = true;
        }

        // Wait until publishers are stopped
        ros::Rate r(500);
        while(1)
        {
            {
                boost::mutex::scoped_lock lock(mMutexReset);
                if(mbPublisherStopped)
                    break;
            }
            r.sleep();
        }

        // Reset Local Mapping
        mpLocalMapper->RequestReset();
        // Reset Loop Closing
        mpLoopClosing->RequestReset();
        // Clear BoW Database
        mpKeyFrameDB->clear();
        // Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        FrameKTL::nNextId = 0;
        mState = NOT_INITIALIZED;

        {
            boost::mutex::scoped_lock lock(mMutexReset);
            mbReseting = false;
        }
    }

    void Tracking::CheckResetByPublishers()
    {
        bool bReseting = false;

        {
            boost::mutex::scoped_lock lock(mMutexReset);
            bReseting = mbReseting;
        }

        if(bReseting)
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            mbPublisherStopped = true;
        }

        // Hold until reset is finished
        ros::Rate r(500);
        while(1)
        {
            {
                boost::mutex::scoped_lock lock(mMutexReset);
                if(!mbReseting)
                {
                    mbPublisherStopped=false;
                    break;
                }
            }
            r.sleep();
        }
    }

    void Tracking::feed_imu_data(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am) 
    {

            // Create our imu data object
            IMUData data;
            data.timestamp = timestamp;
            data.wm = wm;
            data.am = am;
            imu_data.emplace_back(data);
            auto it0 = imu_data.begin();
            while(it0 != imu_data.end()) {
                if(timestamp-(*it0).timestamp > 35) {
                    it0 = imu_data.erase(it0);
                } else {
                    it0++;
                }
            }

    }

    void Tracking::feed_depth_data(double timestamp, double depth) 
    {

            DepthData data;
            data.timestamp = timestamp;
            data.depth=depth;
            
            depth_data.emplace_back(data);
            auto it0 = depth_data.begin();
            while(it0 != depth_data.end()) {
                if(timestamp-(*it0).timestamp > 20) {
                    it0 = depth_data.erase(it0);
                } else {
                    it0++;
                }
            }

    }

    std::vector<IMUData> Tracking::select_imu_readings(double time0, double time1) 
    {

        // Our vector imu readings
        std::vector<IMUData> prop_data;
        if(imu_data.empty()) {
            printf(YELLOW "Check the Mode of Operation !!!! or No IMU measurements. IMU-CAMERA are likely messed up!!!\n" RESET);
            return prop_data;
        }

        for(size_t i=0; i<imu_data.size()-1; i++) {

            
            if(imu_data.at(i+1).timestamp > time0 && imu_data.at(i).timestamp < time0) {
                IMUData data = Tracking::interpolate_data(imu_data.at(i),imu_data.at(i+1), time0);
                prop_data.push_back(data);
                //printf("propagation #%d = CASE 1 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,time0-prop_data.at(0).timestamp);
                continue;
            }

            if(imu_data.at(i).timestamp < time1  && imu_data.at(i).timestamp >= time0 ) {
                
                prop_data.push_back(imu_data.at(i));
                //printf("propagation #%d = CASE 3.2 = %.3f => %.3f\n", (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
                
            }
            if(imu_data.at(i).timestamp > time1){
                break;
            }

        }
        if(prop_data.empty()) {
            printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET, (int)prop_data.size());
            return prop_data;
        }
        for (size_t i=0; i < prop_data.size()-1; i++) {
            if (std::abs(prop_data.at(i+1).timestamp-prop_data.at(i).timestamp) < 1e-12) {
                printf(YELLOW "Propagator::select_imu_readings(): Zero DT between IMU reading %d and %d, removing it!\n" RESET, (int)i, (int)(i+1));
                prop_data.erase(prop_data.begin()+i);
                i--;
            }
        }
        if(prop_data.size() < 2) {
            printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET, (int)prop_data.size());
            return prop_data;
        }
        return prop_data;
    }
    bool Tracking::select_depth_readings_txt(double time0, double time1, double& depth) 
    {   /*
        ifstream inFile;   
        inFile.open("/home/chinthaka/catkin_ws/src/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_V2_01/stamped_groundtruth.txt");
        if (!inFile) {
            cout << "Unable to open file";
            exit(1); // terminate with error
        }
        const double mean = 0.0;
        const double stddev = 0.01;
        std::default_random_engine generator;
        std::normal_distribution<double> dist(mean, stddev);     

        for(std::string line; std::getline(inFile, line); )   //read stream line by line
        {
            std::istringstream in(line);      //make a stream for the line itself
            double t_time;
            in >> t_time;                  //and read the first whitespace-separated token

            if(t_time == time1)       //and check its value
            {
                double x, y, z;
                in >> x >> y >> z;       //now read the whitespace-separated floats
                depth = z + dist(generator);
                   
            }
            if(t_time > time1) 
            {
                break;
            }
        }*/


        
        bool depthInterpolate = true; // this should be false for harbor sequence
        // Our vector imu readings
        ifstream inFile;   
        inFile.open("/home/chinthaka/USLAM/Trajectories/euroc/VI/MH_05/stamped_groundtruth.txt");
        if (!inFile) {
            cout << "Unable to open EuroC ground thruth txt file";
            exit(1); // terminate with error
        }
        std::vector<double> depth_vec;
        double last_depth0, last_depth1;
        double last_time0, last_time1;


        for(std::string line; std::getline(inFile, line); )   //read stream line by line
        {
            std::istringstream in(line);      //make a stream for the line itself
            double t_time;
            in >> t_time;                  //and read the first whitespace-separated token

            if(t_time > time0 && t_time < time1)       //and check its value
            {
                double x, y, z;
                in >> x >> y >> z;       //now read the whitespace-separated floats
                depth_vec.push_back(z);
                last_time0 = t_time;
                last_depth0 = z;
            }

            if(t_time > time1) 
            {
                double x, y, z;
                in >> x >> y >> z;
                last_depth1 = z;
                last_time1 = t_time;
                break;
            }
        }

        if(depth_vec.empty()) {
            return false;
        }
        else{
            auto n = depth_vec.size(); 
            if ( n != 0) {
                depth = std::accumulate(depth_vec.begin(), depth_vec.end(), 0.0) / n; 
            }

            if(depthInterpolate)
            {
                double t1 = time1 - last_time0;
                double t2 = last_time1 - last_time0;
                depth = last_depth0 + (last_depth1-last_depth0)*t1/t2;
            }
            return true;
        } 
        
    }


    bool Tracking::select_depth_readings(double time0, double time1,double& depth, double& depth_time) 
    {
        bool depthInterpolate = false; // this should be false for harbor sequence
        // Our vector imu readings
        
        if(depth_data.empty()) {
            printf(YELLOW "Check the Mode of operation V-Only ? VI ? VIP ?  OR No depth measurements. depth-CAMERA are likely messed up!!!\n" RESET);
            return false;
        }
        std::vector<double> depth_vec;
        std::vector<double> time_vec;
        size_t nextDepthID = 0;
        for(size_t i=0; i<depth_data.size(); i++) {
           
            if(depth_data.at(i).timestamp > time0  && depth_data.at(i).timestamp < time1 ) {
                
                depth_vec.push_back(depth_data.at(i).depth);
                time_vec.push_back(depth_data.at(i).timestamp);
                nextDepthID = i;
            }
        }
        if(depth_vec.empty()) {
            return false;
        }
        else{
            auto n = depth_vec.size(); 
            if ( n != 0) {
                depth = std::accumulate(depth_vec.begin(), depth_vec.end(), 0.0) / n; 
                depth_time = std::accumulate(time_vec.begin(), time_vec.end(), 0.0) / n;
            }

            if(depthInterpolate)
            {
                double t1 = time1 - depth_data.at(nextDepthID).timestamp;
                double t2 = depth_data.at(nextDepthID+1).timestamp - depth_data.at(nextDepthID).timestamp;
                depth = depth_data.at(nextDepthID).depth + (depth_data.at(nextDepthID+1).depth - depth_data.at(nextDepthID).depth)*t1/t2;
                depth_time = time1;
            }
            return true;
        } 
    }


    void Tracking::RecomputeIMUBiasAndCurrentNavstate(NavState& nscur)
    {
        size_t N = mv20FramesReloc.size();

        //Test log
        if(N!=20) cerr<<"Frame vector size not 20 to compute bias after reloc??? size: "<<mv20FramesReloc.size()<<endl;

        // Estimate gyr bias
        Vector3d bg = Optimizer::OptimizeInitialGyroBias(mv20FramesReloc);
        // Update gyr bias of Frames
        for(size_t i=0; i<N; i++)
        {
            FrameKTL& frame = mv20FramesReloc[i];
            //Test log
            if(frame.GetNavState().Get_BiasGyr().norm()!=0 || frame.GetNavState().Get_dBias_Gyr().norm()!=0)
                cerr<<"Frame "<<frame.mnId<<" gyr bias or delta bias not zero???"<<endl;

            frame.SetNavStateBiasGyr(bg);
        }
        // Re-compute IMU pre-integration
        vector<IMUPreintegrator> v19IMUPreint;
        v19IMUPreint.reserve(20-1);
        for(size_t i=0; i<N; i++)
        {
            if(i==0)
                continue;

            const FrameKTL& Fi = mv20FramesReloc[i-1];
            const FrameKTL& Fj = mv20FramesReloc[i];

            IMUPreintegrator imupreint;
            Fj.ComputeIMUPreIntSinceLastFrame(&Fi,imupreint);
            v19IMUPreint.push_back(imupreint);
        }
        // Construct [A1;A2;...;AN] * ba = [B1;B2;.../BN], solve ba
        cv::Mat A = cv::Mat::zeros(3*(N-2),3,CV_32F);
        cv::Mat B = cv::Mat::zeros(3*(N-2),1,CV_32F);
        const cv::Mat& gw = mpLocalMapper->GetGravityVec();
        const cv::Mat& Tcb = ConfigParam::GetMatT_cb();
        for(int i=0; i<N-2; i++)
        {
            const FrameKTL& F1 = mv20FramesReloc[i];
            const FrameKTL& F2 = mv20FramesReloc[i+1];
            const FrameKTL& F3 = mv20FramesReloc[i+2];
            const IMUPreintegrator& PreInt12 = v19IMUPreint[i];
            const IMUPreintegrator& PreInt23 = v19IMUPreint[i+1];
            // Delta time between frames
            double dt12 = PreInt12.getDeltaTime();
            double dt23 = PreInt23.getDeltaTime();
            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(PreInt12.getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(PreInt12.getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(PreInt23.getDeltaP());
            cv::Mat Jpba12 = Converter::toCvMat(PreInt12.getJPBiasa());
            cv::Mat Jvba12 = Converter::toCvMat(PreInt12.getJVBiasa());
            cv::Mat Jpba23 = Converter::toCvMat(PreInt23.getJPBiasa());
            // Pose of body in world frame
            cv::Mat Twb1 = Converter::toCvMatInverse(F1.mTcw)*Tcb;
            cv::Mat Twb2 = Converter::toCvMatInverse(F2.mTcw)*Tcb;
            cv::Mat Twb3 = Converter::toCvMatInverse(F3.mTcw)*Tcb;
            // Position of body, Pwb
            cv::Mat pb1 = Twb1.rowRange(0,3).col(3);
            cv::Mat pb2 = Twb2.rowRange(0,3).col(3);
            cv::Mat pb3 = Twb3.rowRange(0,3).col(3);
            // Rotation of body, Rwb
            cv::Mat Rb1 = Twb1.rowRange(0,3).colRange(0,3);
            cv::Mat Rb2 = Twb2.rowRange(0,3).colRange(0,3);
            //cv::Mat Rb3 = Twb3.rowRange(0,3).colRange(0,3);
            // Stack to A/B matrix
            // Ai * ba = Bi
            cv::Mat Ai = Rb1*Jpba12*dt23 - Rb2*Jpba23*dt12 - Rb1*Jvba12*dt12*dt23;
            cv::Mat Bi = (pb2-pb3)*dt12 + (pb2-pb1)*dt23 + Rb2*dp23*dt12 - Rb1*dp12*dt23 + Rb1*dv12*dt12*dt23 + 0.5*gw*(dt12*dt12*dt23+dt12*dt23*dt23);
            Ai.copyTo(A.rowRange(3*i+0,3*i+3));
            Bi.copyTo(B.rowRange(3*i+0,3*i+3));

            //Test log
            if(fabs(F2.mTimeStamp-F1.mTimeStamp-dt12)>1e-6 || fabs(F3.mTimeStamp-F2.mTimeStamp-dt23)>1e-6) cerr<<"delta time not right."<<endl;

            //        // lambda*s + phi*dthetaxy + zeta*ba = psi
            //        cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
            //        cv::Mat phi = - 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*Rwi*SkewSymmetricMatrix(GI);  // note: this has a '-', different to paper
            //        cv::Mat zeta = Rc2*Rcb*Jpba23*dt12 + Rc1*Rcb*Jvba12*dt12*dt23 - Rc1*Rcb*Jpba12*dt23;
            //        cv::Mat psi = (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - (Rc2-Rc3)*pcb*dt12
            //                     - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt23*dt12 - 0.5*Rwi*GI*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper
            //        lambda.copyTo(C.rowRange(3*i+0,3*i+3).col(0));
            //        phi.colRange(0,2).copyTo(C.rowRange(3*i+0,3*i+3).colRange(1,3)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
            //        zeta.copyTo(C.rowRange(3*i+0,3*i+3).colRange(3,6));
            //        psi.copyTo(D.rowRange(3*i+0,3*i+3));

        }

        // Use svd to compute A*x=B, x=ba 3x1 vector
        // A = u*w*vt, u*w*vt*x=B
        // Then x = vt'*winv*u'*B
        cv::Mat w2,u2,vt2;
        // Note w2 is 3x1 vector by SVDecomp()
        // A is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
        cv::SVDecomp(A,w2,u2,vt2,cv::SVD::MODIFY_A);
        // Compute winv
        cv::Mat w2inv=cv::Mat::eye(3,3,CV_32F);
        for(int i=0;i<3;i++)
        {
            if(fabs(w2.at<float>(i))<1e-10)
            {
                w2.at<float>(i) += 1e-10;
                // Test log
                cerr<<"w2(i) < 1e-10, w="<<endl<<w2<<endl;
            }
            w2inv.at<float>(i,i) = 1./w2.at<float>(i);
        }
        // Then y = vt'*winv*u'*B
        cv::Mat ba_cv = vt2.t()*w2inv*u2.t()*B;
        Vector3d ba = Converter::toVector3d(ba_cv);

        // Update acc bias
        for(size_t i=0; i<N; i++)
        {
            FrameKTL& frame = mv20FramesReloc[i];
            //Test log
            if(frame.GetNavState().Get_BiasAcc().norm()!=0 || frame.GetNavState().Get_dBias_Gyr().norm()!=0 || frame.GetNavState().Get_dBias_Acc().norm()!=0)
                cerr<<"Frame "<<frame.mnId<<" acc bias or delta bias not zero???"<<endl;

            frame.SetNavStateBiasAcc(ba);
        }

        // Compute Velocity of the last 2 Frames
        Vector3d Pcur;
        Vector3d Vcur;
        Matrix3d Rcur;
        {
            FrameKTL& F1 = mv20FramesReloc[N-2];
            FrameKTL& F2 = mv20FramesReloc[N-1];
            const IMUPreintegrator& imupreint = v19IMUPreint.back();
            const double dt12 = imupreint.getDeltaTime();
            const Vector3d dp12 = imupreint.getDeltaP();
            const Vector3d gweig = Converter::toVector3d(gw);
            const Matrix3d Jpba12 = imupreint.getJPBiasa();
            const Vector3d dv12 = imupreint.getDeltaV();
            const Matrix3d Jvba12 = imupreint.getJVBiasa();

            // Velocity of Previous Frame
            // P2 = P1 + V1*dt12 + 0.5*gw*dt12*dt12 + R1*(dP12 + Jpba*ba + Jpbg*0)
            cv::Mat Twb1 = Converter::toCvMatInverse(F1.mTcw)*Tcb;
            cv::Mat Twb2 = Converter::toCvMatInverse(F2.mTcw)*Tcb;
            Vector3d P1 = Converter::toVector3d(Twb1.rowRange(0,3).col(3));
            /*Vector3d */Pcur = Converter::toVector3d(Twb2.rowRange(0,3).col(3));
            Matrix3d R1 = Converter::toMatrix3d(Twb1.rowRange(0,3).colRange(0,3));
            /*Matrix3d */Rcur = Converter::toMatrix3d(Twb2.rowRange(0,3).colRange(0,3));
            Vector3d V1 = 1./dt12*( Pcur - P1 - 0.5*gweig*dt12*dt12 - R1*(dp12 + Jpba12*ba) );

            // Velocity of Current Frame
            Vcur = V1 + gweig*dt12 + R1*( dv12 + Jvba12*ba );

            // Test log
            if(F2.mnId != mCurrentFrame.mnId) cerr<<"framecur.mnId != mCurrentFrame.mnId. why??"<<endl;
            if(fabs(F2.mTimeStamp-F1.mTimeStamp-dt12)>1e-6) cerr<<"timestamp not right?? in compute vel"<<endl;
        }

        // Set NavState of Current Frame, P/V/R/bg/ba/dbg/dba
        nscur.Set_Pos(Pcur);
        nscur.Set_Vel(Vcur);
        nscur.Set_Rot(Rcur);
        nscur.Set_BiasGyr(bg);
        nscur.Set_BiasAcc(ba);
        nscur.Set_DeltaBiasGyr(Vector3d::Zero());
        nscur.Set_DeltaBiasAcc(Vector3d::Zero());

        //mv20FramesReloc
    }
    bool Tracking::IMU_Relocalisation()
    {
        //PredictNavStateByIMU(false);
        ORBmatcher matcher(0.9,false);  
        set<MapPoint*> sFound;
        mCurrentFrame.mvpMapPoints.resize(mCurrentFrame.GetN());
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
       
        int nmatches = matcher.SearchByProjection(mCurrentFrame,mpLastKeyFrame,sFound,10,100);
        
        // If few matches, uses a wider window search
        if(nmatches<60)
        {
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); 
            int nmatches = matcher.SearchByProjection(mCurrentFrame,mpLastKeyFrame,sFound,15,200);
        }

        if(nmatches<60){
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
            return false;
        }
        IMUPreintegrator imupreint = GetIMUPreIntSinceLastKF(&mCurrentFrame, mpLastKeyFrame, mvIMUSinceLastKF);
        //int nmatches2= Optimizer::PoseOptimization(&mCurrentFrame);
        nmatches=Optimizer::PoseOptimization(&mCurrentFrame,mpLastKeyFrame,imupreint,mpLocalMapper->GetGravityVec(),
                mpLocalMapper->GetGravityRotation(),true,ini_depth,depth_cov);
        //cout<<nmatches2<<" "<<nmatches<<endl;
        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.GetN(); i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }
        if (nmatchesMap>=25){
            //mbRelocBiasPrepare = true;
            mbCreateNewKFAfterReloc=true;
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }
    }  

    void Tracking::calculate_G()
    {
        Eigen::Matrix<double,3,1> linsum = Eigen::Matrix<double,3,1>::Zero();
        for(IMUData data : imu_data) {
            linsum += data.am;
        }
        Eigen::Vector3d linavg = Eigen::Vector3d::Zero();
        linavg = linsum/imu_data.size();
        z_axis = linavg/linavg.norm();
    }
    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.GetN(); i++)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint* pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

}//namespace USLAM