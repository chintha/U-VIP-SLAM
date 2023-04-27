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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include <boost/thread.hpp>
#include "KeyFrameDatabase.h"
#include "hash.h"
#include "cluster.h"
#include "src/IMU/configparam.h"
#include "Converter.h"


namespace USLAM
{

class Tracking;
class LoopClosing;
class Map;
class LoopClosingHaloc;

class LocalMapping
{
public:
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        INITIALIZING=2,
        WORKING=3,
        LOST=4,
        IMURELOCALIZATION=5,
        R_INITIALIZING=6
    };
    ConfigParam* mpParams;
    void AddToLocalWindow(KeyFrame* pKF);
    void DeleteBadInLocalWindow(void);

    bool TryInitVIO(void);
    bool GetVINSInited(void);
    void SetVINSInited(bool flag);

    bool GetFirstVINSInited(void);
    void SetFirstVINSInited(bool flag);

    cv::Mat GetGravityVec(void);
    cv::Mat GetGravityRotation(void);

    bool GetMapUpdateFlagForTracking();
    void SetMapUpdateFlagInTracking(bool bflag);
    KeyFrame* GetMapUpdateKF();
    bool isFinished();
    void RequestFinish();
    bool needboth_Inits=true;
    double mnStartTime;
    bool mbFirstTry;
    double mnVINSInitScale;
    bool mbFinished;
    cv::Mat mGravityVec; // gravity vector in map frame
    cv::Mat mGravityRot; // Rotation map frame to Globle frame

    boost::mutex mMutexVINSInitFlag;
    bool mbVINSInited;

    boost::mutex mMutexFirstVINSInitFlag;
    bool mbFirstVINSInited;

    unsigned int mnLocalWindowSize;
    std::list<KeyFrame*> mlLocalKeyFrames;

    boost::mutex mMutexMapUpdateFlag;
    bool mbMapUpdateFlagForTracking;
    KeyFrame* mpMapUpdateKF;
    void SetFinish();
    bool mbFinishRequested;
    bool CheckFinish();
    bool Loop_Closer=false;
    int ini_time_limit=25;
    
public:

    
    LocalMapping(Map* pMap,ConfigParam* pParams,string strSettingPath);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    void SetLoopCloserHaloc(LoopClosingHaloc* ploop_closing_haloc);

    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();

    void Stop();

    void Release();

    bool isStopped();

    bool stopRequested();

    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);

    void InterruptBA();

    haloc::Hash haloc;

    static int ClusterId;

    KeyFrame* searchKF_loop_closer(int cluster_id);

    void getCandidates_Proximity(int cluster_id, int window_center, int window, int best_n, vector<int> &neighbors, vector<int> no_candidates);

    static double poseDiff2D(Eigen::Vector4f pose_1, Eigen::Vector4f pose_2);
    
    static bool sortByDistance(const pair<int, double> d1, const pair<int, double> d2);
   
    KeyFrame* mpLastKeyFrame;
    
    void clearLocalKF();

    int fixedKF_id=0;

    int KeyframesInQueue(){
        boost::mutex::scoped_lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }
 
protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    void CreateNewMapPointsEdited();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    void ResetIfRequested();
    bool mbResetRequested;
    boost::mutex mMutexReset;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    
    Tracking* mpTracker;

    LoopClosingHaloc* mploop_closing_haloc;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;
    

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    boost::mutex mMutexNewKFs;    

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    boost::mutex mMutexStop;
    boost::mutex mMutexFinish;
    

    bool mbAcceptKeyFrames;
    boost::mutex mMutexAccept;

private:

    vector<Eigen::Vector4f> initial_cluster_pose_history_;
    vector< pair< int,KeyFrame* > > cluster_frame_relation_; //!> Stores the cluster/frame relation (cluster_id, frame_id)
    
};

} //namespace USLAM

#endif // LOCALMAPPING_H
