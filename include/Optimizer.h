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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "FrameKTL.h"
#include <string> 

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace USLAM
{

class LoopClosing;

class Optimizer
{

public:
    void static GlobalBundleAdjustmentNavState(Map* pMap, const cv::Mat& gw, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust);

    int static PoseOptimization(FrameKTL *pFrame, KeyFrame* pLastKF, const IMUPreintegrator& imupreint,const cv::Mat& gw, const cv::Mat& grot, const bool& bComputeMarg=false,double ini_depth=0.0, double depth_cov=0.0);
    int static PoseOptimization(FrameKTL *pFrame, FrameKTL* pLastFrame, const IMUPreintegrator& imupreint, const cv::Mat& gw,const cv::Mat& grot, const bool& bComputeMarg=false, double ini_depth=0.0, double depth_cov=0.0);

    void static LocalBundleAdjustmentNavState(KeyFrame *pKF, const std::list<KeyFrame*> &lLocalKeyFrames, bool* pbStopFlag, Map* pMap,
     const cv::Mat& gw,const cv::Mat& grot,LocalMapping* pLM,double ini_depth, double depth_cov,int fixedKF_id);

    Vector3d static OptimizeInitialGyroBias(const std::list<KeyFrame*> &lLocalKeyFrames);
    Vector3d static OptimizeInitialGyroBias(const std::vector<KeyFrame*> &vLocalKeyFrames);
    Vector3d static OptimizeInitialGyroBias(const std::vector<FrameKTL> &vFrames);

    double static OptimizeInitialScale(const std::vector<KeyFrame*> &vLocalKeyFrames, Eigen::Matrix3d& Rg_w,double ini_depth,double &AVG);
    double static OptimizeInitialScaleSecond(const std::vector<KeyFrame*> &vLocalKeyFrames, Eigen::Matrix3d& Rg_w,double ini_depth,cv::Mat biasa, double depth_cov=0.0);
    

    
public:
    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP, int nIterations = 5, bool *pbStopFlag=NULL);
    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL);
    void static RecoveryBundleAdjustemnt(KeyFrame* pKFini,KeyFrame* pKFcur,int nIterations,bool *pbStopFlag=NULL);
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag,Map* pMap,LocalMapping* pLM);// originally pbStopFlag==NULL
    int static PoseOptimization(FrameKTL* pFrame);

    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, g2o::Sim3 &Scurw,
                                       LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       std::map<KeyFrame*, set<KeyFrame*> > &LoopConnections);


    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, float th2 = 10);
    
};

} //namespace USLAM

#endif // OPTIMIZER_H
