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
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.c
*/

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include<set>

#include<boost/thread.hpp>



namespace USLAM
{

class MapPoint;
class KeyFrame;

class KFIdComapre
{
public:
    bool operator()(const KeyFrame* kfleft,const KeyFrame* kfright) const;
};

class Map
{
public:
    // Update after an absolute scale is available
    void UpdateScale(const double &scale,const cv::Mat &grot);

public:
    Map();
    
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetCurrentCameraPose(cv::Mat Tcw);
    void SetReferenceKeyFrames(const std::vector<KeyFrame*> &vpKFs);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void addpose(cv::Mat pose);

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    cv::Mat GetCameraPose();
    std::vector<KeyFrame*> GetReferenceKeyFrames();
    std::vector<MapPoint*> GetReferenceMapPoints();
    std::vector<cv::Mat> GetLastFrames();

    int MapPointsInMap();
    int KeyFramesInMap();

    void SetFlagAfterBA();
    bool isMapUpdated();
    void ResetUpdated();

    unsigned int GetMaxKFid();

    void clear();
    boost::mutex mMutexMapUpdate;

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*,KFIdComapre> mspKeyFrames;
    std::vector<cv::Mat> Last_Frames;
    std::vector<MapPoint*> mvpReferenceMapPoints;
    

    unsigned int mnMaxKFid;

    boost::mutex mMutexMap;
    
    bool mbMapUpdated;
};

} //namespace USLAM

#endif // MAP_H
