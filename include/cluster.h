/**
 * @file
 * @brief The cluster class represents one cv::KeyPoints clustering of the camera frame.
 */

#ifndef CLUSTER_H
#define CLUSTER_H

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"

using namespace std;

namespace USLAM
{
class MapPoint;
class KeyFrame;

class Cluster
{

public:

  /** \brief Class constructor
   */
  Cluster();

  /** \brief Class constructor
   */
  Cluster(int id, int frame_id, cv::Mat camera_pose, vector<cv::KeyPoint> kp, cv::Mat orb_desc,  vector<MapPoint*> points);

  /** \brief Computes and returns the 3D points in world coordinates
   * @return the 3D points in world coordinates
   */
  vector<MapPoint*> getWorldPoints();

  /** \brief Get the cluster id
   */
  inline long unsigned int getId() const {return id_;}

  /** \brief Get the frame id
   */
  inline int getKeyFrameId() const {return Keyframe_id_;}

  /** \brief Get left cv::KeyPoints
   */
  inline vector<cv::KeyPoint> getKp() const {return kp_;}

  
  /** \brief Get orb descriptors
   */
  inline cv::Mat getOrb() const {return orb_desc_;}

  /** \brief Get 3D camera points
   */
  inline vector<MapPoint*> getPoints() const {return points_;}

  /** \brief Get camera pose
   */
  inline cv::Mat getCameraPose() const {return camera_pose_;}

  static MapPoint* transformPoint(MapPoint* point, cv::Mat base);
  

private:


  long unsigned int id_; //!> Cluster id

  long unsigned int Keyframe_id_; //!> Corresponding frame id

  KeyFrame* mCuKeyFrame;

  cv::Mat camera_pose_; //!> Camera world position

  vector<cv::KeyPoint> kp_; //!> left cv::KeyPoints.


  cv::Mat orb_desc_; //!> ORB descriptors

  vector<MapPoint*> points_; //!> Stereo 3D points in camera frame

};

} // namespace

#endif // CLUSTER_H