#include "cluster.h"

namespace USLAM
{
  Cluster::Cluster() : id_(-1){}

  Cluster::Cluster(int id,int Keyframe_id, cv::Mat camera_pose, vector<cv::KeyPoint> kp, cv::Mat orb_desc,  vector<MapPoint*> points) :
                  id_(id), Keyframe_id_(Keyframe_id),camera_pose_(camera_pose), kp_(kp),orb_desc_(orb_desc),points_(points){}


  vector<MapPoint*> Cluster::getWorldPoints()
  {
    vector<MapPoint*> out;
    for (uint i=0; i<points_.size(); i++)
    {
      MapPoint* p = Cluster::transformPoint(points_[i], camera_pose_);
      out.push_back(p);
    }

    return out;
  }

  MapPoint* Cluster::transformPoint(MapPoint* point, cv::Mat base){
    
    /*
    tf::Transform out;
    out.setIdentity();
    tf::Vector3 t_out(base.at<float>(0,3),base.at<float>(1,3),base.at<float>(2,3));
    out.setOrigin(t_out);
    // this is wrong Correct it Chinthaka......

    tf::Vector3 p_tf(point.x, point.y, point.z);
    tf::Vector3 p_tf_world = out * p_tf;
    cv::Point3f new_point(p_tf_world.x(), p_tf_world.y(), p_tf_world.z());
    return new_point;

    */
   return point;
  } 
}