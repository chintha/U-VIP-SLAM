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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "FrameKTL.h"
#include "ORBVocabulary.h"
#include "hash.h"

#include<boost/thread.hpp>


namespace USLAM
{

class KeyFrame;
class FrameKTL;


class KeyFrameDatabase
{
public:

    KeyFrameDatabase(const ORBVocabulary &voc);

   void add(KeyFrame* pKF);

   void erase(KeyFrame* pKF);

   void clear();

   // Loop Detection
   std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);
   std::vector<KeyFrame *> DetectLoopCandidatesHaloc(KeyFrame* pKF, float minScore, haloc::Hash* haloc);
   std::vector<KeyFrame *> kfVec;
   vector< pair<int, int > > cluster_lc_found_;
   
   static const int LC_DISCARD_WINDOW = 10;

   // Relocalisation
   std::vector<KeyFrame*> DetectRelocalisationCandidates(FrameKTL* F);

protected:

  // Associated vocabulary
  const ORBVocabulary* mpVoc;
  static bool sortByMatching(const pair<KeyFrame*,float> &d1,const pair<KeyFrame*,float> &d2);

  // Inverted file
  std::vector<list<KeyFrame*> > mvInvertedFile;

  // Mutex
  boost::mutex mMutex;
};

} //namespace USLAM

#endif
