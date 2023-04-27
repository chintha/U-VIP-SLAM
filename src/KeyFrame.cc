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

#include "KeyFrame.h"
#include "Converter.h"
#include <ros/ros.h>

namespace USLAM
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(FrameKTL &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mfGridElementWidthInv(F.mfGridElementWidthInv),
    mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0),mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnRelocQuery(0),fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), mBowVec(F.mBowVec),
    im(F.im), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK),
    mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mDescriptors(F.mDescriptors.clone()),
    mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mFeatVec(F.mFeatVec),
    mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
    mnScaleLevels(F.mnScaleLevels), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mpMap(pMap),mnBAGlobalForKF(0),mDepth(F.mDepth),mHas_depth(F.mHas_depth),mDepth_time(F.mDepth_time),mfLogScaleFactor(F.mfLogScaleFactor)
{
    mnId=nNextId++;
    RL_flag=false;
    mnGridCols=FRAME_GRID_COLS;
    mnGridRows=FRAME_GRID_ROWS;
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

KeyFrame::KeyFrame(FrameKTL &F, Map *pMap, KeyFrameDatabase *pKFDB,std::vector<IMUData> vIMUData,KeyFrame* pPrevKF):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mfGridElementWidthInv(F.mfGridElementWidthInv),
    mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0),mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnRelocQuery(0),fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), mBowVec(F.mBowVec),
    im(F.im), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK),
    mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mDescriptors(F.mDescriptors.clone()),
    mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mFeatVec(F.mFeatVec),
    mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
    mnScaleLevels(F.mnScaleLevels), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mpMap(pMap),mnBAGlobalForKF(0),mDepth(F.mDepth),mHas_depth(F.mHas_depth),mDepth_time(F.mDepth_time),mfLogScaleFactor(F.mfLogScaleFactor)
{
    mvIMUData = vIMUData;
    if(pPrevKF)
    {
        pPrevKF->SetNextKeyFrame(this);
    }
    mpPrevKeyFrame = pPrevKF;
    mpNextKeyFrame = NULL;
    mnId=nNextId++;

    mnGridCols=FRAME_GRID_COLS;
    mnGridRows=FRAME_GRID_ROWS;
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }
    RL_flag=false;
    SetPose(F.mTcw);    
}
void KeyFrame::UpdateNavStatePVRFromTcw(const cv::Mat &Tcw,const cv::Mat &Tbc)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    cv::Mat Twb = Converter::toCvMatInverse(Tbc*Tcw);
    Matrix3d Rwb = Converter::toMatrix3d(Twb.rowRange(0,3).colRange(0,3));
    Vector3d Pwb = Converter::toVector3d(Twb.rowRange(0,3).col(3));

    Matrix3d Rw1 = mNavState.Get_RotMatrix();
    Vector3d Vw1 = mNavState.Get_V();
    Vector3d Vw2 = Rwb*Rw1.transpose()*Vw1;   // bV1 = bV2 ==> Rwb1^T*wV1 = Rwb2^T*wV2 ==> wV2 = Rwb2*Rwb1^T*wV1

    mNavState.Set_Pos(Pwb);
    mNavState.Set_Rot(Rwb);
    mNavState.Set_Vel(Vw2);
}

void KeyFrame::SetInitialNavStateAndBias(const NavState& ns)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState = ns;
    // Set bias as bias+delta_bias, and reset the delta_bias term
    mNavState.Set_BiasGyr(ns.Get_BiasGyr()+ns.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(ns.Get_BiasAcc()+ns.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}
void KeyFrame::UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    Converter::updateNS(mNavState,imupreint,gw);
}

void KeyFrame::SetNavState(const NavState& ns)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState = ns;
}

const NavState& KeyFrame::GetNavState(void)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    return mNavState;
}

void KeyFrame::SetNavStateBiasGyr(const Vector3d &bg)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_BiasGyr(bg);
}

void KeyFrame::SetNavStateBiasAcc(const Vector3d &ba)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_BiasAcc(ba);
}

void KeyFrame::SetNavStateVel(const Vector3d &vel)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_Vel(vel);
}

void KeyFrame::SetNavStatePos(const Vector3d &pos)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_Pos(pos);
}

void KeyFrame::SetNavStateRot(const Matrix3d &rot)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_Rot(rot);
}

void KeyFrame::SetNavStateRot(const Sophus::SO3 &rot)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_Rot(rot);
}

void KeyFrame::SetNavStateDeltaBg(const Vector3d &dbg)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_DeltaBiasGyr(dbg);
}

void KeyFrame::SetNavStateDeltaBa(const Vector3d &dba)
{
    boost::mutex::scoped_lock lock(mMutexNavState);
    mNavState.Set_DeltaBiasAcc(dba);
}

KeyFrame* KeyFrame::GetPrevKeyFrame(void)
{
    boost::mutex::scoped_lock lock(mMutexPrevKF);
    return mpPrevKeyFrame;
}

KeyFrame* KeyFrame::GetNextKeyFrame(void)
{
    boost::mutex::scoped_lock lock(mMutexNextKF);
    return mpNextKeyFrame;
}

void KeyFrame::SetPrevKeyFrame(KeyFrame* pKF)
{
    boost::mutex::scoped_lock lock(mMutexPrevKF);
    mpPrevKeyFrame = pKF;
}

void KeyFrame::SetNextKeyFrame(KeyFrame* pKF)
{
    boost::mutex::scoped_lock lock(mMutexNextKF);
    mpNextKeyFrame = pKF;
}
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

const IMUPreintegrator & KeyFrame::GetIMUPreInt(void)
{
    boost::mutex::scoped_lock lock(mMutexIMUData);
    return mIMUPreInt;
}

void KeyFrame::ComputePreInt(void)
{
    boost::mutex::scoped_lock lock(mMutexIMUData);
    if(mpPrevKeyFrame == NULL)
    {
        if(mnId!=0)
        {
            cerr<<"previous KeyFrame is NULL, pre-integrator not changed. id: "<<mnId<<endl;
        }
        return;
    }
    else
    {
        // Debug log
        //cout<<std::fixed<<std::setprecision(3)<<
        //      "gyro bias: "<<mNavState.Get_BiasGyr().transpose()<<
        //      ", acc bias: "<<mNavState.Get_BiasAcc().transpose()<<endl;
        //cout<<std::fixed<<std::setprecision(3)<<
        //      "pre-int terms. prev KF time: "<<mpPrevKeyFrame->mTimeStamp<<endl<<
        //      "pre-int terms. this KF time: "<<mTimeStamp<<endl<<
        //      "imu terms times: "<<endl;

        // Reset pre-integrator first
        mIMUPreInt.reset();

        // IMU pre-integration integrates IMU data from last to current, but the bias is from last
        Vector3d bg = mpPrevKeyFrame->GetNavState().Get_BiasGyr();
        Vector3d ba = mpPrevKeyFrame->GetNavState().Get_BiasAcc();

        // remember to consider the gap between the last KF and the first IMU
        /*
        {
            const IMUData& imu = mvIMUData.front();
            double dt = imu.timestamp - mpPrevKeyFrame->mTimeStamp;
            mIMUPreInt.update(imu.wm - bg,imu.wm - ba,dt);

            // Test log
            if(dt < 0)
            {
                cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this KF vs last imu time: "<<mTimeStamp<<" vs "<<imu.timestamp<<endl;
                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
            }
            // Debug log
            //cout<<std::fixed<<std::setprecision(3)<<imu._t<<", int dt: "<<dt<<"first imu int since prevKF"<<endl;
        }
        */
        // integrate each imu
        for(size_t i=0; i<mvIMUData.size(); i++)
        {
            const IMUData& imu = mvIMUData[i];
            double nextt;
            if(i==mvIMUData.size()-1)
                nextt = mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = mvIMUData[i+1].timestamp;  // regular condition, next is imu data

            // delta time
            double dt = nextt - imu.timestamp;
            // update pre-integrator
            if(dt>0){
                mIMUPreInt.update(imu.wm - bg,imu.am - ba,dt);
            }
            // Debug log
            //cout<<std::fixed<<std::setprecision(3)<<imu._t<<", int dt: "<<dt<<endl;

            // Test log
            if(dt < 0)
            {
                cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu.timestamp<<" vs "<<nextt<<endl;
                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
            }
        }
    }
    // Debug log
    //cout<<"pre-int delta time: "<<mIMUPreInt.getDeltaTime()<<", deltaR:"<<endl<<mIMUPreInt.getDeltaR()<<endl;
}

void KeyFrame::UpdatePoseFromNS(const cv::Mat &Tbc)
{
    cv::Mat Rbc_ = Tbc.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Pbc_ = Tbc.rowRange(0,3).col(3).clone();

    cv::Mat Rwb_ = Converter::toCvMat(mNavState.Get_RotMatrix());
    cv::Mat Pwb_ = Converter::toCvMat(mNavState.Get_P());

    cv::Mat Rcw_ = (Rwb_*Rbc_).t();
    cv::Mat Pwc_ = Rwb_*Pbc_ + Pwb_;
    cv::Mat Pcw_ = -Rcw_*Pwc_;

    cv::Mat Tcw_ = cv::Mat::eye(4,4,CV_32F);
    Rcw_.copyTo(Tcw_.rowRange(0,3).colRange(0,3));
    Pcw_.copyTo(Tcw_.rowRange(0,3).col(3));

    SetPose(Tcw_);
    //testing
    mNavState.Set_BiasGyr(mNavState.Get_BiasGyr() +mNavState.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(mNavState.Get_BiasAcc() +mNavState.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}

void KeyFrame::ComputeHaloc(){
    haloc::Hash haloc;
    cv::Mat des_temp; 
    mHalocVec= haloc.getHash(mDescriptors);// length of this vector is 768 always
}


void KeyFrame::ComputeHaloc(haloc::Hash* haloc){

    mHalocVec= haloc->getHash(mDescriptors);  
}

void KeyFrame::Set_Depth(const std::vector<double> &depth, const std::vector<double> &time)
{
    Depth_Vec=depth;
    Depth_Vec_Time=time;
}

void KeyFrame::SetPose(const cv::Mat &Rcw,const cv::Mat &tcw)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    tcw.copyTo(Tcw.col(3).rowRange(0,3));

    Ow=-Rcw.t()*tcw;
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    
}

cv::Mat KeyFrame::GetPose()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{

    boost::mutex::scoped_lock lock(mMutexPose);
    /*
    cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
    cv::Mat Rwc = (Tcw.rowRange(0,3).colRange(0,3)).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    twc.copyTo(Twc.rowRange(0,3).col(3));
    */
    return Twc.clone();
}

cv::Mat KeyFrame::GetProjectionMatrix()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return mK*Tcw.rowRange(0,3);
}

cv::Mat KeyFrame::GetCameraCenter()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
        

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    boost::mutex::scoped_lock lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mvpMapPoints[idx]=NULL;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=NULL;
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<mvKeys.size(); i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

cv::KeyPoint KeyFrame::GetKeyPointUn(const size_t &idx) const
{
    return mvKeysUn[idx];
}

int KeyFrame::GetKeyPointScaleLevel(const size_t &idx) const
{
    return mvKeysUn[idx].octave;
}

cv::Mat KeyFrame::GetDescriptor(const size_t &idx)
{
    return mDescriptors.row(idx).clone();
}

cv::Mat KeyFrame::GetDescriptors()
{
    return mDescriptors.clone();
}

vector<cv::KeyPoint> KeyFrame::GetKeyPoints() const
{
    return mvKeys;
}

vector<cv::KeyPoint> KeyFrame::GetKeyPointsUn() const
{
    return mvKeysUn;
}

cv::Mat KeyFrame::GetCalibrationMatrix() const
{
    return mK.clone();
}

DBoW2::FeatureVector KeyFrame::GetFeatureVector()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mFeatVec;
}

DBoW2::BowVector KeyFrame::GetBowVector()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mBowVec;
}

std::vector<float> KeyFrame::GetHalocVector(){
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mHalocVec;
}

cv::Mat KeyFrame::GetImage()
{
    boost::mutex::scoped_lock lock(mMutexImage);
    return im.clone();
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        boost::mutex::scoped_lock lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 1;

    map<int,KeyFrame*> OrderedKF;
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }

        OrderedKF[mit->first->mnId]=mit->first;
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    int minID=mnId;

    {
        boost::mutex::scoped_lock lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // finding the minmum ID of the 
        
        for(int i=0; i<mvpOrderedConnectedKeyFrames.size() && i<20;i++){ 
            if(mvpOrderedConnectedKeyFrames[i]->mnId<minID)
                minID=mvpOrderedConnectedKeyFrames[i]->mnId;
            
        }
        
        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
    /*
    if (mnId>2){
        map<int,vector<int>> IndexVectorMatches;
        vector<int> IndexVec= this->IndexforLastKfKp;
        vector<int> IndexVec2;
        KeyFrame* TargetKeyFrame;
        for (int i=mnId-1; i>minID; i--){
            IndexVectorMatches[i]=IndexVec;
            TargetKeyFrame= OrderedKF[i];
            IndexVec2= TargetKeyFrame->IndexforLastKfKp;
            for(int j=0;j<IndexVec.size();j++){
                if(IndexVec[i]==1 || IndexVec[i]==-1)
                    continue;
                IndexVec[i]= IndexVec2[IndexVec[i]];
                cout<<IndexVec.size()<<endl;
            }
        }
        cout<<IndexVectorMatches.size()<<endl;
    }
    */
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}


void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}


void KeyFrame::SetBadFlag()
{   
    {


        
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        boost::mutex::scoped_lock lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        //clearing haloc related

        //clusters_inKF.clear();
        cluster_centroids_.clear();
        clusters_.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysUn.size());

    int nMinCellX = floor((x-mnMinX-r)*mfGridElementWidthInv);
    nMinCellX = max(0,nMinCellX);
    if(nMinCellX>=mnGridCols)
        return vIndices;

    int nMaxCellX = ceil((x-mnMinX+r)*mfGridElementWidthInv);
    nMaxCellX = min(mnGridCols-1,nMaxCellX);
    if(nMaxCellX<0)
        return vIndices;

    int nMinCellY = floor((y-mnMinY-r)*mfGridElementHeightInv);
    nMinCellY = max(0,nMinCellY);
    if(nMinCellY>=mnGridRows)
        return vIndices;

    int nMaxCellY = ceil((y-mnMinY+r)*mfGridElementHeightInv);
    nMaxCellY = min(mnGridRows-1,nMaxCellY);
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(abs(kpUn.pt.x-x)<=r && abs(kpUn.pt.y-y)<=r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

float KeyFrame::ComputeSceneMedianDepth(int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
    boost::mutex::scoped_lock lock(mMutexFeatures);
    boost::mutex::scoped_lock lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(mvpMapPoints.size());
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::regionClustering()
  {
    clusters_.clear();
    vector< vector<int> > clusters;
    const float eps = 50.0;
    const int min_pts = 100;// number of minimum cluster points
    vector<bool> clustered;
    vector<int> noise;
    vector<bool> visited;
    vector<int> neighbor_pts;
    vector<int> neighbor_pts_;
    int c;

    uint no_keys = mvKeys.size();

    //init clustered and visited
    for(uint k=0; k<no_keys; k++)
    {
      clustered.push_back(false);
      visited.push_back(false);
    }
    clusters_ = clusters;

    c = -1;

    //for each unvisited point P in dataset keypoints
    for(uint i=0; i<no_keys; i++)
    {  
        
        if (mvpMapPoints[i]){
            if(!visited[i])
            { 
                // Mark P as visited
                visited[i] = true;
                neighbor_pts = regionQuery(&mvKeys, &mvKeys.at(i), eps);
                if(neighbor_pts.size() < min_pts)
                {
                // Mark P as Noise
                noise.push_back(i);
                clustered[i] = true;
                }
                else
                {
                clusters.push_back(vector<int>());
                c++;

                // expand cluster
                // add P to cluster c
                clusters[c].push_back(i);
                clustered[i] = true;

                // for each point P' in neighbor_pts
                for(uint j=0; j<neighbor_pts.size(); j++)
                {
                    // if P' is not visited
                    if(!visited[neighbor_pts[j]])
                    {
                    // Mark P' as visited
                    visited[neighbor_pts[j]] = true;
                    neighbor_pts_ = regionQuery(&mvKeys, &mvKeys.at(neighbor_pts[j]), eps);
                    if(neighbor_pts_.size() >= min_pts)
                    {
                        neighbor_pts.insert(neighbor_pts.end(), neighbor_pts_.begin(), neighbor_pts_.end());
                    }
                    }
                    // if P' is not yet a member of any cluster
                    // add P' to cluster c
                    if(!clustered[neighbor_pts[j]])
                    {
                    clusters[c].push_back(neighbor_pts[j]);
                    clustered[neighbor_pts[j]] = true;
                    }
                }
            }
        }
      }
    }

    // Discard small clusters
    for (uint i=0; i<clusters.size(); i++)
    {
      if (clusters[i].size() >= min_pts)
        clusters_.push_back(clusters[i]);
      else
      {
        for (uint j=0; j<clusters[i].size(); j++)
          noise.push_back(clusters[i][j]);
      }
    }

    // Refine points treated as noise
    bool iterate = true;
    while (iterate && noise.size() > 0)
    {
      uint size_a = noise.size();
      vector<int> noise_tmp;
      for (uint n=0; n<noise.size(); n++)
      {
        int idx = -1;
        bool found = false;
        cv::KeyPoint p_n = mvKeys.at(noise[n]);
        for (uint i=0; i<clusters_.size(); i++)
        {
          for (uint j=0; j<clusters_[i].size(); j++)
          {
            cv::KeyPoint p_c = mvKeys.at(clusters_[i][j]);
            float dist = sqrt(pow((p_c.pt.x - p_n.pt.x),2)+pow((p_c.pt.y - p_n.pt.y),2));
            if(dist <= eps && dist != 0.0)
            {
              idx = i;
              found = true;
              break;
            }
          }
          if (found)
            break;
        }

        if (found && idx >= 0)
          clusters_[idx].push_back(noise[n]);          
        else
          noise_tmp.push_back(noise[n]);
      }

      if (noise_tmp.size() == 0 || noise_tmp.size() == size_a)
        iterate = false;

      noise = noise_tmp;
    }

    // If 1 cluster, add all keypoints
    if (clusters_.size() <= 1)
    {
      vector<int> cluster_tmp;
      for (uint i=0; i<mvKeys.size(); i++){
        if (mvpMapPoints[i]){
            cluster_tmp.push_back(int(i));
        }
      }
      clusters_.clear();
      clusters_.push_back(cluster_tmp);
    }
    
    // Compute the clusters centroids
    for (uint i=0; i<clusters_.size(); i++)
    {
      PointCloudXYZ::Ptr cluster_points(new PointCloudXYZ);
      for (uint j=0; j<clusters_[i].size(); j++)
      {
        int idx = clusters_[i][j];
        //if(!mvpMapPoints[idx]){
         //   cout<<"empty mappoint :"<<idx<<endl;
        //}

        cv::Point3f p_cv(mvpMapPoints[idx]->GetWorldPos());
        PointXYZ p(p_cv.x, p_cv.y, p_cv.z);
        cluster_points->push_back(p);
      }

      Eigen::Vector4f centroid;
      compute3DCentroid(*cluster_points, centroid);
      cluster_centroids_.push_back(centroid);
    }
}
  
vector<int> KeyFrame::regionQuery(vector<cv::KeyPoint> *keypoints, cv::KeyPoint *keypoint, float eps)
  {
    float dist;
    vector<int> ret_keys;
    for(uint i=0; i<keypoints->size(); i++)
    {
        MapPoint* itr=mvpMapPoints[i];
        if (itr != NULL){
            dist = sqrt(pow((keypoint->pt.x - keypoints->at(i).pt.x),2)+pow((keypoint->pt.y - keypoints->at(i).pt.y),2));
            if(dist <= eps && dist != 0.0)
            {
                ret_keys.push_back(i);
            }
        }
    }
    return ret_keys;
  }

} //namespace USLAM
