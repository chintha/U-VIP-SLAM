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

#include "FrameKTL.h"
#include "Converter.h"

#include <ros/ros.h>

namespace USLAM
{
long unsigned int FrameKTL::nNextId=0;
bool FrameKTL::mbInitialComputations=true;
float FrameKTL::cx, FrameKTL::cy, FrameKTL::fx, FrameKTL::fy;
int FrameKTL::mnMinX, FrameKTL::mnMinY, FrameKTL::mnMaxX, FrameKTL::mnMaxY;
float FrameKTL::mfGridElementWidthInv, FrameKTL::mfGridElementHeightInv;
//int FrameKTL::pyr_levels = 7;// KTL parameters best work 5 for aqualoc dataset
//cv::Size FrameKTL::win_size = cv::Size(25, 25);//KTL parameters

FrameKTL::FrameKTL()
{}

//Copy Constructor
FrameKTL::FrameKTL(const FrameKTL &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractor(frame.mpORBextractor), im(frame.im.clone()), mTimeStamp(frame.mTimeStamp),
     mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()), N(frame.N), mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn),
     mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec), mDescriptors(frame.mDescriptors.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier),
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     imgpyr(frame.imgpyr),mIMUdata(frame.mIMUdata),mHas_depth(frame.mHas_depth), mDepth(frame.mDepth),mDepth_time(frame.mDepth_time),
     mfLogScaleFactor(frame.mfLogScaleFactor),mPyr_Levels(frame.mPyr_Levels),mWin_Size(frame.mWin_Size)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)                            
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
    {
        
        mTcw = frame.mTcw.clone();
        SetPose(mTcw);
    }

    mNavState = frame.GetNavState();
    mMargCovInv = frame.mMargCovInv;
    mNavStatePrior = frame.mNavStatePrior;
    
    

}


FrameKTL::FrameKTL(cv::Mat &im_, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,std::vector<IMUData> &IMUdata,bool has_depth, double depth, double depth_time, int PyrLevel, cv::Size WinSize)
    :mpORBvocabulary(voc),mpORBextractor(extractor), im(im_),mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()),mIMUdata(IMUdata), mHas_depth(has_depth), mDepth(depth), mDepth_time(depth_time), mPyr_Levels(PyrLevel),mWin_Size(WinSize)
{
    
    //mIMUdata=IMUdata;
    cv::buildOpticalFlowPyramid(im, imgpyr, mWin_Size, mPyr_Levels); 

    // This is done for the first created FrameKTL
    if(mbInitialComputations)
    {
        ComputeImageBounds();// this is to mark the undistroted image- like remove the black areas and set max X and Y (rows and colums)

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);// this is like making the grid
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);

        mbInitialComputations=false;
    }

    N=0;
    //mvbOutlier.resize(1000,bool(false));
    mnId=nNextId++; // frame id current and next start from 1 
    mfLogScaleFactor = log(mfScaleFactor);

}
void FrameKTL::ComputeIMUPreIntSinceLastFrame(const FrameKTL* pLastF, IMUPreintegrator& IMUPreInt) const
{
    // Reset pre-integrator first
    IMUPreInt.reset();

    const std::vector<IMUData>& vIMUSInceLastFrame = mIMUdata;

    Vector3d bg = pLastF->GetNavState().Get_BiasGyr();
    Vector3d ba = pLastF->GetNavState().Get_BiasAcc();
    

    // remember to consider the gap between the last KF and the first IMU
    /*
    {
        const IMUData& imu = vIMUSInceLastFrame.front();
        double dt = imu.timestamp - pLastF->mTimeStamp;
        IMUPreInt.update(imu.wm - bg, imu.am - ba, dt);

        // Test log
        if(dt < 0)
        {
            cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this frame vs last imu time: "<<pLastF->mTimeStamp<<" vs "<<imu.timestamp<<endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }
    */
    // integrate each imu
    for(size_t i=0; i<vIMUSInceLastFrame.size(); i++)
    {
        const IMUData& imu = vIMUSInceLastFrame[i];
        double nextt;
        if(i==vIMUSInceLastFrame.size()-1)
            nextt = mTimeStamp;         // last IMU, next is this KeyFrame
        else
            nextt = vIMUSInceLastFrame[i+1].timestamp;  // regular condition, next is imu data

        // delta time
        double dt = nextt - imu.timestamp;
        // update pre-integrator
        if (dt>0){
            IMUPreInt.update(imu.wm - bg, imu.am - ba, dt);
        }
        // Test log
        if(dt < 0)
        {
            cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu.timestamp<<" vs "<<nextt<<endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }
}

void FrameKTL::SetInitialNavStateAndBias(const NavState& ns)
{
    mNavState = ns;
    // Set bias as bias+delta_bias, and reset the delta_bias term
    mNavState.Set_BiasGyr(ns.Get_BiasGyr()+ns.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(ns.Get_BiasAcc()+ns.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}
void FrameKTL::UpdatePoseFromNS(const cv::Mat &Tbc)
{
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Pbc = Tbc.rowRange(0,3).col(3).clone();

    cv::Mat Rwb = Converter::toCvMat(mNavState.Get_RotMatrix());
    cv::Mat Pwb = Converter::toCvMat(mNavState.Get_P());

    cv::Mat Rcw = (Rwb*Rbc).t();
    cv::Mat Pwc = Rwb*Pbc + Pwb;
    cv::Mat Pcw = -Rcw*Pwc;

    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    Pcw.copyTo(Tcw.rowRange(0,3).col(3));

    SetPose(Tcw);
    //mNavState.Set_BiasGyr(mNavState.Get_BiasGyr() +mNavState.Get_dBias_Gyr());
    //mNavState.Set_BiasAcc(mNavState.Get_BiasAcc() +mNavState.Get_dBias_Acc());
    //mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    //mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}
void FrameKTL::UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw)
{
    Converter::updateNS(mNavState,imupreint,gw);
}

const NavState& FrameKTL::GetNavState(void) const
{
    return mNavState;
}
void FrameKTL::SetNavStateBiasGyr(const Vector3d &bg)
{
    mNavState.Set_BiasGyr(bg);
}

void FrameKTL::SetNavStateBiasAcc(const Vector3d &ba)
{
    mNavState.Set_BiasAcc(ba);
}

void FrameKTL::SetNavState(const NavState& ns)
{
    mNavState = ns;
}

void FrameKTL::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void FrameKTL::SetN(int nts)
{
    N=nts;
    //mvbOutlier =  vector<bool>(N,false);
    mvbOutlier.resize(N,bool(false));
}

int FrameKTL::GetN()
{   
    if(N!=mvKeys.size()){
        SetN(mvKeys.size());
    }
    return N;
}

void FrameKTL::compute_descriptors(){

    //(*mpORBextractor)(im,cv::Mat(),mvKeys,mDescriptors); // mvKeys is a cv:: keypoint vector / mDescriptors is a cv::Mat // ORB descriptor, each row associated to a keypoint
    
  
     //Scale Levels Info
    mnScaleLevels = mpORBextractor->GetLevels();
    mfScaleFactor = mpORBextractor->GetScaleFactor();

    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<mnScaleLevels; i++)
    {
        mvScaleFactors[i]=mvScaleFactors[i-1]*mfScaleFactor;        
        mvLevelSigma2[i]=mvScaleFactors[i]*mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for(int i=0; i<mnScaleLevels; i++)
        mvInvLevelSigma2[i]=1/mvLevelSigma2[i];

    // Assign Features to Grid Cells
    int nReserve = 0.5*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);


    for(size_t i=0;i<mvKeysUn.size();i++)
    {
        cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }


   // mvbOutlier = vector<bool>(N,false);

}
    
void FrameKTL::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

cv::Mat FrameKTL::GetRotation()
{
    return mRcw.clone();
}

cv::Mat FrameKTL::GetRotationInverse()
{
    return mRwc.clone();
}

cv::Mat FrameKTL::GetTranslation()
{
    return mtcw.clone();
}

cv::Mat FrameKTL::GetCameraCenter()
{
    return mOw.clone();
}

bool FrameKTL::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float PcX = Pc.at<float>(0);
    const float PcY= Pc.at<float>(1);
    const float PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale level acording to the distance
    float ratio = dist/minDistance;

    const int nPredictedLevel = pMP->PredictScale(dist, this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    //pMP->mTrackProjXR = u - mbf * invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> FrameKTL::GetFeaturesInArea(const float &x, const float  &y, const float  &r, int minLevel, int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysUn.size());

    int nMinCellX = floor((x-mnMinX-r)*mfGridElementWidthInv);
    nMinCellX = max(0,nMinCellX);
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    int nMaxCellX = ceil((x-mnMinX+r)*mfGridElementWidthInv);
    nMaxCellX = min(FRAME_GRID_COLS-1,nMaxCellX);
    if(nMaxCellX<0)
        return vIndices;

    int nMinCellY = floor((y-mnMinY-r)*mfGridElementHeightInv);
    nMinCellY = max(0,nMinCellY);
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    int nMaxCellY = ceil((y-mnMinY+r)*mfGridElementHeightInv);
    nMaxCellY = min(FRAME_GRID_ROWS-1,nMaxCellY);
    if(nMaxCellY<0)
        return vIndices;

    bool bCheckLevels=true;
    bool bSameLevel=false;
    if(minLevel==-1 && maxLevel==-1)
        bCheckLevels=false;
    else
        if(minLevel==maxLevel)
            bSameLevel=true;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels && !bSameLevel)
                {
                    if(kpUn.octave<minLevel || kpUn.octave>maxLevel)
                        continue;
                }
                else if(bSameLevel)
                {
                    if(kpUn.octave!=minLevel)
                        continue;
                }

                if(abs(kpUn.pt.x-x)>r || abs(kpUn.pt.y-y)>r)
                    continue;

                vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;

}

bool FrameKTL::PosInGrid(cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void FrameKTL::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


void FrameKTL::ComputeImageBounds()
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=im.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=im.rows;
        mat.at<float>(3,0)=im.cols; mat.at<float>(3,1)=im.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        //cv::fisheye::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);// this needed for aqualoc harbor dataset
        mat=mat.reshape(1);

        mnMinX = min(floor(mat.at<float>(0,0)),floor(mat.at<float>(2,0)));
        mnMaxX = max(ceil(mat.at<float>(1,0)),ceil(mat.at<float>(3,0)));
        mnMinY = min(floor(mat.at<float>(0,1)),floor(mat.at<float>(1,1)));
        mnMaxY = max(ceil(mat.at<float>(2,1)),ceil(mat.at<float>(3,1)));

    }
    else
    {
        mnMinX = 0;
        mnMaxX = im.cols;
        mnMinY = 0;
        mnMaxY = im.rows;
    }
}

} //namespace USLAM
