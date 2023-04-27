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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<boost/thread.hpp>

#include "src/IMU/configparam.h"
#include "src/IMU/g2otypes.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_with_hessian.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_cholmod.h"

namespace USLAM
{

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

void Optimizer::GlobalBundleAdjustmentNavState(Map* pMap, const cv::Mat& gw, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = Converter::toVector3d(gw);

    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        // PVR
        g2o::VertexNavStatePVR * vNSPVR = new g2o::VertexNavStatePVR();
        vNSPVR->setEstimate(pKF->GetNavState());
        vNSPVR->setId(pKF->mnId*2);
        vNSPVR->setFixed(pKF->mnId==0);
        optimizer.addVertex(vNSPVR);
        // Bias
        g2o::VertexNavStateBias * vNSBias = new g2o::VertexNavStateBias();
        vNSBias->setEstimate(pKF->GetNavState());
        vNSBias->setId(pKF->mnId*2+1);
        vNSBias->setFixed(pKF->mnId==0);
        optimizer.addVertex(vNSBias);

        if(pKF->mnId*2+1>maxKFid)
            maxKFid=pKF->mnId*2+1;
    }

    // Add NavState PVR/Bias edges
    const float thHuberNavStatePVR = sqrt(21.666);
    const float thHuberNavStateBias = sqrt(16.812);
    // Inverse covariance of bias random walk
    Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF1 = vpKFs[i];
        if(pKF1->isBad())
        {
            cout<<"pKF is bad in gBA, id "<<pKF1->mnId<<endl;   //Debug log
            continue;
        }

        KeyFrame* pKF0 = pKF1->GetPrevKeyFrame();
        if(!pKF0)
        {
            if(pKF1->mnId!=0) cerr<<"Previous KeyFrame is NULL?"<<endl;  //Test log
            continue;
        }

        // PVR edge
        {
            g2o::EdgeNavStatePVR * epvr = new g2o::EdgeNavStatePVR();
            epvr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId)));
            epvr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId)));
            epvr->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
            epvr->setMeasurement(pKF1->GetIMUPreInt());

            Matrix9d InvCovPVR = pKF1->GetIMUPreInt().getCovPVPhi().inverse();
            epvr->setInformation(InvCovPVR);
            epvr->SetParams(GravityVec);

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                epvr->setRobustKernel(rk);
                rk->setDelta(thHuberNavStatePVR);
            }

            optimizer.addEdge(epvr);
        }
        // Bias edge
        {
            g2o::EdgeNavStateBias * ebias = new g2o::EdgeNavStateBias();
            ebias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
            ebias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId+1)));
            ebias->setMeasurement(pKF1->GetIMUPreInt());

            ebias->setInformation(InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime());

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                ebias->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateBias);
            }

            optimizer.addEdge(ebias);
        }

    }

    const float thHuber2D = sqrt(5.99);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || 2*pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            //if(pKF->mvuRight[mit->second]<0) // this is the original one(relate to stereo VIO)
            if(true)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeNavStatePVRPointXYZ* e = new g2o::EdgeNavStatePVRPointXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->GetInvSigma2(kpUn.octave);
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->SetParams(pKF->fx,pKF->fy,pKF->cx,pKF->cy,Rbc,Pbc);

                optimizer.addEdge(e);
            }
            else
            {
                cerr<<"Stereo not supported"<<endl;
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        //g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        //g2o::SE3Quat SE3quat = vSE3->estimate();
        g2o::VertexNavStatePVR* vNSPVR = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(2*pKF->mnId));
        g2o::VertexNavStateBias* vNSBias = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(2*pKF->mnId+1));
        const NavState& nspvr = vNSPVR->estimate();
        const NavState& nsbias = vNSBias->estimate();
        NavState ns_recov = nspvr;
        ns_recov.Set_DeltaBiasGyr(nsbias.Get_dBias_Gyr());
        ns_recov.Set_DeltaBiasAcc(nsbias.Get_dBias_Acc());

        if(nLoopKF==0)
        {
            //pKF->SetPose(Converter::toCvMat(SE3quat));
            pKF->SetNavState(ns_recov);
            pKF->UpdatePoseFromNS(ConfigParam::GetMatTbc());
        }
        else
        {
            pKF->mNavStateGBA = ns_recov;

            pKF->mTcwGBA.create(4,4,CV_32F);
            //Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            cv::Mat Twb_ = cv::Mat::eye(4,4,CV_32F);
            Converter::toCvMat(pKF->mNavStateGBA.Get_RotMatrix()).copyTo(Twb_.rowRange(0,3).colRange(0,3));
            Converter::toCvMat(pKF->mNavStateGBA.Get_P()).copyTo(Twb_.rowRange(0,3).col(3));
            cv::Mat Twc_ = Twb_*ConfigParam::GetMatTbc();
            pKF->mTcwGBA = Converter::toCvMatInverse(Twc_);

            pKF->mnBAGlobalForKF = nLoopKF;
        }

//        //Debug log
//        cv::Mat tTwb1 = pKF->GetPoseInverse()*ConfigParam::GetMatT_cb();
//        if((Converter::toVector3d(tTwb1.rowRange(0,3).col(3))-pKF->GetNavState().Get_P()).norm()>1e-6)
//            cout<<"in gBA, Twc*Tcb != NavState for GBA KFs, id "<<pKF->mnId<<": "<<tTwb1.rowRange(0,3).col(3).t()<<"/"<<pKF->GetNavState().Get_P().transpose()<<endl;

    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

int Optimizer::PoseOptimization(FrameKTL *pFrame, FrameKTL* pLastFrame, const IMUPreintegrator& imupreint,
 const cv::Mat& gw,const cv::Mat& grot, const bool& bComputeMarg,double ini_depth,double depth_cov)
{
    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = Converter::toVector3d(gw);

    Matrix3d GravityRot = Converter::toMatrix3d(grot);

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    //linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    const int FramePVRId = 0;
    const int FrameBiasId = 1;
    const int LastFramePVRId = 2;
    const int LastFrameBiasId = 3;

    // Set FrameKTL vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSFPVR = new g2o::VertexNavStatePVR();
    {
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFPVR->setId(FramePVRId);
        vNSFPVR->setFixed(false);
        optimizer.addVertex(vNSFPVR);
    }
    g2o::VertexNavStateBias * vNSFBias = new g2o::VertexNavStateBias();
    {
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFBias->setId(FrameBiasId);
        vNSFBias->setFixed(false);
        optimizer.addVertex(vNSFBias);
    }

    // Set LastFrame vertex
    g2o::VertexNavStatePVR * vNSFPVRlast = new g2o::VertexNavStatePVR();
    {
        vNSFPVRlast->setEstimate(pLastFrame->GetNavState());
        vNSFPVRlast->setId(LastFramePVRId);
        vNSFPVRlast->setFixed(false);
        optimizer.addVertex(vNSFPVRlast);
    }
    g2o::VertexNavStateBias * vNSFBiaslast = new g2o::VertexNavStateBias();
    {
        vNSFBiaslast->setEstimate(pLastFrame->GetNavState());
        vNSFBiaslast->setId(LastFrameBiasId);
        vNSFBiaslast->setFixed(false);
        optimizer.addVertex(vNSFBiaslast);
    }

    // Set prior edge for Last FrameKTL, from mMargCovInv
    g2o::EdgeNavStatePriorPVRBias* eNSPrior = new g2o::EdgeNavStatePriorPVRBias();
    {
        eNSPrior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
        eNSPrior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSPrior->setMeasurement(pLastFrame->mNavStatePrior);

        eNSPrior->setInformation(pLastFrame->mMargCovInv);

        const float thHuberNavState = sqrt(30.5779);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPrior->setRobustKernel(rk);
        rk->setDelta(thHuberNavState);

        optimizer.addEdge(eNSPrior);
    }

    // Set PVR edge between LastFrame-FrameKTL
    g2o::EdgeNavStatePVR* eNSPVR = new g2o::EdgeNavStatePVR();
    {
        eNSPVR->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
        eNSPVR->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
        eNSPVR->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSPVR->setMeasurement(imupreint);

        Matrix9d InvCovPVR = imupreint.getCovPVPhi().inverse() ;
        eNSPVR->setInformation(InvCovPVR);

        eNSPVR->SetParams(GravityVec);

        const float thHuberNavStatePVR = sqrt(21.666);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPVR->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);

        optimizer.addEdge(eNSPVR);
    }
    // Set Bias edge between LastFrame-FrameKTL
    g2o::EdgeNavStateBias* eNSBias = new g2o::EdgeNavStateBias();
    {
        eNSBias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSBias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FrameBiasId)));
        eNSBias->setMeasurement(imupreint);

        Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
        InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
        InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
        eNSBias->setInformation(InvCovBgaRW/imupreint.getDeltaTime());

        const float thHuberNavStateBias = sqrt(16.812);//16.812
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSBias->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateBias);

        optimizer.addEdge(eNSBias);
    }
    // Set Depth edge to current frame 
    if (pFrame->mHas_depth){/*
        g2o::EdgeNavStateDepth* eNSDepth = new g2o::EdgeNavStateDepth();
            {
                eNSDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                eNSDepth->setMeasurement(pFrame->mDepth- ini_depth);

                Matrix<double,1,1> InvCovDepth;
                InvCovDepth(0,0)= 1/(depth_cov*depth_cov);
                
                eNSDepth->setInformation(InvCovDepth);
                //eNSDepth->SetParams(GravityRot);

                const float thHuberNavStateDepth = sqrt(16.812);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                eNSDepth->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateDepth);

                optimizer.addEdge(eNSDepth);
            }*/
        g2o::EdgeNavStateDepthProjected* eNSDepth2 = new g2o::EdgeNavStateDepthProjected();
        {
            
            eNSDepth2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
            eNSDepth2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
            eNSDepth2->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
            double shi = (pFrame->mTimeStamp - pLastFrame->mTimeStamp)/(pFrame->mDepth_time - pLastFrame->mTimeStamp);
            eNSDepth2->SetParams(shi,imupreint);
        
            eNSDepth2->setMeasurement(pFrame->mDepth- ini_depth);

            Matrix9d InvCovPVR = imupreint.getCovPVPhi();
            Matrix3d CovPos = InvCovPVR.block(0,0,2,2);
            //eNSDepth->setInformation(InvCovPVR);
            Vector3d e3 = Eigen::Vector3d::Zero();
            e3(2)=1.0;
            Matrix3d Rwb = pLastFrame->mNavState.Get_RotMatrix();
            Matrix<double,1,1> cov3 = e3.transpose()*Rwb.transpose()*CovPos*Rwb*e3;
            double cov1 = (shi*shi*depth_cov*depth_cov) + (double)cov3(0,0);
            Matrix<double,1,1> InvCovDepth;
            InvCovDepth(0,0)= 1/(cov1*cov1);
            
            eNSDepth2->setInformation(InvCovDepth);
            //eNSDepth->SetParams(GravityRot);

            const float thHuberNavStateDepth = sqrt(16.812);
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            eNSDepth2->setRobustKernel(rk);
            rk->setDelta(thHuberNavStateDepth);

            optimizer.addEdge(eNSDepth2);
        }   

    }


    if (pLastFrame->mHas_depth){/*
        g2o::EdgeNavStateDepth* eNSDepth = new g2o::EdgeNavStateDepth();
            {
                eNSDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
                eNSDepth->setMeasurement(pLastFrame->mDepth- ini_depth);

                Matrix<double,1,1> InvCovDepth;
                InvCovDepth(0,0)= 1/(depth_cov*depth_cov);
                
                eNSDepth->setInformation(InvCovDepth);
                //eNSDepth->SetParams(GravityRot);

                const float thHuberNavStateDepth = sqrt(16.812);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                eNSDepth->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateDepth);

                optimizer.addEdge(eNSDepth);
            }
        */   
       

    }

    // Set MapPoint vertices
    const int Ncur = pFrame->GetN();
    const int Nlast = pLastFrame->GetN();

    vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(Ncur);
    vnIndexEdgeMono.reserve(Ncur);

    vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMonoLast;
    vector<size_t> vnIndexEdgeMonoLast;
    vpEdgesMonoLast.reserve(Nlast);
    vnIndexEdgeMonoLast.reserve(Nlast);


    const float deltaMono = sqrt(5.991);

    {
     boost::mutex::scoped_lock lock(MapPoint::mGlobalMutex);

    for(int i=0; i<Ncur; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
    
        if(pMP)
        {
            // Monocular observation
            //if(pFrame->mvuRight[i]<0)
            if(true)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->SetParams(pFrame->fx,pFrame->fy,pFrame->cx,pFrame->cy,Rbc,Pbc,Converter::toVector3d(pMP->GetWorldPos()));

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                cerr<<"stereo shouldn't in poseoptimization"<<endl;
            }
        }    

    }
    
    // Add Point-Pose edges for last frame
    for(int i=0; i<Nlast; i++)
    {
        MapPoint* pMP = pLastFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            //if(pLastFrame->mvuRight[i]<0)
            if(true)
            {
                //nInitialCorrespondences++;
                pLastFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pLastFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
                e->setMeasurement(obs);
                const float invSigma2 = pLastFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->SetParams(pLastFrame->fx,pLastFrame->fy,pLastFrame->cx,pLastFrame->cy,Rbc,Pbc,Converter::toVector3d(pMP->GetWorldPos()));

                optimizer.addEdge(e);

                vpEdgesMonoLast.push_back(e);
                vnIndexEdgeMonoLast.push_back(i);
            }
            else  // Stereo observation
            {
                cerr<<"stereo shouldn't in poseoptimization"<<endl;
            }
        }
    }
    }


    if(nInitialCorrespondences<4)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    //const float chi2Mono[4]={9.210,7.378,5.991,5.991};
    //const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

    //    //Debug log
    //    cout<<"total Points: "<<vpEdgesMono.size()<<endl;

    int nBad=0;
    int nBadLast=0;
    for(size_t it=0; it<4; it++)
    {
        // Reset estimates
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFPVRlast->setEstimate(pLastFrame->GetNavState());
        vNSFBiaslast->setEstimate(pLastFrame->GetNavState());

        //optimizer.setVerbose(true);
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nBadLast=0;
        for(size_t i=0, iend=vpEdgesMonoLast.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMonoLast[i];

            const size_t idx = vnIndexEdgeMonoLast[i];

            if(pLastFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pLastFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadLast++;
            }
            else
            {
                pLastFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        //        //Debug log
        //        cout<<nBad<<" bad Points in iter "<<it<<", rest points: "<<optimizer.edges().size()<<endl;
        //        cout<<nBadLast<<" bad Points of last FrameKTL in iter "<<it<<endl;
        //        cout<<"NavState edge chi2: "<<eNS->chi2()<<endl;

        //if(vpEdgesMono.size() - nBad < 10)
        if(optimizer.edges().size()<10){
            break;

            }
    }

    // Debug log
    //if(eNSPVR->chi2()>21.666) cout<<"F-F PVR edge chi2:"<<eNSPVR->chi2()<<endl;
    //if(eNSBias->chi2()>16.812) cout<<"F-F Bias edge chi2:"<<eNSBias->chi2()<<endl;
    //if(eNSPrior->chi2()>30.5779) cout<<"F-F Prior edge chi2:"<<eNSPrior->chi2()<<endl;

    // Recover optimized pose and return number of inliers
    g2o::VertexNavStatePVR* vNSPVR_recov = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(FramePVRId));
    const NavState& nsPVR_recov = vNSPVR_recov->estimate();
    g2o::VertexNavStateBias* vNSBias_recov = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(FrameBiasId));
    const NavState& nsBias_recov = vNSBias_recov->estimate();
    NavState ns_recov = nsPVR_recov;
    ns_recov.Set_DeltaBiasGyr(nsBias_recov.Get_dBias_Gyr());
    ns_recov.Set_DeltaBiasAcc(nsBias_recov.Get_dBias_Acc());
    pFrame->SetNavState(ns_recov);
    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
    if(bComputeMarg)
    {
        std::vector<g2o::OptimizableGraph::Vertex*> margVerteces;
        margVerteces.push_back(optimizer.vertex(FramePVRId));
        margVerteces.push_back(optimizer.vertex(FrameBiasId));

        g2o::SparseBlockMatrixXd spinv;
        optimizer.computeMarginals(spinv, margVerteces);
        // spinv include 2 blocks, 9x9-(0,0) for PVR, 6x6-(1,1) for Bias
        Matrix<double,15,15> margCov = Matrix<double,15,15>::Zero();
        margCov.topLeftCorner(9,9) = spinv.block(0,0)->eval();
        margCov.topRightCorner(9,6) = spinv.block(0,1)->eval();
        margCov.bottomLeftCorner(6,9) = spinv.block(1,0)->eval();
        margCov.bottomRightCorner(6,6) = spinv.block(1,1)->eval();
        pFrame->mMargCovInv = margCov.inverse();
        pFrame->mNavStatePrior = ns_recov;

        //Debug log
        //cout<<"inv MargCov 2: "<<endl<<pFrame->mMargCovInv<<endl;
    }

    //Test log
    if( (nsPVR_recov.Get_BiasGyr()-nsBias_recov.Get_BiasGyr()).norm()>1e-6 || (nsPVR_recov.Get_BiasAcc()-nsBias_recov.Get_BiasAcc()).norm()>1e-6 )
    {
        std::cerr<<"1 recovered bias gyr not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasGyr().transpose()<<" / "<<nsBias_recov.Get_BiasGyr().transpose()<<std::endl;
        std::cerr<<"1 recovered bias acc not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasAcc().transpose()<<" / "<<nsBias_recov.Get_BiasAcc().transpose()<<std::endl;
    }
    if( (ns_recov.Get_dBias_Gyr()-nsBias_recov.Get_dBias_Gyr()).norm()>1e-6 || (ns_recov.Get_dBias_Acc()-nsBias_recov.Get_dBias_Acc()).norm()>1e-6 )
    {
        std::cerr<<"1 recovered delta bias gyr not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Gyr().transpose()<<" / "<<nsBias_recov.Get_dBias_Gyr().transpose()<<std::endl;
        std::cerr<<"1 recovered delta bias acc not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Acc().transpose()<<" / "<<nsBias_recov.Get_dBias_Acc().transpose()<<std::endl;
    }

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseOptimization(FrameKTL *pFrame, KeyFrame* pLastKF, const IMUPreintegrator& imupreint,const cv::Mat& gw, 
const cv::Mat& grot, const bool& bComputeMarg,double ini_depth, double depth_cov)
{
    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = Converter::toVector3d(gw);
    //Matrix3d GravityRot = Converter::toMatrix3d(grot);  

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    //linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    const int FramePVRId = 0;
    const int FrameBiasId = 1;
    const int LastKFPVRId = 2;
    const int LastKFBiasId = 3;

    // Set FrameKTL vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSFPVR = new g2o::VertexNavStatePVR();
    {
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFPVR->setId(FramePVRId);
        vNSFPVR->setFixed(false);
        optimizer.addVertex(vNSFPVR);
    }
    g2o::VertexNavStateBias * vNSFBias = new g2o::VertexNavStateBias();
    {
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFBias->setId(FrameBiasId);
        vNSFBias->setFixed(false);
        optimizer.addVertex(vNSFBias);
    }

    // Set KeyFrame vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSKFPVR = new g2o::VertexNavStatePVR();
    {
        vNSKFPVR->setEstimate(pLastKF->GetNavState());
        vNSKFPVR->setId(LastKFPVRId);
        vNSKFPVR->setFixed(true);
        optimizer.addVertex(vNSKFPVR);
    }
    g2o::VertexNavStateBias * vNSKFBias = new g2o::VertexNavStateBias();
    {
        vNSKFBias->setEstimate(pLastKF->GetNavState());
        vNSKFBias->setId(LastKFBiasId);
        vNSKFBias->setFixed(true);
        optimizer.addVertex(vNSKFBias);
    }

    // Set PVR edge between LastKF-FrameKTL
    g2o::EdgeNavStatePVR* eNSPVR = new g2o::EdgeNavStatePVR();
    {
        eNSPVR->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFPVRId)));
        eNSPVR->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
        eNSPVR->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFBiasId)));
        eNSPVR->setMeasurement(imupreint);

        Matrix9d InvCovPVR = imupreint.getCovPVPhi().inverse() ;
        eNSPVR->setInformation(InvCovPVR);

        eNSPVR->SetParams(GravityVec);

        const float thHuberNavStatePVR = sqrt(21.666);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPVR->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);

        optimizer.addEdge(eNSPVR);
    }
    // Set Bias edge between LastKF-FrameKTL
    g2o::EdgeNavStateBias* eNSBias = new g2o::EdgeNavStateBias();
    {
        eNSBias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFBiasId)));
        eNSBias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FrameBiasId)));
        eNSBias->setMeasurement(imupreint);

        Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
        InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
        InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
        eNSBias->setInformation(InvCovBgaRW/imupreint.getDeltaTime());

        const float thHuberNavStateBias = sqrt(16.812);//16.812
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSBias->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateBias);

        optimizer.addEdge(eNSBias);
    }

     if (pFrame->mHas_depth){/*
        g2o::EdgeNavStateDepth* eNSDepth = new g2o::EdgeNavStateDepth();
            {
                eNSDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                eNSDepth->setMeasurement(pFrame->mDepth- ini_depth);

                Matrix<double,1,1> InvCovDepth;
                InvCovDepth(0,0)= 1/(depth_cov*depth_cov);
                
                eNSDepth->setInformation(InvCovDepth);
                //eNSDepth->SetParams(GravityRot);

                const float thHuberNavStateDepth = sqrt(16.812);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                eNSDepth->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateDepth);

                optimizer.addEdge(eNSDepth);
            }*/
        
        g2o::EdgeNavStateDepthProjected* eNSDepth2 = new g2o::EdgeNavStateDepthProjected();
            {
                cout<<"calling Depth"<<endl;
                eNSDepth2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFPVRId)));
                eNSDepth2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                eNSDepth2->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFBiasId)));
                double shi = (pFrame->mTimeStamp - pLastKF->mTimeStamp)/(pFrame->mDepth_time - pLastKF->mTimeStamp);
                eNSDepth2->SetParams(shi,imupreint);
            
                eNSDepth2->setMeasurement(pFrame->mDepth- ini_depth);

                Matrix9d InvCovPVR = imupreint.getCovPVPhi();
                Matrix3d CovPos = InvCovPVR.block(0,0,2,2);
                //eNSDepth->setInformation(InvCovPVR);
                Vector3d e3 = Eigen::Vector3d::Zero();
                e3(2)=1.0;
                Matrix3d Rwb = pLastKF->mNavState.Get_RotMatrix();
                Matrix<double,1,1> cov3 = e3.transpose()*Rwb.transpose()*CovPos*Rwb*e3;
                double cov1 = (shi*shi*depth_cov*depth_cov) + (double)cov3(0,0);
                Matrix<double,1,1> InvCovDepth;
                InvCovDepth(0,0)= 1/(cov1*cov1);
                
                eNSDepth2->setInformation(InvCovDepth);
                //eNSDepth->SetParams(GravityRot);

                const float thHuberNavStateDepth = sqrt(16.812);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                eNSDepth2->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateDepth);

                optimizer.addEdge(eNSDepth2);
            }
        
    }

    // Set MapPoint vertices
    const int N = pFrame->GetN();

    vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    const float deltaMono = sqrt(5.991);

    {
     boost::mutex::scoped_lock lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            //if(pFrame->mvuRight[i]<0)
            if(true)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->SetParams(pFrame->fx,pFrame->fy,pFrame->cx,pFrame->cy,Rbc,Pbc,Converter::toVector3d(pMP->GetWorldPos()));

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                cerr<<"stereo shouldn't in poseoptimization"<<endl;
            }
        }

    }
    }
    //cout<<endl;


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    //const float chi2Mono[4]={9.210,7.378,5.991,5.991};
    
    //const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

    //    //Debug log
    //    cout<<"total Points: "<<vpEdgesMono.size()<<endl;

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        // Reset estimate for vertex
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFBias->setEstimate(pFrame->GetNavState());

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        //        //Debug log
        //        cout<<nBad<<" bad Points in iter "<<it<<", rest points: "<<optimizer.edges().size()<<endl;

        if(optimizer.edges().size()<10){
            break;
        }
    }

    // Debug log
    //if(eNSPVR->chi2()>21.666) cout<<"KF-F PVR edge chi2:"<<eNSPVR->chi2()<<endl;
    //if(eNSBias->chi2()>16.812) cout<<"KF-F Bias edge chi2:"<<eNSBias->chi2()<<endl;

    // Recover optimized pose and return number of inliers
    g2o::VertexNavStatePVR* vNSPVR_recov = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(FramePVRId));
    const NavState& nsPVR_recov = vNSPVR_recov->estimate();
    g2o::VertexNavStateBias* vNSBias_recov = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(FrameBiasId));
    const NavState& nsBias_recov = vNSBias_recov->estimate();
    NavState ns_recov = nsPVR_recov;
    ns_recov.Set_DeltaBiasGyr(nsBias_recov.Get_dBias_Gyr());
    ns_recov.Set_DeltaBiasAcc(nsBias_recov.Get_dBias_Acc());
    pFrame->SetNavState(ns_recov);
    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
    if(bComputeMarg)
    {
        std::vector<g2o::OptimizableGraph::Vertex*> margVerteces;
        margVerteces.push_back(optimizer.vertex(FramePVRId));
        margVerteces.push_back(optimizer.vertex(FrameBiasId));

        //TODO: how to get the joint marginalized covariance of PVR&Bias
        g2o::SparseBlockMatrixXd spinv;
        optimizer.computeMarginals(spinv, margVerteces);
        // spinv include 2 blocks, 9x9-(0,0) for PVR, 6x6-(1,1) for Bias
        Matrix<double,15,15> margCovInv = Matrix<double,15,15>::Zero();
        margCovInv.topLeftCorner(9,9) = spinv.block(0,0)->inverse();
        margCovInv.bottomRightCorner(6,6) = spinv.block(1,1)->inverse();
        pFrame->mMargCovInv = margCovInv;
        pFrame->mNavStatePrior = ns_recov;
    }

    //Test log
    if( (nsPVR_recov.Get_BiasGyr()-nsBias_recov.Get_BiasGyr()).norm()>1e-6 || (nsPVR_recov.Get_BiasAcc()-nsBias_recov.Get_BiasAcc()).norm()>1e-6 )
    {
        std::cerr<<"recovered bias gyr not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasGyr().transpose()<<" / "<<nsBias_recov.Get_BiasGyr().transpose()<<std::endl;
        std::cerr<<"recovered bias acc not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasAcc().transpose()<<" / "<<nsBias_recov.Get_BiasAcc().transpose()<<std::endl;
    }
    if( (ns_recov.Get_dBias_Gyr()-nsBias_recov.Get_dBias_Gyr()).norm()>1e-6 || (ns_recov.Get_dBias_Acc()-nsBias_recov.Get_dBias_Acc()).norm()>1e-6 )
    {
        std::cerr<<"recovered delta bias gyr not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Gyr().transpose()<<" / "<<nsBias_recov.Get_dBias_Gyr().transpose()<<std::endl;
        std::cerr<<"recovered delta bias acc not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Acc().transpose()<<" / "<<nsBias_recov.Get_dBias_Acc().transpose()<<std::endl;
    }

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustmentNavState(KeyFrame *pCurKF, const std::list<KeyFrame*> &lLocalKeyFrames, bool* pbStopFlag, Map* pMap, 
const cv::Mat& gw,const cv::Mat& grot,LocalMapping* pLM,double ini_depth,double depth_cov,int fixedKF_id)
{
    // Check current KeyFrame in local window
    if(pCurKF != lLocalKeyFrames.back())
        cerr<<"pCurKF != lLocalKeyFrames.back. check"<<endl;

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = Converter::toVector3d(gw);
    Matrix3d GravityRot = Converter::toMatrix3d(grot);

    // All KeyFrames in Local window are optimized
    for(list<KeyFrame*>::const_iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        pKFi->mnBALocalForKF = pCurKF->mnId;
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::const_iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pCurKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pCurKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    // Add the KeyFrame before local window.
    KeyFrame* pKFPrevLocal = lLocalKeyFrames.front()->GetPrevKeyFrame();
    unsigned int minKFID=pKFPrevLocal->mnId;
    unsigned int limitID=0;
    if(pKFPrevLocal)
    {
        // Test log
        if(pKFPrevLocal->isBad()) cerr<<"KeyFrame before local window is bad?"<<endl;
        if(pKFPrevLocal->mnBAFixedForKF==pCurKF->mnId) cerr<<"KeyFrame before local, has been added to lFixedKF?"<<endl;
        if(pKFPrevLocal->mnBALocalForKF==pCurKF->mnId) cerr<<"KeyFrame before local, has been added to lLocalKF?"<<endl;
        //limitID = pKFPrevLocal->mnId;
        pKFPrevLocal->mnBAFixedForKF = pCurKF->mnId;
        if(!pKFPrevLocal->isBad())
            lFixedCameras.push_back(pKFPrevLocal);
        else
            cerr<<"pKFPrevLocal is Bad?"<<endl;
    }
    // Test log
    else {cerr<<"pKFPrevLocal is NULL?"<<endl;}
    // Covisible KeyFrames
    
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pCurKF->mnId && pKFi->mnBAFixedForKF!=pCurKF->mnId)
            {
                pKFi->mnBAFixedForKF=pCurKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int maxKFid = 0;
    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::const_iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        int idKF = pKFi->mnId*2;
        // Vertex of PVR
        {
            g2o::VertexNavStatePVR * vNSPVR = new g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(pKFi->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(pKFi->mnId == fixedKF_id);
            optimizer.addVertex(vNSPVR);
        }
        // Vertex of Bias
        {
            g2o::VertexNavStateBias * vNSBias = new g2o::VertexNavStateBias();
            vNSBias->setEstimate(pKFi->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(pKFi->mnId == fixedKF_id);
            optimizer.addVertex(vNSBias);
        }

        if(idKF+1>maxKFid)
            maxKFid=idKF+1;
        // Test log
        if(pKFi->mnId == 0) cerr<<"pKFi->mnId == 0, shouldn't in LocalBA of NavState"<<endl;
    
    }

    // Set Fixed KeyFrame vertices. Including the pKFPrevLocal.
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        int idKF = pKFi->mnId*2;
        // For common fixed KeyFrames, only add PVR vertex
        {
            g2o::VertexNavStatePVR * vNSPVR = new g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(pKFi->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(true);
            optimizer.addVertex(vNSPVR);
        }
        // For Local-Window-Previous KeyFrame, add Bias vertex
        if(pKFi == pKFPrevLocal)
        {
            g2o::VertexNavStateBias * vNSBias = new g2o::VertexNavStateBias();
            vNSBias->setEstimate(pKFi->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(true);
            optimizer.addVertex(vNSBias);
        }

        if(idKF+1>maxKFid)
            maxKFid=idKF+1;
    }

    // Edges between KeyFrames in Local Window
    // and
    // Edges between 1st KeyFrame of Local Window and its previous (fixed)KeyFrame - pKFPrevLocal
    vector<g2o::EdgeNavStatePVR*> vpEdgesNavStatePVR;
    vector<g2o::EdgeNavStateBias*> vpEdgesNavStateBias;
    // Use chi2inv() in MATLAB to compute the value corresponding to 0.95/0.99 prob. w.r.t 15DOF: 24.9958/30.5779
    // 12.592/16.812 for 0.95/0.99 6DoF
    // 16.919/21.666 for 0.95/0.99 9DoF
    //const float thHuberNavState = sqrt(30.5779);
    const float thHuberNavStatePVR = sqrt(21.666);
    const float thHuberNavStateBias = sqrt(16.812);//16.812
    // Inverse covariance of bias random walkInvCovPVR
    Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE

    for(list<KeyFrame*>::const_iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF1 = *lit;                      // Current KF, store the IMU pre-integration between previous-current
        KeyFrame* pKF0 = pKF1->GetPrevKeyFrame();   // Previous KF

        // PVR edge
        {
            g2o::EdgeNavStatePVR * epvr = new g2o::EdgeNavStatePVR();
            epvr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId)));
            epvr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId)));
            epvr->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
            epvr->setMeasurement(pKF1->GetIMUPreInt());

            Matrix9d InvCovPVR = pKF1->GetIMUPreInt().getCovPVPhi().inverse();
            epvr->setInformation(InvCovPVR);
            epvr->SetParams(GravityVec);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            epvr->setRobustKernel(rk);
            rk->setDelta(thHuberNavStatePVR);

            optimizer.addEdge(epvr);
            vpEdgesNavStatePVR.push_back(epvr);
        }
        // Bias edge
        {
            g2o::EdgeNavStateBias * ebias = new g2o::EdgeNavStateBias();
            ebias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
            ebias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId+1)));
            ebias->setMeasurement(pKF1->GetIMUPreInt());

            ebias->setInformation(InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ebias->setRobustKernel(rk);
            rk->setDelta(thHuberNavStateBias);

            optimizer.addEdge(ebias);
            vpEdgesNavStateBias.push_back(ebias);
        }
        
        // Depth edge
        if(pKF1->mHas_depth)
            {   
                
                /*
                g2o::EdgeNavStateDepth* eDepth = new g2o::EdgeNavStateDepth();
                eDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId)));
                eDepth->setMeasurement(pKF1->mDepth- ini_depth);
                Matrix<double,1,1> InvCovDepth;
                InvCovDepth(0,0)= 1/(depth_cov*depth_cov);
                
                eDepth->setInformation(InvCovDepth);
                //eDepth->SetParams(GravityRot);

                const float thHuberNavStateDepth = sqrt(5.66);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                eDepth->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateDepth);

                optimizer.addEdge(eDepth);
                */
                /*
                g2o::EdgeNavStateDepthProjected* eNSDepth2 = new g2o::EdgeNavStateDepthProjected();
                {
                    
                    eNSDepth2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId)));
                    eNSDepth2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId)));
                    eNSDepth2->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
                    double shi = (pKF1->mTimeStamp - pKF0->mTimeStamp)/(pKF1->mDepth_time - pKF0->mTimeStamp);
                    const IMUPreintegrator& imupreint  = pKF1->GetIMUPreInt();
                    eNSDepth2->SetParams(shi,imupreint);
                
                    eNSDepth2->setMeasurement(pKF1->mDepth- ini_depth);

                    Matrix9d InvCovPVR = imupreint.getCovPVPhi();
                    Matrix3d CovPos = InvCovPVR.block(0,0,2,2);
                    //eNSDepth->setInformation(InvCovPVR);
                    Vector3d e3 = Eigen::Vector3d::Zero();
                    e3(2)=1.0;
                    Matrix3d Rwb = pKF0->mNavState.Get_RotMatrix();
                    Matrix<double,1,1> cov3 = e3.transpose()*Rwb.transpose()*CovPos*Rwb*e3;
                    double cov1 = (shi*shi*depth_cov*depth_cov) + (double)cov3(0,0);
                    Matrix<double,1,1> InvCovDepth;
                    InvCovDepth(0,0)= 1/(cov1*cov1);
                    
                    eNSDepth2->setInformation(InvCovDepth);
                    //eNSDepth->SetParams(GravityRot);

                    const float thHuberNavStateDepth = sqrt(16.812);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    eNSDepth2->setRobustKernel(rk);
                    rk->setDelta(thHuberNavStateDepth);

                    optimizer.addEdge(eNSDepth2);
                } */  
            }
            
        if(!(pKF1->Depth_Vec.empty()))
        {
            for(int i=0;i<pKF1->Depth_Vec.size();i++)
            {   
                double depth_measurment = pKF1->Depth_Vec[i];
                double depth_time = pKF1->Depth_Vec_Time[i];
                //std::string eNSDepth = "eNSDepthP" + std::to_string(i);
                g2o::EdgeNavStateDepthProjected* eNSDepth = new g2o::EdgeNavStateDepthProjected();
                if ((pKF1->mTimeStamp - pKF1->Depth_Vec_Time[i]) < (pKF1->Depth_Vec_Time[i] - pKF0->mTimeStamp))
                {
                    std::cout<<"Running Depth Forward......................."<<std::endl<<std::endl;
                    eNSDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->mnId)));
                    eNSDepth->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId)));
                    eNSDepth->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId+1)));
                    double shi = (pKF1->mTimeStamp - pKF0->mTimeStamp)/(depth_time - pKF0->mTimeStamp);
                    const IMUPreintegrator& imupreint  = pKF1->GetIMUPreInt();
                    eNSDepth->SetParams(shi,imupreint);
                
                    eNSDepth->setMeasurement(depth_measurment - ini_depth);

                    Matrix9d InvCovPVR = imupreint.getCovPVPhi();
                    Matrix3d CovPos = InvCovPVR.block(0,0,2,2);
                    
                    //eNSDepth->setInformation(InvCovPVR);
                    Vector3d e3 = Eigen::Vector3d::Zero();
                    e3(2)=1.0;
                    Matrix3d Rwb = pKF0->mNavState.Get_RotMatrix();
                    Matrix<double,1,1> cov3 = e3.transpose()*Rwb.transpose()*CovPos*Rwb*e3;
                    double cov1 = (shi*shi*depth_cov*depth_cov) + (double)cov3(0,0);
                    Matrix<double,1,1> InvCovDepth;
                    InvCovDepth(0,0)= 1/(cov1*cov1);

                   // std::cout<<"...................."<<std::endl<<std::endl;
                   // std::cout<<"START of Varience: " << std::endl;
                    
                   // Matrix<double,6,6> InvCovBgaRWVV = InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime();
                   // std::cout<<"IMU sensor varience : "<< CovPos(2,2) << std::endl;
                   // std::cout<<"Gryro bias sensor varience : "<< InvCovBgaRWVV(2,2) << std::endl;
                   // std::cout<<"Acc bias sensor varience : "<< InvCovBgaRWVV(5,5) << std::endl;
                   // std::cout<<"depth sensor varience : "<< shi*shi*depth_cov*depth_cov << std::endl;
                   // std::cout<<"forward depth varience : "<< cov1*cov1 << std::endl;


                    eNSDepth->setInformation(InvCovDepth);
                    //eNSDepth->SetParams(GravityRot);

                    const float thHuberNavStateDepth = sqrt(16.812);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    eNSDepth->setRobustKernel(rk);
                    rk->setDelta(thHuberNavStateDepth);

                    optimizer.addEdge(eNSDepth);

                }
                else
                {   
                    if  (minKFID +1 < pKF0->mnId)
                    {
                        std::cout<<"Running Depth Backward......................."<<std::endl<<std::endl;
                        KeyFrame* pKF_1 = pKF0->GetPrevKeyFrame();
                        eNSDepth->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->mnId)));
                        eNSDepth->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF_1->mnId)));
                        eNSDepth->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF_1->mnId+1)));
                        double shi = (pKF0->mTimeStamp - pKF_1->mTimeStamp)/(depth_time - pKF_1->mTimeStamp);
                        const IMUPreintegrator& imupreint  = pKF_1->GetIMUPreInt();
                        eNSDepth->SetParams(shi,imupreint);
                        
                        eNSDepth->setMeasurement(depth_measurment - ini_depth);
                        //std::cout<<"current depth"<<depth_measurment - ini_depth<<std::endl;
                        //std::cout<<"current z pose"<< pKF_1->GetPose()<<std::endl;
                        Matrix9d InvCovPVR = imupreint.getCovPVPhi();
                        Matrix3d CovPos = InvCovPVR.block(0,0,2,2);
                        //eNSDepth->setInformation(InvCovPVR);
                        Vector3d e3 = Eigen::Vector3d::Zero();
                        e3(2)=1.0;
                        Matrix3d Rwb = pKF_1->mNavState.Get_RotMatrix();
                        Matrix<double,1,1> cov3 = e3.transpose()*Rwb.transpose()*CovPos*Rwb*e3;
                        double cov1 = (shi*shi*depth_cov*depth_cov) + (double)cov3(0,0);
                        Matrix<double,1,1> InvCovDepth;
                        InvCovDepth(0,0)= 1/(cov1*cov1);
                        
                        eNSDepth->setInformation(InvCovDepth);
                        //eNSDepth->SetParams(GravityRot);

                        const float thHuberNavStateDepth = sqrt(16.812);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        eNSDepth->setRobustKernel(rk);
                        rk->setDelta(thHuberNavStateDepth);

                        optimizer.addEdge(eNSDepth); 
                    }

                }
            }
        }
        // Test log
        if(pKF1->GetIMUPreInt().getDeltaTime() < 1e-3)
        {
            cerr<<"IMU pre-integrator delta time between 2 KFs too small: "<<pKF1->GetIMUPreInt().getDeltaTime()<<endl;
            cerr<<"No EdgeNavState added"<<endl;
            continue;
        }
        if(lit == lLocalKeyFrames.begin())
        {
            // First KF in Local Window, link (fixed) pKFPrevLocal
            if(pKF0 != pKFPrevLocal) cerr<<"pKF0 != pKFPrevLocal for 1st KF in Local Window, id: "<<pKF0->mnId<<","<<pKFPrevLocal->mnId<<endl;
        }
        else
        {
            // KFs in Local Window, link another local KF
        }
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeNavStatePVRPointXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // Set edges between KeyFrame and MapPoint
        //float avr_varience = 0.0;
        //int count =0;
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad() && pKFi->mnId >= limitID)
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                //if(pKFi->mvuRight[mit->second]<0)
                
                if (true)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeNavStatePVRPointXYZ* e = new g2o::EdgeNavStatePVRPointXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->GetInvSigma2(kpUn.octave);
                    //avr_varience = avr_varience + invSigma2; 
                    //count ++;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->SetParams(pKFi->fx,pKFi->fy,pKFi->cx,pKFi->cy,Rbc,Pbc);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else
                {
                    // Test log
                    cerr<<"Stereo not supported yet, why here?? check."<<endl;
                }
            }
        }
        //std::cout<<"map point average varience : "<< avr_varience/count << std::endl;
        //std::cout<<"END of Varience: " << std::endl;
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    // First try
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {
    // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }
            // this is remoing outliers, this was omitted in original implementation
            //    // Check inlier observations
            //    for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
            //    {
            //        g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
            //        if(e->chi2()>21.666)
            //        {
            //            //e->setLevel(1);
            //            //cout<<"1 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
            //        }
            //        //e->setRobustKernel(0);
            //    }
            //    for(size_t i=0, iend=vpEdgesNavStateBias.size(); i<iend; i++)
            //    {
            //        g2o::EdgeNavStateBias* e = vpEdgesNavStateBias[i];
            //        if(e->chi2()>16.812)
            //        {
            //            //e->setLevel(1);
            //            //cout<<"1 Bias edge "<<i<<", chi2 "<<e->chi2()<<". ";
            //        }
            //        //e->setRobustKernel(0);
            //    }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    //
    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size());

    double PosePointchi2=0;
    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }

        PosePointchi2 += e->chi2();
    }

//    // Debug log
//    // Check inlier observations
//    for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
//    {
//        g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
//        if(e->chi2()>21.666)
//        {
//            //cout<<"2 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
//        }
//    }
//    for(size_t i=0, iend=vpEdgesNavStateBias.size(); i<iend; i++)
//    {
//        g2o::EdgeNavStateBias* e = vpEdgesNavStateBias[i];
//        if(e->chi2()>16.812)
//        {
//            //cout<<"2 Bias edge "<<i<<", chi2 "<<e->chi2()<<". ";
//        }
//    }
//    //cout<<"pose-point chi2: "<<PosePointchi2<<", pose-pose chi2: "<<PosePosechi2<<endl;

    // Get Map Mutex
    boost::mutex::scoped_lock lock(pMap->mMutexMapUpdate);
   
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    //Keyframes
    for(list<KeyFrame*>::const_iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexNavStatePVR* vNSPVR = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(2*pKFi->mnId));
        g2o::VertexNavStateBias* vNSBias = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(2*pKFi->mnId+1));
        // In optimized navstate, bias not changed, delta_bias not zero, should be added to bias
        const NavState& optPVRns = vNSPVR->estimate();
        const NavState& optBiasns = vNSBias->estimate();
        NavState primaryns = pKFi->GetNavState();
        // Update NavState
        pKFi->SetNavStatePos(optPVRns.Get_P());
        pKFi->SetNavStateVel(optPVRns.Get_V());
        pKFi->SetNavStateRot(optPVRns.Get_R());
        //if(optBiasns.Get_dBias_Acc().norm()<1e-2 && optBiasns.Get_BiasGyr().norm()<1e-4)
        //{
        pKFi->SetNavStateDeltaBg(optBiasns.Get_dBias_Gyr());
        pKFi->SetNavStateDeltaBa(optBiasns.Get_dBias_Acc());
        //}

        // Update pose Tcw
        pKFi->UpdatePoseFromNS(ConfigParam::GetMatTbc());

        // Test log
        if( (primaryns.Get_BiasGyr() - optPVRns.Get_BiasGyr()).norm() > 1e-6 || (primaryns.Get_BiasGyr() - optBiasns.Get_BiasGyr()).norm() > 1e-6 )
            cerr<<"gyr bias change in optimization?"<<endl;
        if( (primaryns.Get_BiasAcc() - optPVRns.Get_BiasAcc()).norm() > 1e-6 || (primaryns.Get_BiasAcc() - optBiasns.Get_BiasAcc()).norm() > 1e-6 )
            cerr<<"acc bias change in optimization?"<<endl;
        // Debug log
        /*
        cout<<"updated delta bias gyr: "<<optns.Get_dBias_Gyr().transpose()<<endl;
        cout<<"updated delta bias acc: "<<optns.Get_dBias_Acc().transpose()<<endl;
        cout<<"before and after opt, navstate.P: "<<primaryns.Get_P().transpose()<<" vs "<<optns.Get_P().transpose()<<endl;
        cout<<"before and after opt, navstate.V: "<<primaryns.Get_V().transpose()<<" vs "<<optns.Get_V().transpose()<<endl;
        cout<<"before and after opt, navstate.R: "<<primaryns.Get_RotMatrix()<<" vs "<<optns.Get_RotMatrix()<<endl;
        */
    }

    //Points
    for(list<MapPoint*>::const_iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
    if(pLM)
    {
        pLM->SetMapUpdateFlagInTracking(true);
    }


}

Vector3d Optimizer::OptimizeInitialGyroBias(const std::vector<FrameKTL> &vFrames)
{
    //size_t N = vpKFs.size();
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Add vertex of gyro bias, to optimizer graph
    g2o::VertexGyrBias * vBiasg = new g2o::VertexGyrBias();
    vBiasg->setEstimate(Eigen::Vector3d::Zero());
    vBiasg->setId(0);
    optimizer.addVertex(vBiasg);

    // Add unary edges for gyro bias vertex
    for(size_t i=0; i<vFrames.size(); i++)
    {
        // Only 19 edges between 20 Frames
        if(i==0)
            continue;

        const FrameKTL& Fi = vFrames[i-1];
        const FrameKTL& Fj = vFrames[i];

        cv::Mat Tiw = Fi.mTcw;      // pose of previous KF
        Eigen::Matrix3d Rwci = Converter::toMatrix3d(Tiw.rowRange(0,3).colRange(0,3).t());
        cv::Mat Tjw = Fj.mTcw;      // pose of this KF
        Eigen::Matrix3d Rwcj = Converter::toMatrix3d(Tjw.rowRange(0,3).colRange(0,3).t());

        //
        IMUPreintegrator imupreint;
        Fj.ComputeIMUPreIntSinceLastFrame(&Fi,imupreint);

        g2o::EdgeGyrBias * eBiasg = new g2o::EdgeGyrBias();
        eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // measurement is not used in EdgeGyrBias
        eBiasg->dRbij = imupreint.getDeltaR();
        eBiasg->J_dR_bg = imupreint.getJRBiasg();
        eBiasg->Rwbi = Rwci*Rcb;
        eBiasg->Rwbj = Rwcj*Rcb;
        eBiasg->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(eBiasg);
    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    g2o::VertexGyrBias * vBgEst = static_cast<g2o::VertexGyrBias*>(optimizer.vertex(0));

    return vBgEst->estimate();
}

Vector3d Optimizer::OptimizeInitialGyroBias(const std::list<KeyFrame *> &lLocalKeyFrames)
{
    return OptimizeInitialGyroBias(std::vector<KeyFrame*>(lLocalKeyFrames.begin(),lLocalKeyFrames.end()));
}

Vector3d Optimizer::OptimizeInitialGyroBias(const std::vector<KeyFrame *> &vpKFs)
{
    //size_t N = vpKFs.size();
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Add vertex of gyro bias, to optimizer graph
    g2o::VertexGyrBias * vBiasg = new g2o::VertexGyrBias();
    vBiasg->setEstimate(Eigen::Vector3d::Zero());
    vBiasg->setId(0);
    optimizer.addVertex(vBiasg);

    // Add unary edges for gyro bias vertex
    KeyFrame* pPrevKF0 = vpKFs.front();
    for(std::vector<KeyFrame*>::const_iterator lit=vpKFs.begin(), lend=vpKFs.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vpKFs.front())
            continue;

        KeyFrame* pPrevKF = pKF->GetPrevKeyFrame();
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        Eigen::Matrix3d Rwci = Converter::toMatrix3d(Twi.rowRange(0,3).colRange(0,3));
        cv::Mat Twj = pKF->GetPoseInverse();        // pose of this KF
        Eigen::Matrix3d Rwcj = Converter::toMatrix3d(Twj.rowRange(0,3).colRange(0,3));

        //
        const IMUPreintegrator& imupreint = pKF->GetIMUPreInt();
        g2o::EdgeGyrBias * eBiasg = new g2o::EdgeGyrBias();
        eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // measurement is not used in EdgeGyrBias
        eBiasg->dRbij = imupreint.getDeltaR();
        eBiasg->J_dR_bg = imupreint.getJRBiasg();
        eBiasg->Rwbi = Rwci*Rcb;
        eBiasg->Rwbj = Rwcj*Rcb;
        eBiasg->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(eBiasg);

        // Test log
        if(pPrevKF0 != pPrevKF) cerr<<"pPrevKF in list != pKF->pPrevKF? in OptimizeInitialGyroBias"<<endl;
        pPrevKF0 = pKF;

        // Debug log
        //cout<<"dRbij in pre-int: "<<endl<<eBiasg->dRbij<<endl;
        //cout<<"Rwbi'*Rwbj by ORBSLAM: "<<endl<<eBiasg->Rwbi.transpose()*eBiasg->Rwbj<<endl;
    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    g2o::VertexGyrBias * vBgEst = static_cast<g2o::VertexGyrBias*>(optimizer.vertex(0));

    return vBgEst->estimate();
}


//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag);
}

void Optimizer::RecoveryBundleAdjustemnt( KeyFrame* pKFini,KeyFrame* pKFcur,int nIterations, bool* pbStopFlag)
{
    vector<KeyFrame*> vpKFs;
    vpKFs.push_back(pKFini);
    vpKFs.push_back(pKFcur);
    vector<MapPoint*> vpMP = pKFcur->getmvpMappoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, int nIterations, bool* pbStopFlag)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // SET KEYFRAME VERTICES
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        if(pKF->mnId==0){
        vSE3->setFixed(true);
        }
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }


    const float thHuber = sqrt(5.991);

    // SET MAP POINT VERTICES
    for(size_t i=0, iend=vpMP.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMP[i];
        //if(!pMP)
            //continue;
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //SET EDGES
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad())
                continue;
            Eigen::Matrix<double,2,1> obs;
            cv::KeyPoint kpUn = pKF->GetKeyPointUn(mit->second);
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            float invSigma2 = pKF->GetInvSigma2(kpUn.octave);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuber);

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            optimizer.addEdge(e);
        }
    }

    // Optimize!

    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        //if(pKF->isBad())
            //continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(size_t i=0, iend=vpMP.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMP[i];
        //if(pMP->isBad())
            //          continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

}

int Optimizer::PoseOptimization(FrameKTL *pFrame)
{    
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    int nInitialCorrespondences=0;

    // SET FRAME VERTEX
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // SET MAP POINT VERTICES
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vector<g2o::VertexSBAPointXYZ*> vVertices;
    vector<float> vInvSigmas2;
    vector<size_t> vnIndexEdge;

    const int N = pFrame->mvpMapPoints.size();
    vpEdges.reserve(N);
    vVertices.reserve(N);
    vInvSigmas2.reserve(N);
    vnIndexEdge.reserve(N);

    const float delta = sqrt(5.991);
    {
    boost::mutex::scoped_lock lock(MapPoint::mGlobalMutex);
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            vPoint->setId(i+1);
            vPoint->setFixed(true);
            optimizer.addVertex(vPoint);
            vVertices.push_back(vPoint);

            nInitialCorrespondences++;
            pFrame->mvbOutlier[i] = false;

            //SET EDGE
            Eigen::Matrix<double,2,1> obs;
            cv::KeyPoint kpUn = pFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(delta);

            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;

            e->setLevel(0);

            optimizer.addEdge(e);

            vpEdges.push_back(e);
            vInvSigmas2.push_back(invSigma2);
            vnIndexEdge.push_back(i);
        }

    }
    }

    // We perform 4 optimizations, decreasing the inlier region
    // From second to final optimization we include only inliers in the optimization
    // At the end of each optimization we check which points are inliers
    const float chi2[4]={9.210,7.378,5.991,5.991};
    const int its[4]={10,10,7,5};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

            const size_t idx = vnIndexEdge[i];

            if(pFrame->mvbOutlier[idx])
                e->computeError();

            if(e->chi2()>chi2[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else if(e->chi2()<=chi2[it])
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }
        }

        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pose.copyTo(pFrame->mTcw);

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag,Map* pMap,LocalMapping* pLM)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // SET LOCAL KEYFRAME VERTICES
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // SET FIXED KEYFRAME VERTICES
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // SET MAP POINT VERTICES
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vpEdges.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKF;
    vpEdgeKF.reserve(nExpectedSize);

    vector<float> vSigmas2;
    vSigmas2.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdge;
    vpMapPointEdge.reserve(nExpectedSize);

    const float thHuber = sqrt(5.991);
    {
    //boost::mutex::scoped_lock lock(MapPoint::mGlobalMutex);// NOT PResent Originally..............................................................

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //SET EDGES
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                Eigen::Matrix<double,2,1> obs;
                cv::KeyPoint kpUn = pKFi->GetKeyPointUn(mit->second);
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                float sigma2 = pKFi->GetSigma2(kpUn.octave);
                float invSigma2 = pKFi->GetInvSigma2(kpUn.octave);
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;

                optimizer.addEdge(e);
                vpEdges.push_back(e);
                vpEdgeKF.push_back(pKFi);
                vSigmas2.push_back(sigma2);
                vpMapPointEdge.push_back(pMP);
            }
        }
    }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inlier observations
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];
        MapPoint* pMP = vpMapPointEdge[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKF[i];
            pKFi->EraseMapPointMatch(pMP);
            pMP->EraseObservation(pKFi);

            optimizer.removeEdge(e);
            vpEdges[i]=NULL;
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }
    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Check inlier observations
    boost::mutex::scoped_lock lock(pMap->mMutexMapUpdate);
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

        if(!e)
            continue;

        MapPoint* pMP = vpMapPointEdge[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKF = vpEdgeKF[i];
            pKF->EraseMapPointMatch(pMP->GetIndexInKeyFrame(pKF));
            pMP->EraseObservation(pKF);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
    if(pLM)
    {
        pLM->SetMapUpdateFlagInTracking(true);
    }
    
}



void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, g2o::Sim3 &Scurw,
                                       LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    unsigned int nMaxKFid = pMap->GetMaxKFid();


    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // SET KEYFRAME VERTICES
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        int nIDi = pKF->mnId;      

        if(CorrectedSim3.count(pKF))
        {
            vScw[nIDi] = CorrectedSim3[pKF];
            VSim3->setEstimate(CorrectedSim3[pKF]);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // SET LOOP EDGES
    for(map<KeyFrame *, set<KeyFrame *> >::iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        set<KeyFrame*> &spConnections = mit->second;
        g2o::Sim3 Siw = vScw[nIDi];
        g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            g2o::Sim3 Sjw = vScw[nIDj];
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // SET NORMAL EDGES
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;
        if(NonCorrectedSim3.count(pKF))
            Swi = NonCorrectedSim3[pKF].inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            if(NonCorrectedSim3.count(pParentKF))
                Sjw = NonCorrectedSim3[pParentKF];
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;
                if(NonCorrectedSim3.count(pLKF))
                    Slw = NonCorrectedSim3[pLKF];
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;
                    if(NonCorrectedSim3.count(pKFn))
                        Snw = NonCorrectedSim3[pKFn];
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // OPTIMIZE

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, float th2)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();

    // Camera poses
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    // SET SIMILARITY VERTEX
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // SET MAP POINT VERTICES
    const int N = vpMatches1.size();
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<float> vSigmas12, vSigmas21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        int id1 = 2*i+1;
        int id2 = 2*(i+1);

        int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // SET EDGE x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        cv::KeyPoint kpUn1 = pKF1->GetKeyPointUn(i);
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        float invSigmaSquare1 = pKF1->GetInvSigma2(kpUn1.octave);
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // SET EDGE x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2 = pKF2->GetKeyPointUn(i2);
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->GetSigma2(kpUn2.octave);
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=NULL;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=NULL;
            vpEdges21[i]=NULL;
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=NULL;
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

double Optimizer::OptimizeInitialScale(const std::vector<KeyFrame *> &vLocalKeyFrames, Eigen::Matrix3d& Rg_w,double ini_depth, double &AVG)
{
    //size_t N = vpKFs.size();
    //Matrix4d Tbc = ConfigParam::GetEigTbc();
    //Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();
    //Rg_w = Rg_w.transpose();
    // Setup optimizer
    int n = vLocalKeyFrames.size();
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    //optimizer.setVerbose(false);

    // Add vertex of scale, to optimizer graph
    g2o::VertexScale* vScale = new g2o::VertexScale();
    Eigen::Matrix<double,1,1> iniValue;
    iniValue(0,0)=2;
    vScale->setEstimate(iniValue);
    vScale->setId(0);
    optimizer.addVertex(vScale);
    optimizer.setVerbose(false);

    cv::Mat Tbc = ConfigParam::GetMatTbc();
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3);
    cv::Mat pbc = Tbc.rowRange(0,3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb*pbc;
    double CV = 5.991;
    int k=0;
    double ScaleAvg=0.0;
    KeyFrame* pPrevKF0 = vLocalKeyFrames.front();
    /*
    for(std::vector<KeyFrame*>::const_iterator lit=vLocalKeyFrames.begin(), lend=vLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vLocalKeyFrames.front())
            continue;
        //KeyFrame* pPrevKF = pKF->GetPrevKeyFrame();
        KeyFrame* pPrevKF = vLocalKeyFrames.front();
        
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        cv::Mat Twj = pKF->GetPoseInverse();  
        cv::Mat Pwi = Twi.rowRange(0,3).col(3);
        cv::Mat Pwj = Twj.rowRange(0,3).col(3);
        cv::Mat Rwi = Twi.rowRange(0,3).colRange(0,3);
        cv::Mat Rwj = Twj.rowRange(0,3).colRange(0,3);  
                 
        g2o::EdgeScaleDepth *eScale = new g2o::EdgeScaleDepth();
        eScale->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   
        eScale->setMeasurement(pKF->mDepth- pPrevKF->mDepth); 
        //std::cout<<std::endl<<"Pressure Values "<<pKF->mDepth<<" "<<pPrevKF->mDepth<<std::endl<<std::endl;   
        
        eScale->Pi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj));  
                                                          
        eScale->Rg_w =Rg_w;
        eScale->setId(k);
        k++;
        Matrix<double,1,1> InvCovDepth;
        InvCovDepth(0)= 1;
        eScale->setInformation(InvCovDepth);

        const float thHuberNavStateDepth = sqrt(CV);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eScale->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale);
        Vector3d d_pos = Rg_w*eScale->Pj- Rg_w*eScale->Pi;
        ScaleAvg = ScaleAvg+ ((pKF->mDepth- pPrevKF->mDepth)/d_pos(2,0));
    }*/

    for(std::vector<KeyFrame*>::const_iterator lit=vLocalKeyFrames.begin(), lend=vLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vLocalKeyFrames.front())
            continue;

        KeyFrame* pPrevKF = pKF->GetPrevKeyFrame();
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        cv::Mat Twj = pKF->GetPoseInverse();  
        cv::Mat Pwi = Twi.rowRange(0,3).col(3);
        cv::Mat Pwj = Twj.rowRange(0,3).col(3);
        cv::Mat Rwi = Twi.rowRange(0,3).colRange(0,3);
        cv::Mat Rwj = Twj.rowRange(0,3).colRange(0,3);  
                 
        g2o::EdgeScaleDepth *eScale = new g2o::EdgeScaleDepth();
        eScale->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   
        eScale->setMeasurement(abs(pKF->mDepth- pPrevKF->mDepth)); 
        //std::cout<<std::endl<<"Pressure Values "<<pKF->mDepth<<" "<<pPrevKF->mDepth<<std::endl<<std::endl;   
        
        eScale->Pi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj));  
                                                          
        eScale->Rg_w =Rg_w;
        eScale->setId(k);
        k++;
        Matrix<double,1,1> InvCovDepth;
        InvCovDepth(0)= 1/(0.01*0.01);
        eScale->setInformation(InvCovDepth);

        const float thHuberNavStateDepth = sqrt(CV);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eScale->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale);
        Vector3d d_pos = Rg_w*eScale->Pj- Rg_w*eScale->Pi;
        ScaleAvg = ScaleAvg+ ((pKF->mDepth- pPrevKF->mDepth)/d_pos(2,0));
    }

    for(std::vector<KeyFrame*>::const_iterator lit=vLocalKeyFrames.begin(), lend=vLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vLocalKeyFrames.front())
            continue;
        KeyFrame* pPrevKF1 = pKF->GetPrevKeyFrame();
        if(pPrevKF1 == vLocalKeyFrames.front())
            continue;

        KeyFrame* pPrevKF = pPrevKF1->GetPrevKeyFrame();
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        cv::Mat Twj = pKF->GetPoseInverse();  
        cv::Mat Pwi = Twi.rowRange(0,3).col(3);
        cv::Mat Pwj = Twj.rowRange(0,3).col(3);
        cv::Mat Rwi = Twi.rowRange(0,3).colRange(0,3);
        cv::Mat Rwj = Twj.rowRange(0,3).colRange(0,3);  
                 
        g2o::EdgeScaleDepth *eScale = new g2o::EdgeScaleDepth();
        eScale->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   
        eScale->setMeasurement(abs(pKF->mDepth- pPrevKF->mDepth)); 
        //std::cout<<std::endl<<"Pressure Values "<<pKF->mDepth<<" "<<pPrevKF->mDepth<<std::endl<<std::endl;   
        
        eScale->Pi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj));  
                                                          
        eScale->Rg_w =Rg_w;
        eScale->setId(k);
        k++;
        Matrix<double,1,1> InvCovDepth;
        InvCovDepth(0)= 1/(0.01*0.01);
        eScale->setInformation(InvCovDepth);

        const float thHuberNavStateDepth = sqrt(CV);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eScale->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale);
        Vector3d d_pos = Rg_w*eScale->Pj- Rg_w*eScale->Pi;
        ScaleAvg = ScaleAvg+ ((pKF->mDepth- pPrevKF->mDepth)/d_pos(2,0));
    }
    
    AVG=ScaleAvg/k;
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    g2o::VertexScale * ScaleEst = static_cast<g2o::VertexScale*>(optimizer.vertex(0));
    const Matrix<double,1,1>& EstBestScale = ScaleEst->estimate();
    return (double)EstBestScale(0);
}

    

double Optimizer::OptimizeInitialScaleSecond(const std::vector<KeyFrame*> &vLocalKeyFrames, Eigen::Matrix3d& Rg_w,double ini_depth,cv::Mat biasa, double depth_cov)
{
    //size_t N = vpKFs.size();
    //Matrix4d Tbc = ConfigParam::GetEigTbc();
    //Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    // Setup optimizer
    int n = vLocalKeyFrames.size();
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Add vertex of scale, to optimizer graph
    g2o::VertexScale* vScale = new g2o::VertexScale();
    Eigen::Matrix<double,1,1> iniValue;
    iniValue(0,0)=10;
    vScale->setEstimate(iniValue);
    vScale->setId(0);
    optimizer.addVertex(vScale);

    cv::Mat Tbc = ConfigParam::GetMatTbc();
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3);
    cv::Mat pbc = Tbc.rowRange(0,3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb*pbc;

    KeyFrame* pPrevKF0 = vLocalKeyFrames.front();
    double CV  = 5.991;
    for(std::vector<KeyFrame*>::const_iterator lit=vLocalKeyFrames.begin(), lend=vLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vLocalKeyFrames.front())
            continue;
        KeyFrame* pPrevKF = pKF->GetPrevKeyFrame();
        //if(pPrevKF1 == vLocalKeyFrames.front()){
            //continue;}
        //KeyFrame* pPrevKF = pPrevKF1->GetPrevKeyFrame();
        //KeyFrame* pPrevKF = vLocalKeyFrames.front();
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        cv::Mat Twj = pKF->GetPoseInverse();  
        cv::Mat Pwi = Twi.rowRange(0,3).col(3);
        cv::Mat Pwj = Twj.rowRange(0,3).col(3);
        cv::Mat Rwi = Twi.rowRange(0,3).colRange(0,3);
        cv::Mat Rwj = Twj.rowRange(0,3).colRange(0,3);  
        cv::Mat Pwbi = Rwi*pcb;
        cv::Mat Pwbj = Rwj*pcb; 
        cv::Mat dPij = Converter::toCvMat(pKF->GetIMUPreInt().getDeltaP());
        cv::Mat Jpbaij = Converter::toCvMat(pKF->GetIMUPreInt().getJPBiasa()); 

        cv::Mat dPijw =  Rwj* Rcb * (dPij - Jpbaij*biasa);   
        g2o::EdgeScaleDepth * eScale1 = new g2o::EdgeScaleDepth();
        eScale1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  
        eScale1->setMeasurement(dPijw.at<double>(0));    
        //eScale->setMeasurement(pKF->mDepth - ini_depth);
        
        eScale1->Pi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale1->Pj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj)); 
        eScale1->Rg_w =Rg_w;
        Matrix<double,1,1> InvCovDepth1;
        Matrix9d covP =   pKF->GetIMUPreInt().getCovPVPhi().inverse();//Why inverse here..?
        InvCovDepth1(0)=  1/covP(2,2);
        eScale1->setInformation(InvCovDepth1);

        //const float thHuberNavStateDepth = sqrt(CV);
        //g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        //eScale1->setRobustKernel(rk1);
        //rk1->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale1);
        /*
        g2o::EdgeScaleDepth * eScale2 = new g2o::EdgeScaleDepth();
        eScale2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  
        eScale2->Delta_Depth = dPijw.at<double>(1);    
        //eScale->setMeasurement(pKF->mDepth - ini_depth);
        
        eScale->Pwi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pwj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj)); 
        eScale->Rg_w =Rg_w;
        eScale->pcbi=Converter::toVector3d(Pwbi);
        eScale->pcbj=Converter::toVector3d(Pwbj);Pwbj); 
        eScale2->J_scale = eScale2->Pwbj(1);
        Matrix<double,1,1> InvCovDepth2;
        InvCovDepth2(0)=  1/covP(1,1);
        eScale2->setInformation(InvCovDepth2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        eScale2->setRobustKernel(rk2);
        rk2->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale2);

        g2o::EdgeScaleDepth * eScale3 = new g2o::EdgeScaleDepth();
        eScale3->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  
        eScale3->Delta_Depth = dPijw.at<double>(2);    
        //eScale->setMeasurement(pKF->mDepth - ini_depth);
        
        eScale->Pwi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pwj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj)); 
        eScale->Rg_w =Rg_w;
        eScale->pcbi=Converter::toVector3d(Pwbi);
        eScale->pcbj=Converter::toVector3d(Pwbj);
        Matrix<double,1,1> InvCovDepth3;
        InvCovDepth3(0)=  1/covP(2,2);
        eScale3->setInformation(InvCovDepth3);

        g2o::RobustKernelHuber* rk3 = new g2o::RobustKernelHuber;
        eScale3->setRobustKernel(rk3);
        rk3->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale3);
        */
    }
    for(std::vector<KeyFrame*>::const_iterator lit=vLocalKeyFrames.begin(), lend=vLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        // Ignore the first KF
        if(pKF == vLocalKeyFrames.front())
            continue;
        KeyFrame* pPrevKF = pKF->GetPrevKeyFrame();
        //if(pPrevKF1 == vLocalKeyFrames.front()){
            //continue;}
        //KeyFrame* pPrevKF = pPrevKF1->GetPrevKeyFrame();
        //KeyFrame* pPrevKF = vLocalKeyFrames.front();
        cv::Mat Twi = pPrevKF->GetPoseInverse();    // pose of previous KF
        cv::Mat Twj = pKF->GetPoseInverse();  
        cv::Mat Pwi = Twi.rowRange(0,3).col(3);
        cv::Mat Pwj = Twj.rowRange(0,3).col(3);
        cv::Mat Rwi = Twi.rowRange(0,3).colRange(0,3);
        cv::Mat Rwj = Twj.rowRange(0,3).colRange(0,3);  
        cv::Mat Pwbi = Rwi*pcb;
        cv::Mat Pwbj = Rwj*pcb;          
        g2o::EdgeScaleDepth * eScale = new g2o::EdgeScaleDepth();
        eScale->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  
        eScale->setMeasurement(pKF->mDepth- pPrevKF->mDepth);    
        //eScale->setMeasurement(pKF->mDepth - ini_depth);
        
        eScale->Pi =Converter::toVector3d(Pwi);//Rg_w*(Converter::toVector3d(Pwbi));   
        eScale->Pj =Converter::toVector3d(Pwj);//Rg_w*(Converter::toVector3d(Pwbj)); 
        eScale->Rg_w =Rg_w;
        Matrix<double,1,1> InvCovDepth;
        Matrix9d covP =   pKF->GetIMUPreInt().getCovPVPhi().inverse();
        InvCovDepth(0)=  1/(depth_cov*depth_cov);
        eScale->setInformation(InvCovDepth);

        //const float thHuberNavStateDepth = sqrt(5.66);
        //g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        //eScale->setRobustKernel(rk);
        //rk->setDelta(thHuberNavStateDepth);

        optimizer.addEdge(eScale);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    g2o::VertexScale * ScaleEst = static_cast<g2o::VertexScale*>(optimizer.vertex(0));
    const Matrix<double,1,1>& EstBestScale = ScaleEst->estimate();
    return (double)EstBestScale(0);
}


} //namespace USLAM
