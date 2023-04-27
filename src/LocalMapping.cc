/*
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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <ros/ros.h>

namespace USLAM
{

//...........................................................
KeyFrame* LocalMapping::GetMapUpdateKF()
{
    boost::mutex::scoped_lock(mMutexMapUpdateFlag);
    return mpMapUpdateKF;
}

bool LocalMapping::GetMapUpdateFlagForTracking()
{
    boost::mutex::scoped_lock(mMutexMapUpdateFlag);
    return mbMapUpdateFlagForTracking;
}

void LocalMapping::SetMapUpdateFlagInTracking(bool bflag)
{
    boost::mutex::scoped_lock(mMutexMapUpdateFlag);
    mbMapUpdateFlagForTracking = bflag;
    if(bflag)
    {
        mpMapUpdateKF = mpCurrentKeyFrame;
    }
}

bool LocalMapping::GetVINSInited(void)
{
    boost::mutex::scoped_lock(mMutexVINSInitFlag);
    return mbVINSInited;
}

void LocalMapping::SetVINSInited(bool flag)
{
    boost::mutex::scoped_lock(mMutexVINSInitFlag);
    mbVINSInited = flag;
}

bool LocalMapping::GetFirstVINSInited(void)
{
    boost::mutex::scoped_lock(mMutexFirstVINSInitFlag);
    return mbFirstVINSInited;
}

void LocalMapping::SetFirstVINSInited(bool flag)
{
    boost::mutex::scoped_lock(mMutexFirstVINSInitFlag);
    mbFirstVINSInited = flag;
}

cv::Mat LocalMapping::GetGravityVec()
{
    return mGravityVec.clone();
}

cv::Mat LocalMapping::GetGravityRotation()
{
    return mGravityRot.clone();
}


bool LocalMapping::TryInitVIO(void)
{
    if(mpMap->KeyFramesInMap()<=mnLocalWindowSize)
        return false;

    static bool fopened = false;
    static ofstream fgw,fscale,fbiasa,fcondnum,ftime,fbiasg;
    if(!fopened)
    {
        // Need to modify this to correct path
        string tmpfilepath = ConfigParam::getTmpFilePath();
        fgw.open(tmpfilepath+"gw.txt");
        fscale.open(tmpfilepath+"scale.txt");
        fbiasa.open(tmpfilepath+"biasa.txt");
        fcondnum.open(tmpfilepath+"condnum.txt");
        ftime.open(tmpfilepath+"computetime.txt");
        fbiasg.open(tmpfilepath+"biasg.txt");
        if(fgw.is_open() && fscale.is_open() && fbiasa.is_open() &&
                fcondnum.is_open() && ftime.is_open() && fbiasg.is_open())
            fopened = true;
        else
        {
            cerr<<"file open error in TryInitVIO"<<endl;
            fopened = false;
        }
        fgw<<std::fixed<<std::setprecision(6);
        fscale<<std::fixed<<std::setprecision(6);
        fbiasa<<std::fixed<<std::setprecision(6);
        fcondnum<<std::fixed<<std::setprecision(6);
        ftime<<std::fixed<<std::setprecision(6);
        fbiasg<<std::fixed<<std::setprecision(6);
    }

    // Extrinsics
    cv::Mat Tbc = ConfigParam::GetMatTbc();
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3);
    cv::Mat pbc = Tbc.rowRange(0,3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb*pbc;

    // Use all KeyFrames in map to compute
    vector<KeyFrame*> vScaleGravityKF = mpMap->GetAllKeyFrames();
    int N = vScaleGravityKF.size();

    // Step 1.
    // Try to compute initial gyro bias, using optimization with Gauss-Newton
    Vector3d bgest = Optimizer::OptimizeInitialGyroBias(vScaleGravityKF);

    // Update biasg and pre-integration in LocalWindow. Remember to reset back to zero
    for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        pKF->SetNavStateBiasGyr(bgest);
    }
    for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        pKF->ComputePreInt();
    }

    int Init_mode=0;
    if (mpTracker->mMode ==1){
        Init_mode=1;// if the SLAM runs visual-inertial only, Use the initialization from "map reuse " paper
    }
    if (mpTracker->mMode ==2){
        Init_mode=2;//This should be 2   chage during last testing due to - values in VIP scale. Chack VIP initialization later CHINTHAKA. this run the developed method
    }
    if (mpTracker->mMode ==2 && needboth_Inits){
        Init_mode=3;
    }

    cv::Mat GI;
    cv::Mat Rwi;
    cv::Mat Rwi_;
    double s_;
    double sstar;
    cv::Mat dbiasa_;
    cv::Mat w2,u2,vt2;
    cv::Mat Rwi_P;
    double Best_scale;
    double AVG_Scale= 0.0;
    cv::Mat dbiasa_P;
    cv::Mat w2P,u2P,vt2P;
    cv::Mat RwiP;
    Vector3d dbiasa_eig;
    Vector3d dbiasa_eigP;
    ros::Time t0=ros::Time::now();
    ros::Time t1=ros::Time::now();
    ros::Time t2=ros::Time::now();
    cv::Mat gI = cv::Mat::zeros(3,1,CV_32F);
    gI.at<float>(2) = 1;
    GI = gI*ConfigParam::GetG();//9.8012;

    if (Init_mode==1 || Init_mode==3){

        t0 =ros::Time::now();
        // Solve A*x=B for x=[s,gw] 4x1 vector
        cv::Mat A = cv::Mat::zeros(3*(N-2),4,CV_32F);
        cv::Mat B = cv::Mat::zeros(3*(N-2),1,CV_32F);
        cv::Mat I3 = cv::Mat::eye(3,3,CV_32F);

        // Step 2.
        // Approx Scale and Gravity vector in 'world' frame (first KF's camera frame)
        for(int i=0; i<N-2; i++)
        {
            KeyFrame* pKF1 = vScaleGravityKF[i];
            KeyFrame* pKF2 = vScaleGravityKF[i+1];
            KeyFrame* pKF3 = vScaleGravityKF[i+2];
            // Delta time between frames
            double dt12 = pKF2->GetIMUPreInt().getDeltaTime();
            double dt23 = pKF3->GetIMUPreInt().getDeltaTime();
            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(pKF3->GetIMUPreInt().getDeltaP());
            // Test log
            if(dt12!=pKF2->mTimeStamp-pKF1->mTimeStamp) cerr<<"dt12!=pKF2->mTimeStamp-pKF1->mTimeStamp"<<endl;
            if(dt23!=pKF3->mTimeStamp-pKF2->mTimeStamp) cerr<<"dt23!=pKF3->mTimeStamp-pKF2->mTimeStamp"<<endl;

            // Pose of camera in world frame
            cv::Mat Twc1 = pKF1->GetPoseInverse();
            cv::Mat Twc2 = pKF2->GetPoseInverse();
            cv::Mat Twc3 = pKF3->GetPoseInverse();
            // Position of camera center
            cv::Mat pc1 = Twc1.rowRange(0,3).col(3);
            cv::Mat pc2 = Twc2.rowRange(0,3).col(3);
            cv::Mat pc3 = Twc3.rowRange(0,3).col(3);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = Twc1.rowRange(0,3).colRange(0,3);
            cv::Mat Rc2 = Twc2.rowRange(0,3).colRange(0,3);
            cv::Mat Rc3 = Twc3.rowRange(0,3).colRange(0,3);

            // Stack to A/B matrix
            // lambda*s + beta*g = gamma
            cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
            cv::Mat beta = 0.5*I3*(dt12*dt12*dt23 + dt12*dt23*dt23);
            cv::Mat gamma = (Rc3-Rc2)*pcb*dt12 + (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt12*dt23;
            lambda.copyTo(A.rowRange(3*i+0,3*i+3).col(0));
            beta.copyTo(A.rowRange(3*i+0,3*i+3).colRange(1,4));
            gamma.copyTo(B.rowRange(3*i+0,3*i+3));
            // Tested the formulation in paper, -gamma. Then the scale and gravity vector is -xx

            // Debug log
            //cout<<"iter "<<i<<endl;
        }
        // Use svd to compute A*x=B, x=[s,gw] 4x1 vector
        // A = u*w*vt, u*w*vt*x=B
        // Then x = vt'*winv*u'*B
        cv::Mat w,u,vt;
        // Note w is 4x1 vector by SVDecomp()
        // A is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A);
        // Debug log
        //cout<<"u:"<<endl<<u<<endl;
        //cout<<"vt:"<<endl<<vt<<endl;
        //cout<<"w:"<<endl<<w<<endl;

        // Compute winv
        cv::Mat winv=cv::Mat::eye(4,4,CV_32F);
        for(int i=0;i<4;i++)
        {
            if(fabs(w.at<float>(i))<1e-10)
            {
                w.at<float>(i) += 1e-10;
                // Test log
                cerr<<"w(i) < 1e-10, w="<<endl<<w<<endl;
            }

            winv.at<float>(i,i) = 1./w.at<float>(i);
        }
        // Then x = vt'*winv*u'*B
        cv::Mat x = vt.t()*winv*u.t()*B;

        // x=[s,gw] 4x1 vector
        sstar = x.at<float>(0);    // scale should be positive
        cv::Mat gwstar = x.rowRange(1,4);   // gravity should be about ~9.8

        // Debug log
        //cout<<"scale sstar: "<<sstar<<endl;
        //cout<<"gwstar: "<<gwstar.t()<<", |gwstar|="<<cv::norm(gwstar)<<endl;

        // Test log
        if(w.type()!=I3.type() || u.type()!=I3.type() || vt.type()!=I3.type())
            cerr<<"different mat type, I3,w,u,vt: "<<I3.type()<<","<<w.type()<<","<<u.type()<<","<<vt.type()<<endl;
        
        // Step 3.
        // Use gravity magnitude 9.8 as constraint
        // gI = [0;0;1], the normalized gravity vector in an inertial frame, NED type with no orientation.
        // Normalized approx. gravity vecotr in world frame
        cv::Mat gwn = gwstar/cv::norm(gwstar);
        // Debug log
        //cout<<"gw normalized 1 : "<<gwn<<endl;

        // vhat = (gI x gw) / |gI x gw|
        cv::Mat gIxgwn = gI.cross(gwn);
        double normgIxgwn = cv::norm(gIxgwn);
        cv::Mat vhat = gIxgwn/normgIxgwn;
        double theta = std::atan2(normgIxgwn,gI.dot(gwn));
        // Debug log
        //cout<<"vhat: "<<vhat<<", theta: "<<theta*180.0/M_PI<<endl;

        Eigen::Vector3d vhateig = Converter::toVector3d(vhat);
        Eigen::Matrix3d RWIeig = Sophus::SO3::exp(vhateig*theta).matrix();
        Rwi = Converter::toCvMat(RWIeig);
        // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector
        
    
        cv::Mat C = cv::Mat::zeros(3*(N-2),6,CV_32F);
        cv::Mat D = cv::Mat::zeros(3*(N-2),1,CV_32F);

        for(int i=0; i<N-2; i++)
        {
            KeyFrame* pKF1 = vScaleGravityKF[i];
            KeyFrame* pKF2 = vScaleGravityKF[i+1];
            KeyFrame* pKF3 = vScaleGravityKF[i+2];
            // Delta time between frames
            double dt12 = pKF2->GetIMUPreInt().getDeltaTime();
            double dt23 = pKF3->GetIMUPreInt().getDeltaTime();
            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(pKF3->GetIMUPreInt().getDeltaP());
            cv::Mat Jpba12 = Converter::toCvMat(pKF2->GetIMUPreInt().getJPBiasa());
            cv::Mat Jvba12 = Converter::toCvMat(pKF2->GetIMUPreInt().getJVBiasa());
            cv::Mat Jpba23 = Converter::toCvMat(pKF3->GetIMUPreInt().getJPBiasa());
            // Pose of camera in world frame
            cv::Mat Twc1 = pKF1->GetPoseInverse();
            cv::Mat Twc2 = pKF2->GetPoseInverse();
            cv::Mat Twc3 = pKF3->GetPoseInverse();
            // Position of camera center
            cv::Mat pc1 = Twc1.rowRange(0,3).col(3);
            cv::Mat pc2 = Twc2.rowRange(0,3).col(3);
            cv::Mat pc3 = Twc3.rowRange(0,3).col(3);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = Twc1.rowRange(0,3).colRange(0,3);
            cv::Mat Rc2 = Twc2.rowRange(0,3).colRange(0,3);
            cv::Mat Rc3 = Twc3.rowRange(0,3).colRange(0,3);
            // Stack to C/D matrix
            // lambda*s + phi*dthetaxy + zeta*ba = psi
            cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
            cv::Mat phi = - 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*Rwi*SkewSymmetricMatrix(GI);  // note: this has a '-', different to paper
            cv::Mat zeta = Rc2*Rcb*Jpba23*dt12 + Rc1*Rcb*Jvba12*dt12*dt23 - Rc1*Rcb*Jpba12*dt23;
            cv::Mat psi = (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - (Rc2-Rc3)*pcb*dt12
                        - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt23*dt12 - 0.5*Rwi*GI*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper
            lambda.copyTo(C.rowRange(3*i+0,3*i+3).col(0));
            phi.colRange(0,2).copyTo(C.rowRange(3*i+0,3*i+3).colRange(1,3)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
            zeta.copyTo(C.rowRange(3*i+0,3*i+3).colRange(3,6));
            psi.copyTo(D.rowRange(3*i+0,3*i+3));
        }

        
        cv::SVDecomp(C,w2,u2,vt2,cv::SVD::MODIFY_A);
        
        cv::Mat w2inv=cv::Mat::eye(6,6,CV_32F);
        for(int i=0;i<6;i++)
        {
            if(fabs(w2.at<float>(i))<1e-10)
            {
                w2.at<float>(i) += 1e-10;
                // Test log
                cerr<<"w2(i) < 1e-10, w="<<endl<<w2<<endl;
            }

            w2inv.at<float>(i,i) = 1./w2.at<float>(i);
        }
        // Then y = vt'*winv*u'*D
        cv::Mat y = vt2.t()*w2inv*u2.t()*D;

        s_ = y.at<float>(0);
        cv::Mat dthetaxy = y.rowRange(1,3);
        dbiasa_ = y.rowRange(3,6);
        dbiasa_eig = Converter::toVector3d(dbiasa_);

        // dtheta = [dx;dy;0]
        cv::Mat dtheta = cv::Mat::zeros(3,1,CV_32F);
        dthetaxy.copyTo(dtheta.rowRange(0,2));
        Eigen::Vector3d dthetaeig = Converter::toVector3d(dtheta);
        // Rwi_ = Rwi*exp(dtheta)
        Eigen::Matrix3d Rwieig_ = RWIeig*Sophus::SO3::exp(dthetaeig).matrix();
        Rwi_ = Converter::toCvMat(Rwieig_);
        cv::Mat gwa_final = Rwi_*GI;
        //cout<<"g_final IMU: "<<gwa_final.t()<<", |g_final|="<<cv::norm(gwa_final)<<endl;
        //cout<<"Time: "<<mpCurrentKeyFrame->mTimeStamp - mnStartTime<<", Scale s_ IMU: "<<s_<<endl<<endl;
       
        // ********************************
        // Todo:
        // Add some logic or strategy to confirm init status

        //................................................................
        //float Avg_scale= 0.0;
    }

    if (Init_mode==2 || Init_mode==3)
    {
        t1 =ros::Time::now();
        cv::Mat g_vec = Converter::toCvMat(mpTracker->z_axis);
        cv::Mat T_bc = ConfigParam::GetMatTbc();
        cv::Mat R_bc = T_bc.rowRange(0,3).colRange(0,3).clone();
        g_vec= R_bc.t()*g_vec*(-9.8012);
        //cout<<"g from imu averaging w.r.t. world 1 : "<< g_vec.t()<<endl<<endl;
        cv::Mat GIP = cv::Mat::zeros(3,1,CV_32F);
        GIP.at<float>(2) = 1;
        // Normalized approx. gravity vecotr in world frame
        cv::Mat Gwn = g_vec/cv::norm(g_vec);
        
        //cout<<"gw normalized 2 : "<<Gwn<<endl;
        // vhat = (gI x gw) / |gI x gw|
        cv::Mat GIxgwn = GIP.cross(Gwn);
        double normGIxgwn = cv::norm(GIxgwn);
        cv::Mat VhatP = GIxgwn/normGIxgwn;
        double ThetaP= std::atan2(normGIxgwn,GIP.dot(Gwn));
        // Debug log
        //cout<<"vhat: "<<vhat<<", theta: "<<theta*180.0/M_PI<<endl;

        Eigen::Vector3d VhateigP = Converter::toVector3d(VhatP);
        Eigen::Matrix3d RWIeigP = Sophus::SO3::exp(VhateigP*ThetaP).matrix();
        Eigen::Matrix3d RGWeigP = RWIeigP.transpose();
        
        RwiP = Converter::toCvMat(RWIeigP);
        Best_scale = Optimizer::OptimizeInitialScale(vScaleGravityKF,RGWeigP,mpTracker->ini_depth,AVG_Scale);
        

        cv::Mat G_IP = GIP*ConfigParam::GetG();//9.8012;
        // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector

        cv::Mat CP = cv::Mat::zeros(3*(N-2),5,CV_32F);
        cv::Mat DP = cv::Mat::zeros(3*(N-2),1,CV_32F);

        for(int i=0; i<N-2; i++)
        {
            KeyFrame* pKF1 = vScaleGravityKF[i];
            KeyFrame* pKF2 = vScaleGravityKF[i+1];
            KeyFrame* pKF3 = vScaleGravityKF[i+2];
            // Delta time between frames
            double dt12 = pKF2->GetIMUPreInt().getDeltaTime();
            double dt23 = pKF3->GetIMUPreInt().getDeltaTime();
            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(pKF2->GetIMUPreInt().getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(pKF3->GetIMUPreInt().getDeltaP());
            cv::Mat Jpba12 = Converter::toCvMat(pKF2->GetIMUPreInt().getJPBiasa());
            cv::Mat Jvba12 = Converter::toCvMat(pKF2->GetIMUPreInt().getJVBiasa());
            cv::Mat Jpba23 = Converter::toCvMat(pKF3->GetIMUPreInt().getJPBiasa());
            // Pose of camera in world frame
            cv::Mat Twc1 = pKF1->GetPoseInverse();
            cv::Mat Twc2 = pKF2->GetPoseInverse();
            cv::Mat Twc3 = pKF3->GetPoseInverse();
            // Position of camera center
            cv::Mat pc1 = Twc1.rowRange(0,3).col(3);
            cv::Mat pc2 = Twc2.rowRange(0,3).col(3);
            cv::Mat pc3 = Twc3.rowRange(0,3).col(3);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = Twc1.rowRange(0,3).colRange(0,3);
            cv::Mat Rc2 = Twc2.rowRange(0,3).colRange(0,3);
            cv::Mat Rc3 = Twc3.rowRange(0,3).colRange(0,3);
            // Stack to C/D matrix
            // lambda*s + phi*dthetaxy + zeta*ba = psi
            cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
            cv::Mat phi = - 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*RwiP*SkewSymmetricMatrix(G_IP);  // note: this has a '-', different to paper
            cv::Mat zeta = Rc2*Rcb*Jpba23*dt12 + Rc1*Rcb*Jvba12*dt12*dt23 - Rc1*Rcb*Jpba12*dt23;
            cv::Mat psi = (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - (Rc2-Rc3)*pcb*dt12
                        - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt23*dt12 - 0.5*RwiP*G_IP*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper
            psi = psi- Best_scale*lambda;
            
            //lambda.copyTo(CP.rowRange(3*i+0,3*i+3).col(0));
            phi.colRange(0,2).copyTo(CP.rowRange(3*i+0,3*i+3).colRange(0,2)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
            zeta.copyTo(CP.rowRange(3*i+0,3*i+3).colRange(2,5));
            psi.copyTo(DP.rowRange(3*i+0,3*i+3));

            // Debug log
            //cout<<"iter "<<i<<endl;
        }

        // Use svd to compute C*x=D, x=[s,dthetaxy,ba] 6x1 vector
        // C = u*w*vt, u*w*vt*x=D
        // Then x = vt'*winv*u'*D
        // Note w2 is 6x1 vector by SVDecomp()
        // C is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
        cv::SVDecomp(CP,w2P,u2P,vt2P,cv::SVD::MODIFY_A);
        // Debug log
        //cout<<"u2:"<<endl<<u2<<endl;
        //cout<<"vt2:"<<endl<<vt2<<endl;
        //cout<<"w2:"<<endl<<w2<<endl;

        // Compute winv
        cv::Mat w2invP=cv::Mat::eye(5,5,CV_32F);
        for(int i=0;i<5;i++)
        {
            if(fabs(w2P.at<float>(i))<1e-10)
            {
                w2P.at<float>(i) += 1e-10;
                // Test log
                cerr<<"w2(i) < 1e-10, i is: "<<i+1<<endl<<"w is: "<<endl<<w2P<<endl;
            }

            w2invP.at<float>(i,i) = 1./w2P.at<float>(i);
        }
        // Then y = vt'*winv*u'*D
        cv::Mat yP = vt2P.t()*w2invP*u2P.t()*DP;

        //double delta_s = yP.at<float>(0);
        cv::Mat dthetaxyP = yP.rowRange(0,2);
        dbiasa_P= yP.rowRange(2,5);
        dbiasa_eigP = Converter::toVector3d(dbiasa_P);

        // dtheta = [dx;dy;0]
        cv::Mat dthetaP = cv::Mat::zeros(3,1,CV_32F);
        dthetaxyP.copyTo(dthetaP.rowRange(0,2));
        Eigen::Vector3d dthetaeigP = Converter::toVector3d(dthetaP);
        Eigen::Matrix3d Rwieig_P = RWIeigP*Sophus::SO3::exp(dthetaeigP).matrix();
        Rwi_P = Converter::toCvMat(Rwieig_P);
        //double Best_scale_final = Best_scale+delta_s;
        //cout<<"Best Scale final :"<<Best_scale_final<<endl;
        cv::Mat gwa_finalP = Rwi_P*GI;
        Eigen::Matrix3d Rwieig_PT = Rwieig_P.transpose();
        double Best_scale2=Best_scale;

        t2 =ros::Time::now();
        //cout<<"g_final VIP: "<<gwa_finalP.t()<<", |g_final|="<<cv::norm(gwa_finalP)<<endl;
        //cout<<"Time: "<<mpCurrentKeyFrame->mTimeStamp - mnStartTime<<", Scale VIP: "<<Best_scale<<endl; 
    }

    //....................................... implement optimization based initialization method here..................................................... 

    // Debug log for both result
    if (false) // if you want to write Init results
    {

        // Debug log.
        if (Init_mode==1)
        {
            cv::Mat gwbefore = Rwi*GI;
            cv::Mat gwafter = Rwi_*GI;
            
            fgw<<mpCurrentKeyFrame->mTimeStamp<<" "
            <<gwbefore.at<float>(0)<<" "<<gwbefore.at<float>(1)<<" "<<gwbefore.at<float>(2)<<" "
            <<gwafter.at<float>(0)<<" "<<gwafter.at<float>(1)<<" "<<gwafter.at<float>(2)<<" "
            <<endl;
            fscale<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<sstar<<" "<<s_<<" "<<endl;
            fbiasa<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<dbiasa_.at<float>(0)<<" "<<dbiasa_.at<float>(1)<<" "<<dbiasa_.at<float>(2)<<" "<<endl;
            fcondnum<<mpCurrentKeyFrame->mTimeStamp<<" "
                    <<w2.at<float>(0)<<" "<<w2.at<float>(1)<<" "<<w2.at<float>(2)<<" "<<w2.at<float>(3)<<" "
                    <<w2.at<float>(4)<<" "<<w2.at<float>(5)<<" "<<endl;
            fbiasg<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<endl;
        }

        if (Init_mode==2)
        {
            cv::Mat G0 = RwiP*GI;
            cv::Mat G1 = Rwi_P*GI;
            
            fgw<<mpCurrentKeyFrame->mTimeStamp<<" "
            <<G0.at<float>(0)<<" "<<G0.at<float>(1)<<" "<<G0.at<float>(2)<<"Scale "
            <<G1.at<float>(0)<<" "<<G1.at<float>(1)<<" "<<G1.at<float>(2)<<"Best_scale"
            <<endl;
            fscale<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<Best_scale<<" "<<AVG_Scale<<" "<<(Best_scale+AVG_Scale)/2<<" "<<endl;
            fbiasa<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<dbiasa_P.at<float>(0)<<" "<<dbiasa_P.at<float>(1)<<" "<<dbiasa_P.at<float>(2)<<" "<<" "<<endl;
            fcondnum<<mpCurrentKeyFrame->mTimeStamp<<" "
                    <<w2P.at<float>(0)<<" "<<w2P.at<float>(1)<<" "<<w2P.at<float>(2)<<" "<<w2P.at<float>(3)<<" "
                    <<w2P.at<float>(4)<<" "<<endl;
            fbiasg<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<endl;
        }
        if (Init_mode==3)
        {
            cv::Mat G0 = Rwi_*GI;// g final from IMU original
            cv::Mat G1 = RwiP*GI;// g from IMU averaging and refining for magnitute
            cv::Mat G2 = Rwi_P*GI;// g final IMU &  P initialization
            //cout<<"Time: "<<mpCurrentKeyFrame->mTimeStamp - mnStartTime<<", sstar: "<<sstar<<", s: "<<s_<<endl;
            // saving for Gravity 1: G  from IMU only  2: IMU average+refine 3: Final IIMU++ Pressure
            fgw<<mpCurrentKeyFrame->mTimeStamp<<" "
            <<G0.at<float>(0)<<" "<<G0.at<float>(1)<<" "<<G0.at<float>(2)<<" "
            <<G1.at<float>(0)<<" "<<G1.at<float>(1)<<" "<<G1.at<float>(2)<<" "
            <<G2.at<float>(0)<<" "<<G2.at<float>(1)<<" "<<G2.at<float>(2)<<" "
            <<endl;
            fscale<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<s_<<" "<<Best_scale<<" "<<AVG_Scale<<" "<<(Best_scale+AVG_Scale)/2<<" "<<endl;
            fbiasa<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<dbiasa_.at<float>(0)<<" "<<dbiasa_.at<float>(1)<<" "<<dbiasa_.at<float>(2)<<" "<<
                dbiasa_P.at<float>(0)<<" "<<dbiasa_P.at<float>(1)<<" "<<dbiasa_P.at<float>(2)<<" "<<endl;
            fcondnum<<mpCurrentKeyFrame->mTimeStamp<<" "
                    <<w2P.at<float>(0)<<" "<<w2P.at<float>(1)<<" "<<w2P.at<float>(2)<<" "<<w2P.at<float>(3)<<" "
                    <<w2P.at<float>(4)<<" "<<w2.at<float>(0)<<" "<<w2.at<float>(1)<<" "<<w2.at<float>(2)<<" "<<w2.at<float>(3)<<" "
                    <<w2.at<float>(4)<<w2.at<float>(5)<<" "<<endl;
            ftime<<mpCurrentKeyFrame->mTimeStamp<<" "<<(t1-t0)<<" "<<(t2-t0)<<" "<<endl;
            fbiasg<<mpCurrentKeyFrame->mTimeStamp<<" "
                <<bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<endl;
        }
        
    }
    
        

    //.......................................................................
    bool bVIOInited = false;
    if(mbFirstTry)
    {
        mbFirstTry = false;
        mnStartTime = mpCurrentKeyFrame->mTimeStamp;
    }
    if(mpCurrentKeyFrame->mTimeStamp - mnStartTime >= ini_time_limit)// ...............................TIME For Scale DETECTION...................................................
    {
        bVIOInited = true;
    }

    if(!bVIOInited)
    {
        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF = *vit;
            pKF->SetNavStateBiasGyr(Vector3d::Zero());
            pKF->SetNavStateBiasAcc(Vector3d::Zero());
            pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
            pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
        }
        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF = *vit;
            pKF->ComputePreInt();
        }
    }
    else
    {
        double scale =0.0;
        cv::Mat gw=cv::Mat::zeros(3,1,CV_32F);
        Vector3d gweig =  Vector3d::Zero();
        Vector3d bias_a = Vector3d::Zero();
        cv::Mat dbiasAcc;
        Eigen::Vector3d dbiasAcc_eig;

                
        if (Init_mode==1){
            scale = s_;
            mnVINSInitScale = s_;
            gw = Rwi_*GI;
            mGravityVec = gw;
            mGravityRot = Rwi;// is this should be Rwi_
            gweig = Converter::toVector3d(gw);
            bias_a=dbiasa_eig;
            dbiasAcc = dbiasa_;
            dbiasAcc_eig = dbiasa_eig;
        }
        else{
            scale = Best_scale;
            mnVINSInitScale = Best_scale;
            gw = Rwi_P*GI;
            mGravityVec = gw;
            mGravityRot = RwiP;// Is this also should be Rwi_P
            gweig = Converter::toVector3d(gw);
            bias_a=dbiasa_eigP;
            dbiasAcc = dbiasa_P;
            dbiasAcc_eig = dbiasa_eigP;
            //dbiasAcc = dbiasa_;
            //dbiasAcc_eig = dbiasa_eig;
            }

        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF = *vit;
            // Position and rotation of visual SLAM
            cv::Mat wPc = pKF->GetPoseInverse().rowRange(0,3).col(3);                   // wPc
            cv::Mat Rwc = pKF->GetPoseInverse().rowRange(0,3).colRange(0,3);            // Rwc
            // Set position and rotation of navstate
            cv::Mat wPb = scale*wPc + Rwc*pcb;
            pKF->SetNavStatePos(Converter::toVector3d(wPb));
            pKF->SetNavStateRot(Converter::toMatrix3d(Rwc*Rcb));
            // Update bias of Gyr & Acc
            pKF->SetNavStateBiasGyr(bgest);
            pKF->SetNavStateBiasAcc(bias_a);
            // Set delta_bias to zero. (only updated during optimization)
            pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
            pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
            // Step 4.
            // compute velocity
            if(pKF != vScaleGravityKF.back())
            {
                KeyFrame* pKFnext = pKF->GetNextKeyFrame();
                // IMU pre-int between pKF ~ pKFnext
                const IMUPreintegrator& imupreint = pKFnext->GetIMUPreInt();
                // Time from this(pKF) to next(pKFnext)
                double dt = imupreint.getDeltaTime();                                       // deltaTime
                cv::Mat dp = Converter::toCvMat(imupreint.getDeltaP());       // deltaP
                cv::Mat Jpba = Converter::toCvMat(imupreint.getJPBiasa());    // J_deltaP_biasa
                cv::Mat wPcnext = pKFnext->GetPoseInverse().rowRange(0,3).col(3);           // wPc next
                cv::Mat Rwcnext = pKFnext->GetPoseInverse().rowRange(0,3).colRange(0,3);    // Rwc next

                cv::Mat vel = - 1./dt*( scale*(wPc - wPcnext) + (Rwc - Rwcnext)*pcb + Rwc*Rcb*(dp + Jpba*dbiasAcc) + 0.5*gw*dt*dt );
                Eigen::Vector3d veleig = Converter::toVector3d(vel);
                pKF->SetNavStateVel(veleig);
            }
            else
            {
                // If this is the last KeyFrame, no 'next' KeyFrame exists
                KeyFrame* pKFprev = pKF->GetPrevKeyFrame();
                const IMUPreintegrator& imupreint_prev_cur = pKF->GetIMUPreInt();
                double dt = imupreint_prev_cur.getDeltaTime();
                Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
                //
                Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                Eigen::Vector3d veleig = velpre + gweig*dt + rotpre*( dv + Jvba*dbiasAcc_eig );
                pKF->SetNavStateVel(veleig);
            }
        }

        // Re-compute IMU pre-integration at last.
        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF = *vit;
            pKF->ComputePreInt();
        }
    }

    return bVIOInited;
}

void LocalMapping::AddToLocalWindow(KeyFrame* pKF)
{
    mlLocalKeyFrames.push_back(pKF);
    if(mlLocalKeyFrames.size() > mnLocalWindowSize)
    {
        mlLocalKeyFrames.pop_front();
    }
}

void LocalMapping::DeleteBadInLocalWindow(void)
{
    std::list<KeyFrame*>::iterator lit = mlLocalKeyFrames.begin();
    while(lit != mlLocalKeyFrames.end())
    {
        KeyFrame* pKF = *lit;
        //Test log
        if(!pKF) cout<<"pKF null?"<<endl;
        if(pKF->isBad())
        {
            lit = mlLocalKeyFrames.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}
///...............................................
   
int LocalMapping::ClusterId=0;

LocalMapping::LocalMapping(Map *pMap,ConfigParam* pParams,string strSettingPath):
    mbResetRequested(false), mpMap(pMap),  mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbAcceptKeyFrames(true),
    mbFinished(true)
{
    mpParams = pParams;
    mnLocalWindowSize = ConfigParam::GetLocalWindowSize();
    cout<<"mnLocalWindowSize:"<<mnLocalWindowSize<<endl;

    mbVINSInited = false;
    mbFirstTry = true;
    mbFirstVINSInited = false;
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    int ELC = fSettings["LoopC"];
    Loop_Closer = ELC;
    ini_time_limit = fSettings["time.Init"];
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::SetLoopCloserHaloc(LoopClosingHaloc* ploop_closing_haloc)
{
    mploop_closing_haloc = ploop_closing_haloc;
}

void LocalMapping::Run()
{
    mbFinished = false;
    // find a way to insert the image size
    ros::Rate r(500);
    while(ros::ok())
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {            
            // Tracking will see that Local Mapping is busy
            SetAcceptKeyFrames(false);

            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();// update mappoints statistic, update like previous keyframe linkages

            // Check recent MapPoints
            MapPointCulling();// removal of mappoints, set BAD flag// We have removed this for euroc dataset

            // Triangulate new MapPoints
            CreateNewMapPoints();
            //CreateNewMapPointsEdited(); // DO LATER

            // Find more matches in neighbor keyframes and fuse point duplications
            SearchInNeighbors();

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {   
                if(mpMap->KeyFramesInMap()>2 && mpTracker->mState== WORKING){
                // Local BA
                    if(!GetVINSInited() || mpTracker->mbRelocBiasPrepare){
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA,mpMap,this);
                    }
                    else
                    {
                        Optimizer::LocalBundleAdjustmentNavState(mpCurrentKeyFrame,mlLocalKeyFrames,&mbAbortBA, mpMap, mGravityVec,mGravityRot,this,mpTracker->ini_depth,mpTracker->depth_cov,fixedKF_id);
                        //Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA,mpMap,this);
                    }
                    // Check redundant local Keyframes
                    //KeyFrameCulling();
                                        
                }
                
                if(!GetVINSInited())
                {   
                    bool tmpbool=false;
                    if (mpTracker->mMode!=0){
                       tmpbool = TryInitVIO(); 
                    }
                    if(tmpbool)
                    {
                        boost::mutex::scoped_lock lock(mpMap->mMutexMapUpdate);
                        SetVINSInited(tmpbool);
                        //cout<<"gravity Rot "<<mGravityRot<<endl;
                        //cout<<"gravity Vec "<<mGravityVec<<endl;
                        mpMap->UpdateScale(mnVINSInitScale, mGravityRot);// this will scal and rotate the map to gravity axis.
                        mGravityRot = cv::Mat::eye(3,3,CV_32F);     //Changing gravity after rotating the map
                        mGravityVec.at<float>(2,0)=9.81;
                        mGravityVec.at<float>(1,0)=0.0;
                        mGravityVec.at<float>(0,0)=0.0;
                        //cout<<"gravity Rot "<<mGravityRot<<endl;
                        //cout<<"gravity Vec "<<mGravityVec<<endl;
                        SetFirstVINSInited(true);
                        std::cout<<std::endl<<"... Map scale updated ..."<<std::endl<<std::endl;
                    }
                    
                }
                mpMap->SetFlagAfterBA();
                SetMapUpdateFlagInTracking(true);
                if(!CheckNewKeyFrames()){
                    SetAcceptKeyFrames(true);
                }
            } 
            mpLastKeyFrame=mpCurrentKeyFrame;  
            if(Loop_Closer){
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame); //dissable Loop Closer
            }
        }

        // Safe area to stop
        if(stopRequested())
        {
            Stop();
            ros::Rate r2(1000);
            while(isStopped() && ros::ok())
            {
                r2.sleep();
            }     
        }

        ResetIfRequested();
        r.sleep();
    }
    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
    SetAcceptKeyFrames(false);
}


bool LocalMapping::CheckNewKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{

    {
        boost::mutex::scoped_lock lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    //generate Clusters
    if (mpCurrentKeyFrame->mnId !=0){
        mpCurrentKeyFrame->regionClustering();

        //get Clusters

        vector< vector<int> > clusters = mpCurrentKeyFrame->getClusters();

        // Loop of frame clusters
        vector<int> vertex_ids;
        vector<Cluster> clusters_to_close_loop;
        vector<Eigen::Vector4f> cluster_centroids = mpCurrentKeyFrame->getClusterCentroids();
        vector<MapPoint*> points = mpCurrentKeyFrame->getmvpMappoints();
        vector<cv::KeyPoint> kp = mpCurrentKeyFrame->GetKeyPoints();
        cv::Mat camera_pose = mpCurrentKeyFrame->GetPose();//  originally this was tf::transform
        cv::Mat orb_desc = mpCurrentKeyFrame->GetDescriptors();
        
        for (uint i=0; i<clusters.size(); i++)
        {
        
        initial_cluster_pose_history_.push_back(cluster_centroids[i]);

        // Add cluster to the graph
        int id = ClusterId;
        ClusterId++;
        
        
        // Store information
        cluster_frame_relation_.push_back(make_pair(id,mpCurrentKeyFrame));
        //local_cluster_poses_.push_back( Tools::vector4fToTransform(cluster_centroids[i]) );
        vertex_ids.push_back(id);

        // Build cluster
        cv::Mat c_desc_orb;
        vector<cv::KeyPoint> c_kp;
        vector<MapPoint*> c_points;
        for (uint j=0; j<clusters[i].size(); j++)
        {
            int idx = clusters[i][j];
            c_kp.push_back(kp[idx]);
            c_points.push_back(points[idx]);
            c_desc_orb.push_back(orb_desc.row(idx));
        }
        Cluster cluster(id, mpCurrentKeyFrame->mnId,camera_pose, c_kp, c_desc_orb, c_points);

        clusters_to_close_loop.push_back(cluster);
        }

        // Send the new clusters to the loop closing thread
        for (uint i=0; i<clusters_to_close_loop.size(); i++){
        // mploop_closing_haloc->addClusterToQueue(clusters_to_close_loop[i]);
            mpCurrentKeyFrame->clusters_inKF.push_back(clusters_to_close_loop[i]);
        }

    }
        
    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();
    mpCurrentKeyFrame->ComputeHaloc(&haloc);

    if(mpCurrentKeyFrame->mnId==0)// skip for the first (0)th key frame
        return;

    // Associate MapPoints to the new keyframe and update normal and descriptor
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    if(mpCurrentKeyFrame->mnId>1) //This operations are already done in the tracking for the first two keyframes
    {
        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            MapPoint* pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();// modify this function for speed
                }
            }
        }
    }

    if(mpCurrentKeyFrame->mnId==1)
    {
        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            MapPoint* pMP = vpMapPointMatches[i];
            if(pMP)
            {
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }  

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();//makes the connected keyframes by weight(number of common mappoints), and by order
    AddToLocalWindow(mpCurrentKeyFrame);
    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((nCurrentKFid-pMP->mnFirstKFid)>=3 && pMP->Observations()<=2)// this limits changed originally 2,2
        {

            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((nCurrentKFid-pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    int Count_pt=0;
    // Take neighbor keyframes in covisibility graph
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(20);// mostly gives last 20 keyframes
    //vector<KeyFrame*> vpNeighKFs = mpTracker->mLast20KF;
    if(vpNeighKFs.size()<20)
    {
        vpNeighKFs.clear();
        vpNeighKFs= mpTracker->mLast20KF;
    }
    ORBmatcher matcher(1,false);// 0.6 originally

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float fx1 = mpCurrentKeyFrame->fx;
    const float fy1 = mpCurrentKeyFrame->fy;
    const float cx1 = mpCurrentKeyFrame->cx;
    const float cy1 = mpCurrentKeyFrame->cy;
    const float invfx1 = 1.0f/fx1;
    const float invfy1 = 1.0f/fy1;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->GetScaleFactor();

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // Small translation errors for short baseline keyframes make scale to diverge
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);
        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepth = baseline/medianDepthKF2;

        if(ratioBaselineDepth<0.01)
            continue;

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);// this is calculated from poses, Not map points.

        // Search matches that fulfil epipolar constraint
        vector<cv::KeyPoint> vMatchedKeysUn1;
        vector<cv::KeyPoint> vMatchedKeysUn2;
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedKeysUn1,vMatchedKeysUn2,vMatchedIndices);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float fx2 = pKF2->fx;
        const float fy2 = pKF2->fy;
        const float cx2 = pKF2->cx;
        const float cy2 = pKF2->cy;
        const float invfx2 = 1.0f/fx2;
        const float invfy2 = 1.0f/fy2;

        // Triangulate each match
        for(size_t ikp=0, iendkp=vMatchedKeysUn1.size(); ikp<iendkp; ikp++)
        {
            const int idx1 = vMatchedIndices[ikp].first;
            const int idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = vMatchedKeysUn1[ikp];
            const cv::KeyPoint &kp2 = vMatchedKeysUn2[ikp];

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0 );
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0 );
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            if(cosParallaxRays<0 || cosParallaxRays>0.9998)
                continue;

            // Linear Triangulation Method
            cv::Mat A(4,4,CV_32F);
            A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
            A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
            A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

            cv::Mat w,u,vt;
            cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            cv::Mat x3D = vt.row(3).t();

            if(x3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
            x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            float sigmaSquare1 = mpCurrentKeyFrame->GetSigma2(kp1.octave);
            float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            float invz1 = 1.0/z1;
            float u1 = fx1*x1*invz1+cx1;
            float v1 = fy1*y1*invz1+cy1;
            float errX1 = u1 - kp1.pt.x;
            float errY1 = v1 - kp1.pt.y;
            if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                continue;

            //Check reprojection error in second keyframe
            float sigmaSquare2 = pKF2->GetSigma2(kp2.octave);
            float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            float invz2 = 1.0/z2;
            float u2 = fx2*x2*invz2+cx2;
            float v2 = fy2*y2*invz2+cy2;
            float errX2 = u2 - kp2.pt.x;
            float errY2 = v2 - kp2.pt.y;
            if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                continue;

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            float ratioDist = dist1/dist2;
            float ratioOctave = mpCurrentKeyFrame->GetScaleFactor(kp1.octave)/pKF2->GetScaleFactor(kp2.octave);
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(pKF2,idx2);
            pMP->AddObservation(mpCurrentKeyFrame,idx1);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
            Count_pt++;

        }
    }
    //cout<< "Nummber of New Map Points....................:"<< Count_pt<<endl<<endl;
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(20);
    vector<KeyFrame*> vpTargetKFs;// fill the neighber KFs
    for(vector<KeyFrame*>::iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher(0.6);
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    mbStopRequested = true;
    boost::mutex::scoped_lock lock2(mMutexNewKFs);
    mbAbortBA = true;
}

void LocalMapping::Stop()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isStopped()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    boost::mutex::scoped_lock lock2(mMutexFinish);
    
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();
}

bool LocalMapping::AcceptKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    boost::mutex::scoped_lock lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    nMPs++;
                    if(pMP->Observations()>3)
                    {
                        int scaleLevel = pKF->GetKeyPointUn(i).octave;
                        map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            int scaleLeveli = pKFi->GetKeyPointUn(mit->second).octave;
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=3)
                                    break;
                            }
                        }
                        if(nObs>=3)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                                  v.at<float>(2),               0,-v.at<float>(0),
                                 -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbResetRequested = true;
    }

    ros::Rate r(500);
    while(ros::ok())
    {
        {
        boost::mutex::scoped_lock lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        r.sleep();
    }
}

void LocalMapping::ResetIfRequested()
{
    boost::mutex::scoped_lock lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
        mlLocalKeyFrames.clear();

        // Add resetting init flags
        mbVINSInited = false;
        mbFirstTry = true;
    }
}

void LocalMapping::clearLocalKF()
{
    boost::mutex::scoped_lock lock(mMutexReset);
    mlpRecentAddedMapPoints.clear();
    mlLocalKeyFrames.clear();    
}

KeyFrame* LocalMapping::searchKF_loop_closer(int cluster_id){
    KeyFrame* frame_ptr = nullptr;
    for (uint i=0; i<cluster_frame_relation_.size(); i++)
    {
      if (cluster_frame_relation_[i].first == cluster_id)
      {
        frame_ptr = cluster_frame_relation_[i].second;
        break;
      }
    }
    return frame_ptr;
}

void LocalMapping::getCandidates_Proximity(int cluster_id, int window_center, int window, int best_n, vector<int> &neighbors, vector<int> no_candidates){

    // Init
    neighbors.clear();
    Eigen::Vector4f vertex_pose= initial_cluster_pose_history_[cluster_id];
    
    // Loop thought all the other nodes
    vector< pair< int,double > > neighbor_distances;
    for (uint i=0; i<initial_cluster_pose_history_.size(); i++)
    {
      if ( (int)i == cluster_id ) continue;
      if ((int)i > window_center-window && (int)i < window_center+window) continue;
      if (find(no_candidates.begin(), no_candidates.end(), (int)i) != no_candidates.end())
            continue;

      // Get the node pose
        Eigen::Vector4f cur_pose = initial_cluster_pose_history_[i];
        double dist = LocalMapping::poseDiff2D(cur_pose, vertex_pose);
        neighbor_distances.push_back(make_pair(i, dist));
    }

    // Exit if no neighbors
    if (neighbor_distances.size() == 0) return;

    // Sort the neighbors
    sort(neighbor_distances.begin(), neighbor_distances.end(), LocalMapping::sortByDistance);

    // Min number
    if ((int)neighbor_distances.size() < best_n){
      best_n = neighbor_distances.size();
    }

    for (int i=0; i<=best_n; i++){  
     neighbors.push_back(neighbor_distances[i].first);
    }
}

double LocalMapping::poseDiff2D(Eigen::Vector4f pose_1, Eigen::Vector4f pose_2){
    Eigen::Vector4f d = pose_1 - pose_2;
    return sqrt(d(0)*d(0) + d(1)*d(1));
}

bool LocalMapping::sortByDistance(const pair<int, double> d1, const pair<int, double> d2)
{
   return (d1.second < d2.second);
}

void LocalMapping::RequestFinish()
{
    boost::mutex::scoped_lock lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    boost::mutex::scoped_lock lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    boost::mutex::scoped_lock lock(mMutexFinish);
    mbFinished = true;  
    boost::mutex::scoped_lock lock2(mMutexStop);  
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    boost::mutex::scoped_lock lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::CreateNewMapPointsEdited()
{   
    if(mpCurrentKeyFrame->mnId<3){return;}
    int Count_pt=0;
    vector<int> idx;
    vector<cv::Point2d> triangulation_points1, triangulation_points2;
    vector<MapPoint*> points2 = mpCurrentKeyFrame->getmvpMappoints();
    vector<MapPoint*> points1 = mpLastKeyFrame->getmvpMappoints();
    vector<int> priKF_idx= mpCurrentKeyFrame->mvIniMatches;

    for(size_t i=0; i<points2.size(); i++){
        if(mpCurrentKeyFrame->mvIniMatches[i]<0){continue;}
        if(mpCurrentKeyFrame->mvIniMatches[i]==2000){continue;}
        if(points2[i]){continue;}
        if(points1[priKF_idx[i]]){
            cout<<"Map Point exsist, it should not"<<endl;
            continue;
        }
        cv::KeyPoint pt1=mpCurrentKeyFrame->mvKeysUn[i];
        cv::KeyPoint pt2=mpLastKeyFrame->mvKeysUn[priKF_idx[i]];
        triangulation_points1.push_back(cv::Point2d((double)pt1.pt.x,(double)pt1.pt.y));
        triangulation_points2.push_back(cv::Point2d((double)pt2.pt.x,(double)pt2.pt.y));
        idx.push_back(i);
    }

    cv::Mat tcw1 = mpLastKeyFrame->GetPose();
    cv::Mat Tcw1 = tcw1.rowRange(0,3).colRange(0,4);
    cv::Mat tcw2 = mpCurrentKeyFrame->GetPose();
    cv::Mat Tcw2 = tcw1.rowRange(0,3).colRange(0,4);
    cv::Mat point3d_homo;
    cv::triangulatePoints(Tcw1, Tcw2,triangulation_points1, triangulation_points2,point3d_homo);
    assert(point3d_homo.cols == triangulation_points1.size());
    point3d_homo.convertTo(point3d_homo, CV_32F);
    for(int i = 0; i < point3d_homo.cols; i++) {
        
        cv::Mat x3D = point3d_homo.col(i);
        cv::Mat p3d;
        p3d = x3D.rowRange(0,3)/x3D.at<float>(3); 
        //cv::convertPointsFromHomogeneous(x3D.t(), p3d);
        MapPoint* pMP = new MapPoint(p3d,mpCurrentKeyFrame,mpMap);  
        pMP->AddObservation(mpLastKeyFrame,priKF_idx[idx[i]]); 
        pMP->AddObservation(mpCurrentKeyFrame,idx[i]); 
        mpLastKeyFrame->AddMapPoint(pMP,priKF_idx[idx[i]]);
        mpCurrentKeyFrame->AddMapPoint(pMP,idx[i]);
        cout<< "index are: "<<priKF_idx[idx[i]]<<" "<<idx[i]<<endl;
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pMP);
        mlpRecentAddedMapPoints.push_back(pMP);
        Count_pt++;
    }

    cout<<"new triangulated Point :"<<Count_pt<<endl;

}

} //namespace USLAM
