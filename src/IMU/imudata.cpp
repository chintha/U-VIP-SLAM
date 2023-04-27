#include "imudata.h"


namespace USLAM
{
//imu.gyrVar: 1.2e-3 #4e-3    # 1.2e-3
//imu.accVar: 8e-3   #2e-2    # 8.0e-3
//imu.gyrBiasVar: 4.0e-6    # 1e-9
//imu.accBiasVar: 4.0e-5    # 1e-8

// covariance of measurement
// From continuous noise_density of dataset sigma_g/sigma_a   rad/s/sqrt(Hz) -- m/s^2/sqrt(Hz)

/*
 * For EuRoc dataset, according to V1_01_easy/imu0/sensor.yaml
 * The params:
 * sigma_g: 1.6968e-4       rad / s / sqrt(Hz)
 * sigma_gw: 1.9393e-5      rad / s^2 / sqrt(Hz)
 * sigma_a: 2.0e-3          m / s^2 / sqrt(Hz)
 * sigma_aw: 3.0e-3         m / s^3 / sqrt(Hz)
 */
// noise given as noice density,
//TODO remove this noise vaues to a traking thread(in tracking as a common value)


Matrix3d IMUData::_gyrMeasCov = Matrix3d::Identity();      
Matrix3d IMUData::_accMeasCov = Matrix3d::Identity();   
double IMUData::_gyrBiasRw2 = 1;  
double IMUData::_accBiasRw2 = 1;  

// covariance of bias random walk
Matrix3d IMUData::_gyrBiasRWCov = Matrix3d::Identity()*_gyrBiasRw2;     
Matrix3d IMUData::_accBiasRWCov = Matrix3d::Identity()*_accBiasRw2;     


IMUData::IMUData(const double& gx, const double& gy, const double& gz,
                 const double& ax, const double& ay, const double& az,
                 const double& t) :
    wm(gx,gy,gz), am(ax,ay,az), timestamp(t)
{
}


//IMUData::IMUData(const IMUData& imu) :
//    _g(imu._g), _a(imu._a), _t(imu._t)
//{
//}

}
