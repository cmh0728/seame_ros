#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <cstdlib>
#include <tuple>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Dense> // Eigen Library : 선형대수 템플릿 라이브러리 
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <optional>
#include <functional>
#include <memory>
#include <iomanip>


#include <limits>  // std::numeric_limits

#include "opencv2/opencv.hpp"    
#include <yaml-cpp/yaml.h>   



// 제어 최적화 라이브러리 casadi
// #include "casadi/casadi.hpp" 
// #include "casadi/core/sparsity_interface.hpp"
// #include <mlpack.hpp> // 머신러닝 라이브러리 통합 헤더 
// #include "Integration/GNSSInfo.h"
// #include "Integration/object_msg_arr.h"





#define NONE -1e9

typedef unsigned char uint8_t;
typedef unsigned short uint16_t; 
typedef unsigned int uint32_t;
typedef short int16_t;
typedef float float32_t;
typedef double float64_t;

using namespace std;
using namespace Eigen;
using namespace chrono;
using namespace cv;


const int32_t c_PARSING_UDP = 0;
const int32_t c_PARSING_ROS = 1;

const int32_t c_CAMERA_BUFFER_SIZE = 2000000;
const int32_t c_IMU_BUFFER_SIZE = 100;
const int32_t c_MAX_FRAME_SIZE = 5000;


const int32_t c_STATE_MODE_LOGGING_ON = 0;
const int32_t c_STATE_MODE_LOGGING_OFF = 1;
const int32_t c_STATE_MODE_PLAY = 2;
const int32_t c_STATE_MODE_STOP = 3; 
const int32_t c_STATE_MODE_NULL = 4;

const int32_t c_STATE_1XZOOMFACTOR = 5;
const int32_t c_STATE_2XZOOMFACTOR = 6;
const int32_t c_STATE_3XZOOMFACTOR = 7;

const int32_t c_TOTAL_POINT_NUM = 240000;

const uint8_t c_POINT_INVALID = 0;
const uint8_t c_POINT_VALID = 255;
const uint8_t c_POINT_GROUND = 128;
const uint8_t c_CLUSTER_CAR = 255;
const uint8_t c_CLUSTER_NONCAR = 0;

const float32_t c_LEAF_SIZE = 1.f;

const float32_t c_DISTANCE_LEAF_SIZE = 1.f;
const float32_t c_AZIMUTH_LEAF_SIZE = 1.f;
const float32_t c_ELEVATION_LEAF_SIZE = 2.5f;

const int32_t c_VOXEL_DISTANCE_MIN = 0;
const int32_t c_VOXEL_DISTANCE_MAX = 150;
const int32_t c_VOXEL_AZIMUTH_MIN = 0;
const int32_t c_VOXEL_AZIMUTH_MAX = 360;
const int32_t c_VOXEL_ELEVATION_MIN = -15;
const int32_t c_VOXEL_ELEVATION_MAX = 5;

const uint32_t c_GRID_DISTANCE_SIZE = (c_VOXEL_DISTANCE_MAX - c_VOXEL_DISTANCE_MIN) / c_DISTANCE_LEAF_SIZE;
const uint32_t c_GRID_AZIMUTH_SIZE = (c_VOXEL_AZIMUTH_MAX - c_VOXEL_AZIMUTH_MIN) / c_AZIMUTH_LEAF_SIZE;
const uint32_t c_GRID_ELEVATION_SIZE = (c_VOXEL_ELEVATION_MAX - c_VOXEL_ELEVATION_MIN) / c_ELEVATION_LEAF_SIZE;


const int32_t c_UNCLUSTERED = -1;
const int32_t c_TOTAL_CLUSTER_NUM = 500;
const int32_t c_CLUSTER_POINT_NUM = 100000;

const uint8_t c_OBJECT_STATIC = 0;
const uint8_t c_OBJECT_DYNAMIC = 1;
const uint8_t c_TOTAL_TRACKING_NUM = 32;

extern float64_t c_ORIGIN_LATITUDE_DEG;
extern float64_t c_ORIGIN_LONGITUDE_DEG;
extern float64_t c_ORIGIN_LATITUDE_RAD;
extern float64_t c_ORIGIN_LONGITUDE_RAD;
extern float64_t c_ORIGIN_ALTITUDE;
extern float64_t c_ORIGIN_REFERENCE_X;
extern float64_t c_ORIGIN_REFERENCE_Y;
extern float64_t c_ORIGIN_REFERENCE_Z;


const float64_t c_LLA2ENU_A = 6378137.0;
const float64_t c_LLA2ENU_FLAT_RATIO = 1 / 298.257223563;
const float64_t c_LLA2ENU_N_2 = (2 * c_LLA2ENU_FLAT_RATIO) - pow(c_LLA2ENU_FLAT_RATIO, 2);


const int32_t c_PLANNING_MAX_PATH_NUM = 10000;
const int32_t c_PLANNING_MAX_SPLINE_NUM = 100000;
const int32_t c_PLANNING_MAX_FRENET_NUM = 1000;
const int32_t c_PLANNING_MAX_FRENET_PATH_NUM = 1000;


const uint8_t c_CONTROL_FLAG_NONE = 0;
const uint8_t c_CONTROL_FLAG_ACC = 1;
const uint8_t c_CONTROL_FLAG_AEB = 2;
const uint8_t c_CONTROL_FLAG_OVERTAKING = 3;

const int32_t c_CONTROL_HORIZON = 10;

//###################################### camera node #########################################//

struct RAW_CAMERA_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_CAMERA_BUFFER_SIZE];
  int32_t s32_CameraHeader;
};

struct RAW_IMU_DATA
{
  uint64_t u64_Timestamp;
  int32_t s32_Num;
  char arc_Buffer[c_IMU_BUFFER_SIZE];
  int32_t s32_IMUHeader;
};

// 차선 요소 기울기 
struct LANE_COEFFICIENT
{
  float64_t f64_Slope;
  float64_t f64_Intercept;
  bool b_IsLeft = false;
};

// 검출한 차선 정보 
struct CAMERA_LANEINFO
{
  cv::Point arst_LaneSample[40]; // 최대 40개의 차선 샘플 포인트
  int32_t s32_SampleCount;
  LANE_COEFFICIENT st_LaneCoefficient; // 차선 정보 (기울기)
  bool b_IsLeft = false;
};

struct KALMAN_STATE
{
  float64_t f64_Distance;
  float64_t f64_Angle;
  float64_t f64_DeltaDistance;
  float64_t f64_DeltaAngle;
};

// lane kalman 
struct LANE_KALMAN
{
  MatrixXf st_A = MatrixXf(4,4);          // 시스템 모델 행렬
  VectorXf st_X = VectorXf(4);            // 추정값
  VectorXf st_PrevX = VectorXf(4);        // 이전 추정값
  VectorXf st_Z = VectorXf(4);            // 측정값   
  MatrixXf st_H = Matrix4f::Identity();   // 상태 전이 행렬   
  MatrixXf st_K = MatrixXf(4, 4);         // Kalman Gain
  MatrixXf st_P = MatrixXf(4, 4);         // 오차 공분산
  MatrixXf st_Q = MatrixXf(4, 4);         // 시스템 노이즈 행렬
  MatrixXf st_R = MatrixXf(4, 4);         // 관측(센서) 노이즈 행렬
  
  bool b_InitializeFlag = false;
  bool b_MeasurementUpdateFlag = false;
  bool b_IsLeft = false;
  int32_t s32_CntNoMatching = 0;

  LANE_COEFFICIENT st_LaneCoefficient;
  KALMAN_STATE st_LaneState;
};



struct CAMERA_PARAM
{
    std::string s_IPMParameterX;
    std::string s_IPMParameterY;

    int32_t s32_MarginX;
    int32_t s32_MarginY;

    int64_t s64_IntervalX;
    int64_t s64_IntervalY;

    int32_t s32_RemapHeight;
    int32_t s32_RemapWidth;
};

// 카메라 데이터 구조체 선언 
struct CAMERA_DATA {
    uint64_t u64_Timestamp{};
    CAMERA_PARAM st_CameraParameter{};
    float32_t arf32_LaneCenterX[1000]{};
    float32_t arf32_LaneCenterY[1000]{};
    int32_t s32_LaneCenterNum{};
    LANE_KALMAN arst_KalmanObject[10]{};
    int32_t s32_KalmanObjectNum{};
    float32_t f32_LastDistanceLeft{};
    float32_t f32_LastAngleLeft{};
    float32_t f32_LastDistanceRight{};
    float32_t f32_LastAngleRight{};
    bool b_ThereIsLeft = false;
    bool b_ThereIsRight = false;

    // for sliding window px limited
    int32_t last_left_start_x  = -1;
    int32_t last_right_start_x = -1;
    bool has_last_left_start   = false;
    bool has_last_right_start  = false;


};

//###################################### IMU #########################################//

struct IMU_DATA
{
    float32_t f32_AccelX;
    float32_t f32_AccelY;
    float32_t f32_AccelZ;

    float32_t f32_GyroX;
    float32_t f32_GyroY;
    float32_t f32_GyroZ;

    float32_t f32_QuaternionX;
    float32_t f32_QuaternionY;
    float32_t f32_QuaternionZ;
    float32_t f32_QuaternionW;
    float32_t f32_Roll_rad;
    float32_t f32_Pitch_rad;
    float32_t f32_Yaw_rad;

    float32_t f32_Roll_deg;
    float32_t f32_Pitch_deg;
    float32_t f32_Yaw_deg;

};


typedef struct FILE_INFO
{
    FILE* pFile;
    string st_FileName;
    int32_t s32_CntFrame;
    int32_t s32_AllSensorBufferSize;
    uint64_t ars64_Frame[c_MAX_FRAME_SIZE];
} FILE_INFO_t;



typedef struct _PATH
{
    float32_t arf32_X[c_PLANNING_MAX_PATH_NUM];
    float32_t arf32_Y[c_PLANNING_MAX_PATH_NUM];
    float32_t arf32_D[c_PLANNING_MAX_PATH_NUM];
    int32_t s32_Num;
} PATH_t;


typedef struct _VEHICLE_DATA {

    int32_t s32_VehicleIdx;
    float32_t f32_X;
    float32_t f32_Y;
    float32_t f32_Z;
    float32_t f32_VelocityX_m_s;
    float32_t f32_VelocityY_m_s;
    float32_t f32_Velocity_m_s;
    float32_t f32_AccelX;
    float32_t f32_AccelY;
    float32_t f32_Accel;

    float32_t f32_S;
    float32_t f32_D;
    float32_t f32_DS;

    float32_t f32_Yaw_rad_ENU;
    float32_t f32_Yaw_rad_NED;

    int32_t s32_CurrentLane;
    float32_t f32_TimeToCollision;
    float32_t f32_SafetyDistance;

    bool b_IsCar = false;
    bool b_IsAhead = false;
    bool b_IsBack = false;
    bool b_IsTracked = false;

} VEHICLE_DATA_t;






typedef struct _PLANNER {

    bool b_VehicleAhead;
    bool b_VehicleLeft;
    bool b_VehicleRight;

    int32_t s32_TargetLane;

    int8_t s8_AccFlag;
    int8_t s8_AEBFlag;
    int8_t s8_OverTakingFlag;
    int8_t s8_OverTakingStart;
    int8_t s8_TargetVehicleIdx;
    int8_t s8_OverTakingTry;

    int32_t s32_AccCnt;
    int32_t s32_OverTakingCnt;

    float32_t f32_SafetyDistance;

} PLANNER_t;


typedef struct _LANE_FILTER {

    int32_t s32_MemorySize = 20;
    int32_t s32_LaneHistoryIdx;
    int32_t s32_LaneHistoryCnt;
    int32_t s32_LaneHistory[20] = {0};

} LANE_FILTER_t;



typedef struct _State
{
  // Current State
  float32_t f32_X;
  float32_t f32_Y;
  float32_t f32_Velocity_m_s;
  float32_t f32_Yaw_rad;
} State_t;


uint64_t getMillisecond();
float32_t deg2rad(float32_t f32_Degree);
float32_t rad2deg(float32_t f32_Radian);

float64_t deg2rad(float64_t f64_Degree);
float64_t rad2deg(float64_t f64_Radian);

float32_t ms2kph(float32_t f32_Speed);
float32_t kph2ms(float32_t f32_Speed);

float64_t ms2kph(float64_t f64_Speed);
float64_t kph2ms(float64_t f64_Speed);

float32_t getDistance3d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_Z1, float32_t f32_X2, float32_t f32_Y2, float32_t f32_Z2);
float64_t getDistance3d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_Z1, float64_t f64_X2, float64_t f64_Y2, float64_t f64_Z2);

float32_t getDistance2d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_X2, float32_t f32_Y2);
float64_t getDistance2d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_X2, float64_t f64_Y2);

float32_t pi2pi(float32_t f32_Angle);
float64_t pi2pi(float64_t f64_Angle);

void GetModData(float32_t min, float32_t max, float32_t &data);
void GetModData(float64_t min, float64_t max, float64_t &data);

void CalcNearestDistIdx(float64_t f64_X, float64_t f64_Y, float64_t* f64_MapX, float64_t* f64_MapY, int32_t s32_MapLength, float64_t& f64_NearDist, int32_t& s32_NearIdx);
void CalcNearestDistIdx(float32_t f32_X, float32_t f32_Y, float32_t* f32_MapX, float32_t* f32_MapY, int32_t s32_MapLength, float32_t& f32_NearDist, int32_t& s32_NearIdx);

void lla2enu(float64_t f64_Lat_deg, float64_t f64_Lon_deg, float64_t f64_Alt, float32_t &f32_E, float32_t &f32_N, float32_t &f32_U);

void rotationMatrix(float32_t roll, float32_t pitch, float32_t yaw, float32_t matrix[3][3]);


#endif
