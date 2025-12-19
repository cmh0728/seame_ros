#include "global.hpp"

extern float64_t f64_Vx, f64_Vy, f64_Ax, f64_Ay, f64_c_V, f64_c_A;

float64_t c_ORIGIN_LATITUDE_DEG = 0.0;
float64_t c_ORIGIN_LONGITUDE_DEG = 0.0;
float64_t c_ORIGIN_LATITUDE_RAD = 0.0;
float64_t c_ORIGIN_LONGITUDE_RAD = 0.0;
float64_t c_ORIGIN_ALTITUDE = 0.0;
float64_t c_ORIGIN_REFERENCE_X = 0.0;
float64_t c_ORIGIN_REFERENCE_Y = 0.0;
float64_t c_ORIGIN_REFERENCE_Z = 0.0;
int32_t s32_FileNum = 0;


// 현재 시간 밀리초 단위 반환
uint64_t getMillisecond()
{
    auto st_Now = std::chrono::steady_clock::now();
    auto st_Now_ms = time_point_cast<std::chrono::milliseconds>(st_Now);
    milliseconds st_Millisecond = duration_cast<std::chrono::milliseconds>(st_Now_ms.time_since_epoch());
    return st_Millisecond.count();
}

// degree --> radian
float32_t deg2rad(float32_t f32_Degree)
{
    return f32_Degree * M_PI / 180.f;
}

// radian --> degree
float32_t rad2deg(float32_t f32_Radian)
{
    return f32_Radian * 180.f / M_PI;
}


float64_t deg2rad(float64_t f64_Degree)
{
    return f64_Degree * M_PI / 180.f;
}


float64_t rad2deg(float64_t f64_Radian)
{
    return f64_Radian * 180.f / M_PI;
}

float32_t ms2kph(float32_t f32_Speed)
{
    return f32_Speed * 3.6f;
}

float32_t kph2ms(float32_t f32_Speed)
{
    return f32_Speed / 3.6f;
}

float64_t ms2kph(float64_t f64_Speed)
{
    return f64_Speed * 3.6;
}

float64_t kph2ms(float64_t f64_Speed)
{
    return f64_Speed / 3.6;
}

float32_t getDistance3d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_Z1, float32_t f32_X2, float32_t f32_Y2, float32_t f32_Z2)
{
    return sqrtf(powf(f32_X1 - f32_X2, 2.f) + powf(f32_Y1 - f32_Y2, 2.f) + powf(f32_Z1 - f32_Z2, 2.f));
}

float64_t getDistance3d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_Z1, float64_t f64_X2, float64_t f64_Y2, float64_t f64_Z2)
{
    return sqrtf(powf(f64_X1 - f64_X2, 2.f) + powf(f64_Y1 - f64_Y2, 2.f) + powf(f64_Z1 - f64_Z2, 2.f));
}


float32_t getDistance2d(float32_t f32_X1, float32_t f32_Y1, float32_t f32_X2, float32_t f32_Y2)
{
    return sqrtf(powf(f32_X1 - f32_X2, 2.f) + powf(f32_Y1 - f32_Y2, 2.f));
}


float64_t getDistance2d(float64_t f64_X1, float64_t f64_Y1, float64_t f64_X2, float64_t f64_Y2)
{
    return sqrtf(powf(f64_X1 - f64_X2, 2.f) + powf(f64_Y1 - f64_Y2, 2.f));
}

float32_t pi2pi(float32_t f32_Angle)
{
    if (f32_Angle > M_PI)
        f32_Angle -= 2 * M_PI;
    else if(f32_Angle < -M_PI)
        f32_Angle += 2 * M_PI;

    return f32_Angle;
}

float64_t pi2pi(float64_t f64_Angle)
{
    if (f64_Angle > M_PI)
        f64_Angle -= 2 * M_PI;
    else if(f64_Angle < -M_PI)
        f64_Angle += 2 * M_PI;

    return f64_Angle;
}

void GetModData(float32_t min, float32_t max, float32_t &data)
{
	if (data > max)
	{
		data = max;
	}
	else if (data < min)
	{
		data = min;
	}
}

void GetModData(float64_t min, float64_t max, float64_t &data)
{
	if (data > max)
	{
		data = max;
	}
	else if (data < min)
	{
		data = min;
	}
}




void CalcNearestDistIdx(float64_t f64_X, float64_t f64_Y, float64_t* f64_MapX, float64_t* f64_MapY, int32_t s32_MapLength, float64_t& f64_NearDist, int32_t& s32_NearIdx)
{
    float64_t f64_MinDist = 999999999.f;
    int32_t s32_I;
    for(s32_I = 0; s32_I < s32_MapLength; s32_I++)
    {
        float64_t f64_Dist = getDistance2d(f64_X, f64_Y, f64_MapX[s32_I], f64_MapY[s32_I]);
        if(f64_MinDist > f64_Dist)
        {
            f64_MinDist = f64_Dist; 
            f64_NearDist = f64_Dist; 
            s32_NearIdx = s32_I; 
        }
    }
}


void CalcNearestDistIdx(float32_t f32_X, float32_t f32_Y, float32_t* f32_MapX, float32_t* f32_MapY, int32_t s32_MapLength, float32_t& f32_NearDist, int32_t& s32_NearIdx)
{
    float32_t f32_MinDist = 999999999.f;
    int32_t s32_I;
    for(s32_I = 0; s32_I < s32_MapLength; s32_I++)
    {
        float32_t f32_Dist = getDistance2d(f32_X, f32_Y, f32_MapX[s32_I], f32_MapY[s32_I]);
        if(f32_MinDist > f32_Dist)
        {
            f32_MinDist = f32_Dist; 
            f32_NearDist = f32_Dist; 
            s32_NearIdx = s32_I; 
        }
    }
}


void lla2enu(float64_t f64_Lat_deg, float64_t f64_Lon_deg, float64_t f64_Alt, float32_t &f32_E, float32_t &f32_N, float32_t &f32_U)
{
    float64_t f64_Lat_rad = deg2rad(f64_Lat_deg);
    float64_t f64_Lon_rad = deg2rad(f64_Lon_deg);

    float64_t f64_Chi = sqrt(1 - c_LLA2ENU_N_2 * pow(sin(f64_Lat_rad), 2));
    float64_t f64_Q = (c_LLA2ENU_A / f64_Chi + f64_Alt) * cos(f64_Lat_rad);

    float64_t f64_X = f64_Q * cos(f64_Lon_rad);
    float64_t f64_Y = f64_Q * sin(f64_Lon_rad);
    float64_t f64_Z = ((c_LLA2ENU_A * (1 - c_LLA2ENU_N_2) / f64_Chi) + f64_Alt) * sin(f64_Lat_rad);

    float64_t f64_dX = f64_X - c_ORIGIN_REFERENCE_X;
    float64_t f64_dY = f64_Y - c_ORIGIN_REFERENCE_Y;
    float64_t f64_dZ = f64_Z - c_ORIGIN_REFERENCE_Z;

    f32_E = (float32_t)(-sin(c_ORIGIN_LONGITUDE_RAD) * f64_dX + cos(c_ORIGIN_LONGITUDE_RAD) * f64_dY);
    f32_N = (float32_t)(-sin(c_ORIGIN_LATITUDE_RAD) * cos(c_ORIGIN_LONGITUDE_RAD) * f64_dX - sin(c_ORIGIN_LATITUDE_RAD) * sin(c_ORIGIN_LONGITUDE_RAD) * f64_dY + cos(c_ORIGIN_LATITUDE_RAD) * f64_dZ);
    f32_U = (float32_t)(cos(c_ORIGIN_LATITUDE_RAD) * cos(c_ORIGIN_LONGITUDE_RAD) * f64_dX + cos(c_ORIGIN_LATITUDE_RAD) * sin(c_ORIGIN_LONGITUDE_RAD) * f64_dY + sin(c_ORIGIN_LATITUDE_RAD) * f64_dZ);
}

void rotationMatrix(float32_t f32_Roll, float32_t f32_Pitch, float32_t f32_Yaw, float32_t f32_matrix[3][3]) {
    float32_t f32_cos_roll = cos(deg2rad(f32_Roll));
    float32_t f32_sin_roll = sin(deg2rad(f32_Roll));
    float32_t f32_cos_pitch = cos(deg2rad(f32_Pitch));
    float32_t f32_sin_pitch = sin(deg2rad(f32_Pitch));
    float32_t f32_cos_yaw = cos(deg2rad(f32_Yaw));
    float32_t f32_sin_yaw = sin(deg2rad(f32_Yaw));

    f32_matrix[0][0] = f32_cos_yaw * f32_cos_pitch;
    f32_matrix[0][1] = f32_cos_yaw * f32_sin_pitch * f32_sin_roll - f32_sin_yaw * f32_cos_roll;
    f32_matrix[0][2] = f32_cos_yaw * f32_sin_pitch * f32_cos_roll + f32_sin_yaw * f32_sin_roll;
    f32_matrix[1][0] = f32_sin_yaw * f32_cos_pitch;
    f32_matrix[1][1] = f32_sin_yaw * f32_sin_pitch * f32_sin_roll + f32_cos_yaw * f32_cos_roll;
    f32_matrix[1][2] = f32_sin_yaw * f32_sin_pitch * f32_cos_roll - f32_cos_yaw * f32_sin_roll;
    f32_matrix[2][0] = -f32_sin_pitch;
    f32_matrix[2][1] = f32_cos_pitch * f32_sin_roll;
    f32_matrix[2][2] = f32_cos_pitch * f32_cos_roll;
}
