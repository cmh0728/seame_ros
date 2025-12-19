// lane detection v2
#include "perception/CamNode.hpp"
#include "perception/msg/lane_pnt.hpp"
#include <cv_bridge/cv_bridge.h>


namespace perception
{
// IPM map 
cv::Mat st_IPMX;
cv::Mat st_IPMY;

//버퍼 미리 생성 
cv::Mat g_IpmImg;       // IPM 결과 (컬러)
cv::Mat g_TempImg;      // 그레이 + 이후 전처리
cv::Mat g_ResultImage;  // 시각화용
bool b_NoLaneLeft = false;
bool b_NoLaneRight = false;
CAMERA_LANEINFO st_LaneInfoLeftMain{};
CAMERA_LANEINFO st_LaneInfoRightMain{};

// trackbar init praram
int g_thresh      = 160;  // 이진화 임계값
int g_canny_low   = 140;  // Canny low
int g_canny_high  = 330;  // Canny high
int g_dilate_ksize = 8;   // 팽창 커널 크기

// 시각화 옵션
bool visualize = false;
bool track_bar = false;
bool cal_lane_width = false ;
static CAMERA_DATA static_camera_data;

// ransac 난수 초기화 전역설정
struct RansacRandomInit {RansacRandomInit() { std::srand(static_cast<unsigned int>(std::time(nullptr))); }} g_ransacRandomInit;

void on_trackbar(int, void*){}
static bool ComputeLaneWidthAngle(const LANE_COEFFICIENT& left,
                                  const LANE_COEFFICIENT& right,
                                  int img_height,
                                  double& width_px,
                                  double& angle_diff_deg);

//################################################## CameraProcessing class functions ##################################################//
CameraProcessing::CameraProcessing() : rclcpp::Node("CameraProcessing_node") // rclcpp node 상속 클래스 
{
    const std::string image_topic = "/camera/camera/color/image_raw/compressed";
    visualize = this->declare_parameter<bool>("visualize", false);
    cal_lane_width = this->declare_parameter<bool>("cal_lane_width", false);

    LoadParam(&static_camera_data);          // cameardata param load
    LoadMappingParam(&static_camera_data);   // cameradata IPM 맵 로드

    //img subscriber
    image_subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(image_topic, rclcpp::SensorDataQoS(),
    std::bind(&CameraProcessing::on_image, this, std::placeholders::_1));

    // lane pub 
    lane_left_pub_ = create_publisher<perception::msg::Lane>("/lane/left", rclcpp::QoS(10));
    lane_right_pub_ = create_publisher<perception::msg::Lane>("/lane/right", rclcpp::QoS(10));


    RCLCPP_INFO(get_logger(), "Perception node subscribing to %s", image_topic.c_str()); //debug msg

    if (track_bar)
    {
        // 디버그용 윈도우 + 트랙바 컨트롤 창
        cv::namedWindow("IPM");
        cv::namedWindow("Temp_Img");
        cv::namedWindow("st_ResultImage");
        cv::namedWindow("PreprocessControl");

        cv::createTrackbar("Threshold", "PreprocessControl", nullptr, 255, on_trackbar);
        cv::createTrackbar("CannyLow",  "PreprocessControl", nullptr, 500, on_trackbar);
        cv::createTrackbar("CannyHigh", "PreprocessControl", nullptr, 500, on_trackbar);
        cv::createTrackbar("DilateK",   "PreprocessControl", nullptr, 31,  on_trackbar);

    }
}

CameraProcessing::~CameraProcessing()
{
  if (!window_name_.empty()) // window있을때 
  {
    cv::destroyWindow(window_name_); // 모든 window제거 
  }
}

// sub callback function
void CameraProcessing::on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  //예외처리 
  try
  {
    cv::Mat img = cv::imdecode(msg->data, cv::IMREAD_COLOR); // cv:Mat 형식 디코딩

    // ros parameter server setting 
    visualize = get_parameter("visualize").as_bool();
    cal_lane_width = get_parameter("cal_lane_width").as_bool();

    // img processing - lane detection
    Lane_detector(img,&static_camera_data); // img processing main pipeline function

    // lane msg publish
    publish_lane_messages();
    
  }
  catch (const cv::Exception & e) // cv에러 예외처리 
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "OpenCV exception during decode: %s", e.what());
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Image callback exception: %s", e.what());
  }
}

//################################################## img processing functions ##################################################//
void Lane_detector(const cv::Mat& img_frame, CAMERA_DATA* camera_data)
{
    // kalman filter variables
    CAMERA_LANEINFO st_LaneInfoLeft, st_LaneInfoRight; // sliding window에서 검출된 차선 정보 담는 구조체 

    //왼쪽 차선 구분 
    st_LaneInfoLeft.b_IsLeft = true; // 왼쪽 차선 표시 --> 칼만 객체 생성에서 구분 
    st_LaneInfoLeft.st_LaneCoefficient.b_IsLeft = true ;
    KALMAN_STATE st_KalmanStateLeft, st_KalmanStateRight; // 새로 계산된 좌우 차선 거리 , 각도 저장 
    int32_t s32_I, s32_J, s32_KalmanStateCnt = 0;
    KALMAN_STATE arst_KalmanState[2] = {0};

    // 매 프레임마다 초기화
    b_NoLaneLeft  = false;
    b_NoLaneRight = false;

    // IPM 결과 버퍼 준비 (카메라 해상도가 바뀔 수 있으니 create 사용)
    g_IpmImg.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        img_frame.type()          // CV_8UC3
    );

    // Temp_Img (gray) 버퍼 준비
    g_TempImg.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        CV_8UC1
    );

    // 결과 이미지 버퍼 준비
    g_ResultImage.create(
        camera_data->st_CameraParameter.s32_RemapHeight,
        camera_data->st_CameraParameter.s32_RemapWidth,
        CV_8UC3
    );
    g_ResultImage.setTo(cv::Scalar(0,0,0)); // 매 프레임 초기화

    // =======================  원본 → IPM remapping =======================
    cv::remap(img_frame, g_IpmImg, st_IPMX, st_IPMY,
              cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    // =======================  IPM → Gray + Blur  =======================
    cv::cvtColor(g_IpmImg, g_TempImg, cv::COLOR_BGR2GRAY); //tempImg 사용
    cv::GaussianBlur(g_TempImg, g_TempImg, cv::Size(3,3), 0);

    // =======================  이미지 전처리 튜닝  ===================
    if(track_bar)
    {
        int thresh     = cv::getTrackbarPos("Threshold", "PreprocessControl");
        int canny_low  = cv::getTrackbarPos("CannyLow",  "PreprocessControl");
        int canny_high = cv::getTrackbarPos("CannyHigh", "PreprocessControl");
        int ksize      = cv::getTrackbarPos("DilateK",   "PreprocessControl");

        // 커널 크기 보정
        if (ksize < 1) ksize = 1;
        if (ksize % 2 == 0) ksize += 1;

        if (canny_low > canny_high) std::swap(canny_low, canny_high);

        cv::Mat st_K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));

        cv::threshold(g_TempImg, g_TempImg, thresh, 255, cv::THRESH_BINARY);
        cv::dilate(g_TempImg, g_TempImg, st_K);
        cv::Canny(g_TempImg, g_TempImg, canny_low, canny_high);
    }
    else
    {
    // =======================  이진화 + 팽창 + Canny  ===================
        cv::Mat st_K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        // 이진화
        cv::threshold(g_TempImg, g_TempImg, 170, 255, cv::THRESH_BINARY);   // 170 보다 크면 255 아니면 0

        // 팽창
        cv::dilate(g_TempImg, g_TempImg, st_K);

        // Canny Edge , 140보다 낮으면 무시, 330보다 높으면 엣지 
        cv::Canny(g_TempImg, g_TempImg, 140, 330); // non zero 스캔 때문에 canny가 더 빠름 
    }

    // =======================  슬라이딩 윈도우 준비  ====================
    cv::Mat st_Tmp;
    cv::findNonZero(g_TempImg, st_Tmp);    // 0이 아닌 Pixel 추출 --> 차선후보픽셀 

    int32_t s32_WindowCentorLeft  = 0; // 인덱스 0으로 초기화 
    int32_t s32_WindowCentorRight = 0;

    // =======================  히스토그램 기반 시작 위치 탐색 ================== 
    FindLaneStartPositions(g_TempImg,
                            s32_WindowCentorLeft, //0
                            s32_WindowCentorRight, // 0
                            b_NoLaneLeft, // false
                            b_NoLaneRight,
                            camera_data->last_left_start_x,
                            camera_data->has_last_left_start,
                            camera_data->last_right_start_x,
                            camera_data->has_last_right_start);


    //히스토그램 로직에서 중앙에 있는 엉뚱한거를 차선으로 안 잡게 로직 추가하기 

    // =======================  슬라이딩 윈도우로 좌/우 차선 탐색 ==========
    // 결과물 Left/Right Lane Info main에 저장
    SlidingWindow(g_TempImg,
                  st_Tmp,
                  st_LaneInfoLeft,
                  st_LaneInfoRight,
                  s32_WindowCentorLeft,
                  s32_WindowCentorRight,
                  g_ResultImage);
    
    // ======================= RANSAC으로 차선 계수 산출 ====================

    // RANSAC으로 차선 계수 산출 , 셈플 6개 미만이면 차선 없는걸로 간주  --> sample count는 윈도우당 1개 
    if (st_LaneInfoLeft.s32_SampleCount < 6)
        b_NoLaneLeft = true;
    if (st_LaneInfoRight.s32_SampleCount < 6)
        b_NoLaneRight = true;

    const int32_t BASE_ITER = 500;  // 기본 RANSAC 반복 수 (튜닝 가능)
    
    if(!b_NoLaneLeft)
    {
        int32_t N = st_LaneInfoLeft.s32_SampleCount;
        // 차선이 있다고 판단한 경우 ransac 수행 (sample point를 이용해 ransac)
        int32_t max_combinations = N * (N - 1) / 2; // 가능한 최대 조합의 수 
        int32_t iterations = std::min(BASE_ITER, max_combinations);
        CalculateLaneCoefficient(st_LaneInfoLeft,iterations,1); // iteration, threshold
        st_LaneInfoLeft.st_LaneCoefficient.b_IsLeft = true; // 왼쪽차선 정보 저장 (기본값 false)
        st_LaneInfoLeftMain = st_LaneInfoLeft; // 관측값 보관용 
    }
    if(!b_NoLaneRight)
    {
        int32_t N = st_LaneInfoRight.s32_SampleCount;
        // 차선이 있다고 판단한 경우 ransac 수행 (sample point를 이용해 ransac)
        int32_t max_combinations = N * (N - 1) / 2; // 가능한 최대 조합의 수 
        int32_t iterations = std::min(BASE_ITER, max_combinations);
        CalculateLaneCoefficient(st_LaneInfoRight,iterations,1);
        st_LaneInfoRightMain = st_LaneInfoRight;
    }

    // ======================= Kalman Filter  ========================
    
    // b_ThereIsLeft / b_ThereIsRight : 추적중인 kalman 객체의 유무 여부 
    // arst_KalmanObject[] : kalman 객체 배열 --> 0 : 이번 프레임 왼쪽, 1 : 이번 프레임 오른쪽 
    // s32_KalmanObjectNum : 현재 kalman 객체 개수

    if (!camera_data->b_ThereIsLeft && !camera_data->b_ThereIsRight) // 양쪽 모두 칼만객체 없는 경우 
    {
        int margin = 50 ; // b_IsLeft 판단용 마진 --> ?

        // ---- 왼쪽 차선 kalman 객체 새로 생성 ----
        if (!camera_data->b_ThereIsLeft && !b_NoLaneLeft) // 왼쪽 칼만 객체 없고, 왼쪽 차선 감지된 경우
        {
            // Kalman state structure 계산 --> 이전 프레임과의 차이 계산하는 로직 
            st_KalmanStateLeft = CalculateKalmanState(
                st_LaneInfoLeft.st_LaneCoefficient, // ransac로 구한 차선 계수
                camera_data->f32_LastDistanceLeft, // 이전 프레임 왼쪽 차선 거리 (초깃값 : 0)
                camera_data->f32_LastAngleLeft // 이전 프레임 왼쪽 차선 각도
            );

            // kalman 객체 생성 및 초기화 
            LANE_KALMAN st_KalmanObject; 
            InitializeKalmanObject(st_KalmanObject); // A,P,Q,R 세팅
            UpdateObservation(st_KalmanObject, st_KalmanStateLeft); // Z 관측값 업데이트 
            SetInitialX(st_KalmanObject); // X <- Z 로 초기화
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoLeft.st_LaneCoefficient;

            // 좌우 판단 로직 
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            // st_KalmanObject.b_IsLeft = (x_intercept < center_x - margin); // 아래쪽 차선이 센터-마진이면 왼쪽 
            st_KalmanObject.b_IsLeft = true ;

            // 전역 상태에 등록  + 결과 이미지에 그리기 
            st_KalmanObject.st_LaneState = st_KalmanStateLeft;
            camera_data->b_ThereIsLeft = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject; // 배열에 push
            camera_data->s32_KalmanObjectNum += 1;

            // 초록색 선 그리기 
            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(0,255,0));
        }


        // ---- 오른쪽 차선 kalman 객체 새로 생성 ----
        if (!camera_data->b_ThereIsRight && !b_NoLaneRight) // 오른쪽 칼만 객체 없고, 오른쪽 차선 감지된 경우
        {
            st_KalmanStateRight = CalculateKalmanState(
                st_LaneInfoRight.st_LaneCoefficient,
                camera_data->f32_LastDistanceRight,
                camera_data->f32_LastAngleRight
            );

            LANE_KALMAN st_KalmanObject; // 왼쪽이랑 같은 칼만객체??
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject, st_KalmanStateRight);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoRight.st_LaneCoefficient;

            // b_IsLeft 판단 : false --> 오 / true --> 왼
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            // 이거 로직 이상함. 수정해야할 수도 
            // st_KalmanObject.b_IsLeft = (x_intercept < center_x + margin);
            st_KalmanObject.b_IsLeft = false ;

            st_KalmanObject.st_LaneState = st_KalmanStateRight;
            camera_data->b_ThereIsRight = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            // 파랑색 선 그리기
            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 0, 0));
        }

    }

    else // 한쪽 칼만 객체라도 있는 경우  : 이미 칼만객체 생성 이후 추적중인 경우 --> 업데이트
    {
        bool has_obs[2] = {!b_NoLaneLeft, !b_NoLaneRight}; // 좌우 차선 관측 유무 저장
        // 관측 없는 쪽도 0으로 매칭하는 문제 생김 
        // 이번 프레임 관측값 저장 
        if (has_obs[0])  // 왼쪽 차선 감지된 경우 
        {
            arst_KalmanState[0] = CalculateKalmanState(
                st_LaneInfoLeft.st_LaneCoefficient,
                camera_data->f32_LastDistanceLeft,
                camera_data->f32_LastAngleLeft
            );
        }

        if (has_obs[1]) // 오른쪽 차선 감지된 경우 
        {
            arst_KalmanState[1] = CalculateKalmanState(
                st_LaneInfoRight.st_LaneCoefficient,
                camera_data->f32_LastDistanceRight,
                camera_data->f32_LastAngleRight
            );
        }

        for (s32_I = 0; s32_I < camera_data->s32_KalmanObjectNum; s32_I++) // 저장된 칼만 객체 순회 
        {
            bool b_SameObj = false; // 같은 객체인지 판별하는 플래그 

            // 칼만 객체와 새 관측을 비교해 동일 차선인지 판별
            for (s32_J = 0; s32_J < 2; s32_J++) // 좌우 2개 관측값 순회 0 : 왼쪽 / 1 : 오른쪽 
            {
                if(!has_obs[s32_J]) // 해당 관측값 없는 경우 스킵 
                    continue;

                bool is_left_meas = (s32_J == 0);

                // 좌/우 매칭 안되면 스킵
                if (camera_data->arst_KalmanObject[s32_I].b_IsLeft != is_left_meas)
                    continue;
    
                //같은 선인지 확인 --> 매칭이 되어야 기존 차선이 업데이트가 됨 
                CheckSameKalmanObject(camera_data->arst_KalmanObject[s32_I],
                                      arst_KalmanState[s32_J]);  // 동일 차선인지 비교

                // 같은 차선으로 매칭 성공 --> 칼만 필터 업데이트
                if (camera_data->arst_KalmanObject[s32_I].b_MeasurementUpdateFlag)
                {
                    UpdateObservation(camera_data->arst_KalmanObject[s32_I],
                                      arst_KalmanState[s32_J]);
                    PredictState(camera_data->arst_KalmanObject[s32_I]);

                    if (s32_J == 0)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient =
                            st_LaneInfoLeft.st_LaneCoefficient;
                    else if (s32_J == 1)
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient =
                            st_LaneInfoRight.st_LaneCoefficient;

                    UpdateMeasurement(camera_data->arst_KalmanObject[s32_I]);
                    MakeKalmanStateBasedLaneCoef(
                        camera_data->arst_KalmanObject[s32_I],
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient
                    );

                    // 이번 프레임으로 데이터 업데이트(칼만객체 비교 )
                    camera_data->arst_KalmanObject[s32_I].st_LaneState = arst_KalmanState[s32_J];

                    if (s32_J == 0)
                        DrawDrivingLane(g_ResultImage,
                                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                        cv::Scalar(0,255,0)); // 초록색 선 그리기
                    else if (s32_J == 1)
                        DrawDrivingLane(g_ResultImage,
                                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                        cv::Scalar(255, 0, 0)); // 파랑색 선 그리기

                    b_SameObj = true;
                    break;
                }
            }

            // kalman 예측 수행 
            if (!b_SameObj) // 같은 차선으로 매칭 실패한 경우 --> 예측만 수행
            {

                if (camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching < 10) // 10 프레임 기준 --> 너무 길면 드리프트 갈겨버림
                {
                    camera_data->arst_KalmanObject[s32_I].s32_CntNoMatching += 1;

                    // 관측이 안 들어오는 동안에는 속도 성분을 조금씩 줄여서 드리프트 억제
                    camera_data->arst_KalmanObject[s32_I].st_X(1) *= 0.5 ; // 거리 절반
                    camera_data->arst_KalmanObject[s32_I].st_X(3) *= 0.5 ; // 각도 절반 

                    PredictState(camera_data->arst_KalmanObject[s32_I]);
                    MakeKalmanStateBasedLaneCoef(
                        camera_data->arst_KalmanObject[s32_I],
                        camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient
                    );
                    // 예측값으로 차선 그리기 
                    DrawDrivingLane(g_ResultImage,
                                    camera_data->arst_KalmanObject[s32_I].st_LaneCoefficient,
                                    cv::Scalar(0, 0, 255)); // 예측값은 빨강색
                }
                else // 30프레임 안에 관측값 매칭 실패하면 차선 추적 종료. 칼만객체 삭제 
                {
                    // 배열에서 칼만 객체 제거 
                    DeleteKalmanObject(*camera_data,
                                       camera_data->s32_KalmanObjectNum,
                                       s32_I);
                }
            }
        }
    }

    if (!camera_data->b_ThereIsLeft && !b_NoLaneLeft) {
        // Kalman state structure 계산 --> 이전 프레임과의 차이 계산하는 로직 
            st_KalmanStateLeft = CalculateKalmanState(
                st_LaneInfoLeft.st_LaneCoefficient, // ransac로 구한 차선 계수
                camera_data->f32_LastDistanceLeft, // 이전 프레임 왼쪽 차선 거리 (초깃값 : 0)
                camera_data->f32_LastAngleLeft // 이전 프레임 왼쪽 차선 각도
            );

            // kalman 객체 생성 및 초기화 
            LANE_KALMAN st_KalmanObject; 
            InitializeKalmanObject(st_KalmanObject); // A,P,Q,R 세팅
            UpdateObservation(st_KalmanObject, st_KalmanStateLeft); // Z 관측값 업데이트 
            SetInitialX(st_KalmanObject); // X <- Z 로 초기화
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoLeft.st_LaneCoefficient;

            // 좌우 판단 로직 
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            // st_KalmanObject.b_IsLeft = (x_intercept < center_x - margin); // 아래쪽 차선이 센터-마진이면 왼쪽 
            st_KalmanObject.b_IsLeft = true ;

            // 전역 상태에 등록  + 결과 이미지에 그리기 
            st_KalmanObject.st_LaneState = st_KalmanStateLeft;
            camera_data->b_ThereIsLeft = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject; // 배열에 push
            camera_data->s32_KalmanObjectNum += 1;

            // 초록색 선 그리기 
            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(0,255,0)); // 초록색선
    }

    // 오른쪽도 동일
    if (!camera_data->b_ThereIsRight && !b_NoLaneRight) {
        st_KalmanStateRight = CalculateKalmanState(
                st_LaneInfoRight.st_LaneCoefficient,
                camera_data->f32_LastDistanceRight,
                camera_data->f32_LastAngleRight
            );

            LANE_KALMAN st_KalmanObject; // 왼쪽이랑 같은 칼만객체??
            InitializeKalmanObject(st_KalmanObject);
            UpdateObservation(st_KalmanObject, st_KalmanStateRight);
            SetInitialX(st_KalmanObject);
            st_KalmanObject.st_LaneCoefficient = st_LaneInfoRight.st_LaneCoefficient;

            // b_IsLeft 판단 : false --> 오 / true --> 왼
            int center_x = camera_data->st_CameraParameter.s32_RemapWidth / 2;
            double x_intercept = -st_KalmanObject.st_LaneCoefficient.f64_Intercept /
                     st_KalmanObject.st_LaneCoefficient.f64_Slope;
            
            // 이거 로직 이상함. 수정해야할 수도 
            // st_KalmanObject.b_IsLeft = (x_intercept < center_x + margin);
            st_KalmanObject.b_IsLeft = false ;

            st_KalmanObject.st_LaneState = st_KalmanStateRight;
            camera_data->b_ThereIsRight = true;
            camera_data->arst_KalmanObject[camera_data->s32_KalmanObjectNum] = st_KalmanObject;
            camera_data->s32_KalmanObjectNum += 1;

            // 파랑색 선 그리기
            DrawDrivingLane(g_ResultImage,
                            st_KalmanObject.st_LaneCoefficient,
                            cv::Scalar(255, 0, 0));
    }

    // =======================  칼만 결과 기반 consistency 보정 ========================
    {
        LANE_COEFFICIENT kalman_left, kalman_right;
        bool has_left  = false;
        bool has_right = false;

        // 현재 프레임에서 칼만 객체로부터 coef 뽑기
        get_lane_coef_from_kalman(*camera_data,
                                  kalman_left, kalman_right,
                                  has_left, has_right);

        // RANSAC sample 개수 기반 신뢰도
        double conf_left  = static_cast<double>(st_LaneInfoLeft.s32_SampleCount);
        double conf_right = static_cast<double>(st_LaneInfoRight.s32_SampleCount);

        if (has_left && has_right &&
            st_LaneInfoLeft.s32_SampleCount  > 0 &&
            st_LaneInfoRight.s32_SampleCount > 0)
        {
            bool left_anchor = (conf_left >= conf_right);

            // IPM 상에서 기대 차선 폭(px) 
            double expected_width_px = 530 ;

            bool ok = EnforceLaneConsistencyAnchor(
                kalman_left,
                kalman_right,
                camera_data->st_CameraParameter.s32_RemapHeight,
                left_anchor,
                expected_width_px
            );

            if (ok) {
                // 보정된 coef를 다시 칼만 객체에 반영
                for (int i = 0; i < camera_data->s32_KalmanObjectNum; ++i)
                {
                    auto& obj = camera_data->arst_KalmanObject[i];
                    if (obj.b_IsLeft) {
                        obj.st_LaneCoefficient = kalman_left;
                    } else {
                        obj.st_LaneCoefficient = kalman_right;
                    }
                }
            }
        }
    }

    // sliding window update
    if (!b_NoLaneLeft) {
    camera_data->last_left_start_x = s32_WindowCentorLeft;
    camera_data->has_last_left_start = true;
    }

    if (!b_NoLaneRight) {
        camera_data->last_right_start_x = s32_WindowCentorRight;
        camera_data->has_last_right_start = true;
    }


    // =======================  (H) Debug GUI ================================
    if (visualize)
    {
        cv::imshow("IPM", g_IpmImg);          // 탑뷰
        cv::imshow("Temp_Img", g_TempImg);    // 현재는 sliding window 결과

        // =======================  RANSAC 디버그 창 ===========================
        cv::Mat ransac_debug;
        g_IpmImg.copyTo(ransac_debug);   // 또는 g_TempImg를 COLOR_GRAY2BGR로 변환해서 써도 됨

        if (!b_NoLaneLeft) {
            DrawDrivingLane(ransac_debug,
                            st_LaneInfoLeftMain.st_LaneCoefficient,
                            cv::Scalar(255, 0, 0));   // 왼쪽 차선 파랑색
        }

        if (!b_NoLaneRight) {
            DrawDrivingLane(ransac_debug,
                            st_LaneInfoRightMain.st_LaneCoefficient,
                            cv::Scalar(0, 0, 255)); // 오른쪽은 빨간색
        }

        cv::imshow("RANSAC Debug", ransac_debug);   // RANSAC 전용 창
        // =======================  RANSAC 디버그 창 ===========================

        // // ---------- IPM 상 차선 폭 계산 + 출력 ---------- //
        if(cal_lane_width) // 약 262 px 
        {
            LANE_COEFFICIENT kalman_left, kalman_right;
            bool has_left = false, has_right = false;

            if (get_lane_coef_from_kalman(*camera_data,
                                          kalman_left, kalman_right,
                                          has_left, has_right)
                && has_left && has_right)
            {
                double width_px = 0.0;
                double angle_diff_deg = 0.0;
                int H = camera_data->st_CameraParameter.s32_RemapHeight;

                if (ComputeLaneWidthAngle(kalman_left,
                                          kalman_right,
                                          H,
                                          width_px,
                                          angle_diff_deg))
                {
                    // 콘솔 출력 --> 400/320기준 305px
                    std::cout << "[IPM] lane width: " << std::fixed << std::setprecision(1) << width_px << " px" << std::endl;

                }
            }
        }
        // // ---------- 차폭계산 , 출력 --------------------------------- //

        cv::imshow("Kalman Result", g_ResultImage); // 차선 + Kalman 결과
        cv::waitKey(1);
    }
}
//######################################### MakeKalmanStateBasedLaneCoef func  ##################################################//

// 칼만 상태를 선형 차선 계수로 변환
void MakeKalmanStateBasedLaneCoef(const LANE_KALMAN& st_KalmanObject, LANE_COEFFICIENT& st_LaneCoefficient)
{
    float64_t f64_Theta_Radian;

    // printf("---------------MakeKalmanStateBasedLaneCoef-----------\n");
    f64_Theta_Radian = st_KalmanObject.st_X[2]* M_PI / 180.0;
    if (std::abs(std::cos(f64_Theta_Radian)) < 1e-6) 
    { // Use a small epsilon to check for zero
        // std::cout << "The line is vertical. Equation of the line: x = " << d / std::sin(f64_Theta_Radian) << std::endl;
    } else {
        // Calculate the slope (m) and y-intercept (c)
        st_LaneCoefficient.f64_Slope = -std::tan(f64_Theta_Radian);
        st_LaneCoefficient.f64_Intercept = st_KalmanObject.st_X[0] / std::cos(f64_Theta_Radian);
        // std::cout << "y = " << st_LaneCoefficient.f64_Slope << "x + " << st_LaneCoefficient.f64_Intercept << std::endl;
    }
}
 //######################################### DeleteKalmanObject func  ##################################################//

// 매칭 실패한 칼만 차선 객체를 제거
void DeleteKalmanObject(CAMERA_DATA &pst_CameraData, int32_t& s32_KalmanObjectNum, int32_t s32_I)
{
    int32_t s32_J;
    if (s32_KalmanObjectNum == 1)
    {
        s32_KalmanObjectNum -= 1;
        if (pst_CameraData.arst_KalmanObject[0].b_IsLeft)
            pst_CameraData.b_ThereIsLeft = false;
        else
            pst_CameraData.b_ThereIsRight = false;
    }
    else
    {
        if (pst_CameraData.arst_KalmanObject[s32_I].b_IsLeft)
            pst_CameraData.b_ThereIsLeft = false;
        else
            pst_CameraData.b_ThereIsRight = false;

        for(s32_J = s32_I; s32_J<s32_KalmanObjectNum-1;s32_J++)
        {
            pst_CameraData.arst_KalmanObject[s32_J] = pst_CameraData.arst_KalmanObject[s32_J+1];
        }
        s32_KalmanObjectNum -= 1;

    }
}
//######################################### CheckSameKalmanObject func  ##################################################//

// 새 관측 차선이 기존 칼만 객체와 동일한지 여부 판단
void CheckSameKalmanObject(LANE_KALMAN& st_KalmanObject, KALMAN_STATE st_KalmanStateLeft)
{
    st_KalmanObject.b_MeasurementUpdateFlag = false;
    // Parameter yaml에서 끌어오도록 수정 필요
    if (abs(st_KalmanObject.st_LaneState.f64_Distance - st_KalmanStateLeft.f64_Distance) < 40) // 40px
    {
        if(abs(st_KalmanObject.st_LaneState.f64_Angle - st_KalmanStateLeft.f64_Angle) < 10) // 10도차이
        {
            st_KalmanObject.b_MeasurementUpdateFlag = true; // 조건 만족시 같은 차선으로 간주 
        }
    }
}
//######################################### PredictState func  ##################################################//

// 칼만 필터 예측 단계
void PredictState(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_PrevX = st_KalmanObject.st_X;

    st_KalmanObject.st_X = st_KalmanObject.st_A * st_KalmanObject.st_X;
    st_KalmanObject.st_P = st_KalmanObject.st_A * st_KalmanObject.st_P * st_KalmanObject.st_A.transpose() + st_KalmanObject.st_Q;

}

//######################################### UpdateMeasurement func  ##################################################//

// 칼만 필터 측정 업데이트 단계
void UpdateMeasurement(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_K = st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() * (st_KalmanObject.st_H * st_KalmanObject.st_P * st_KalmanObject.st_H.transpose() + st_KalmanObject.st_R).inverse();
    st_KalmanObject.st_P = st_KalmanObject.st_P - st_KalmanObject.st_K * st_KalmanObject.st_H * st_KalmanObject.st_P;
    st_KalmanObject.st_X = st_KalmanObject.st_X + st_KalmanObject.st_K * (st_KalmanObject.st_Z - st_KalmanObject.st_H * st_KalmanObject.st_X);

    st_KalmanObject.s32_CntNoMatching = 0;

}

//######################################### SetInitialX func  ##################################################//

// 관측 기반으로 상태 벡터 초기화
void SetInitialX(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_X(0) = st_KalmanObject.st_Z(0);
    st_KalmanObject.st_X(1) = st_KalmanObject.st_Z(1);
    st_KalmanObject.st_X(2) = st_KalmanObject.st_Z(2);
    st_KalmanObject.st_X(3) = st_KalmanObject.st_Z(3);
}

//######################################### UpdateObservation func  ##################################################//

// 관측 벡터에 거리/각도 값을 기록
void UpdateObservation(LANE_KALMAN& st_KalmanObject, const KALMAN_STATE st_KalmanState)
{
    st_KalmanObject.st_Z(0) = st_KalmanState.f64_Distance;
    st_KalmanObject.st_Z(1) = st_KalmanState.f64_DeltaDistance;
    st_KalmanObject.st_Z(2) = st_KalmanState.f64_Angle;
    st_KalmanObject.st_Z(3) = st_KalmanState.f64_DeltaAngle;
}

//######################################### CalculateKalmanState func  ##################################################//

// 직선 모델을 거리·각도 형태의 칼만 상태로 변환
KALMAN_STATE CalculateKalmanState(const LANE_COEFFICIENT& st_LaneCoef, float32_t& f64_Distance, float32_t& f64_Angle) 
{

    KALMAN_STATE st_KalmanState;
    float64_t s64_X, s64_Y;

    s64_X = -st_LaneCoef.f64_Slope * st_LaneCoef.f64_Intercept / (st_LaneCoef.f64_Slope*st_LaneCoef.f64_Slope + 1);
    s64_Y = st_LaneCoef.f64_Slope * s64_X + st_LaneCoef.f64_Intercept;

    st_KalmanState.f64_Distance = sqrt(pow(s64_X,2)+pow(s64_Y,2));
    st_KalmanState.f64_Angle = 90 - atan2(s64_Y,s64_X) * (180.0 / M_PI);
    // st_KalmanState.f64_Angle = atan2(s64_Y,s64_X) * (180.0 / M_PI);
    st_KalmanState.f64_DeltaDistance =  st_KalmanState.f64_Distance - f64_Distance;
    st_KalmanState.f64_DeltaAngle =  st_KalmanState.f64_Angle - f64_Angle;

    // Update Last Distance, Angle
    f64_Distance = st_KalmanState.f64_Distance;
    f64_Angle = st_KalmanState.f64_Angle;
    return st_KalmanState;
}

//######################################### InitializeKalmanObject func  ##################################################//

// 칼만 필터 행렬 및 공분산 초기화
void InitializeKalmanObject(LANE_KALMAN& st_KalmanObject)
{
    st_KalmanObject.st_A << 1, 1, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 1,
                            0, 0, 0, 1;

    st_KalmanObject.st_P << 0.01, 0   , 0   , 0,
                            0   , 0.01, 0   , 0,
                            0   , 0   , 0.01, 0,
                            0   , 0   , 0   , 0.01;

    // Q 작으면 상태가 거의 안 바뀐다는 가정 
    st_KalmanObject.st_Q << 0.00013 , 0   , 0   , 0,
                            0   , 0.00013 , 0   , 0,
                            0   , 0   , 0.00013 , 0,
                            0   , 0   , 0   , 0.00013;

    // R 측정 노이즈 공분산 행렬 --> 값이 크면 측정값 신뢰도 낮음 
    st_KalmanObject.st_R << 2.5   , 0   , 0   , 0,
                            0   , 5   , 0   , 0,
                            0   , 0   , 10   , 0,
                            0   , 0   , 0   , 15;

    st_KalmanObject.st_X.setZero();
    st_KalmanObject.st_PrevX.setZero();
    st_KalmanObject.st_Z.setZero();

}

//######################################### SlidingWindow func ##################################################//

// edge에서 슬라이딩 윈도로 좌/우 차선 포인트를 추출
void SlidingWindow(const cv::Mat& st_EdgeImage,
                   const cv::Mat& st_NonZeroPosition,
                   CAMERA_LANEINFO& st_LaneInfoLeft,
                   CAMERA_LANEINFO& st_LaneInfoRight,
                   int32_t& s32_WindowCentorLeft,
                   int32_t& s32_WindowCentorRight,
                   cv::Mat& st_ResultImage)
{
    // 이미지 크기
    const int cols = st_EdgeImage.cols; // x
    const int rows = st_EdgeImage.rows; // y
    const int32_t ImgHeight = rows - 1; // IPM 좌표계 맞추기용

    // 윈도우 파라미터
    int32_t s32_WindowHeight = rows - 1;        // 현재 윈도우의 "아래쪽" y (맨 아래에서 시작)
    const int32_t s32_MarginX = 20;             // 윈도우 가로 반폭
    const int32_t s32_MarginY = 30;             // 윈도우 세로 높이 --> 윈도우 총 개수를 결정
    int32_t s32_I, s32_CentorX, s32_CentorY;

    // 유효 윈도우 유지 여부
    bool b_ValidWindowLeft  = true;
    bool b_ValidWindowRight = true;

    // RANSAC용 샘플 개수 초기화
    st_LaneInfoLeft.s32_SampleCount  = 0;
    st_LaneInfoRight.s32_SampleCount = 0;

    // 컨투어 버퍼
    std::vector<std::vector<cv::Point>> st_Contours;

    // 윈도우 유효 여부 플래그
    bool b_CheckValidWindowLeft  = false;
    bool b_CheckValidWindowRight = false;

    // 연속으로 실패한 윈도우 카운트
    int32_t s32_CountAnvalidLeft  = 0;
    int32_t s32_CountAnvalidRight = 0;

    cv::Mat st_WindowMask;
    cv::Rect st_LeftWindow;
    cv::Rect st_RightWindow;

    const int img_center_x = cols / 2; // 이미지 전체 중앙

    // ====================== 슬라이딩 윈도우 루프 ======================
    while (s32_WindowHeight > 0)
    {
        // 윈도우 세로(높이) 방향 경계
        int32_t s32_WindowMinHeight = s32_WindowHeight - s32_MarginY;
        if (s32_WindowMinHeight < 0) s32_WindowMinHeight = 0;

        // ---- 현재 Left/Right 윈도우의 좌우 경계 계산 ----
        int32_t s32_WindowMinWidthLeft  = std::max(s32_WindowCentorLeft  - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthLeft  = std::min(s32_WindowCentorLeft  + s32_MarginX, cols - 1);
        int32_t s32_WindowMinWidthRight = std::max(s32_WindowCentorRight - s32_MarginX, 0);
        int32_t s32_WindowMaxWidthRight = std::min(s32_WindowCentorRight + s32_MarginX, cols - 1);

        int32_t s32_WidthLeft  = s32_WindowMaxWidthLeft  - s32_WindowMinWidthLeft  + 1;
        int32_t s32_WidthRight = s32_WindowMaxWidthRight - s32_WindowMinWidthRight + 1;

        // --- 윈도우 유효 플래그 초기화 (이번 루프에서 다시 세팅) ---
        b_CheckValidWindowLeft  = false;
        b_CheckValidWindowRight = false;

        // ======================= Left Lane =======================
        if (!b_NoLaneLeft)
        {
            if (b_ValidWindowLeft)
            {
                // 현재 윈도우 영역(Rect) 정의
                st_LeftWindow = cv::Rect(
                    s32_WindowMinWidthLeft,
                    s32_WindowMinHeight,
                    s32_WidthLeft,
                    s32_MarginY);

                // 해당 영역만 잘라서 컨투어 검출
                st_WindowMask = st_EdgeImage(st_LeftWindow);
                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                // 모멘트 기반 윈도우 내부 중심 계산 → 다음 윈도우 x 중심 갱신
                for (s32_I = 0; s32_I < (int)st_Contours.size(); ++s32_I)
                {
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    double m00 = M.m00, m10 = M.m10, m01 = M.m01;
                    if (m00 != 0.0)
                    {
                        s32_CentorX = static_cast<int>(m10 / m00);
                        s32_CentorY = static_cast<int>(m01 / m00);
                        s32_WindowCentorLeft = s32_WindowMinWidthLeft + s32_CentorX; // 로컬→글로벌 x
                        b_CheckValidWindowLeft = true;
                    }
                }

                cv::rectangle(
                    st_EdgeImage,
                    cv::Point(s32_WindowMinWidthLeft,  s32_WindowMinHeight),
                    cv::Point(s32_WindowMaxWidthLeft,  s32_WindowHeight),
                    cv::Scalar(255, 255, 255),
                    2);
                
            }

            // ---- 이번 윈도우에서 유효한 차선 조각을 못 찾은 경우 ----
            if (!b_CheckValidWindowLeft)
            {
                ++s32_CountAnvalidLeft;
                if (s32_CountAnvalidLeft >= 7)
                {
                    b_ValidWindowLeft = false; // 더 이상 윈도우 올리지 않음
                }
            }
            else
            {
                // 유효 윈도우 → 샘플 픽셀 선택
                s32_CountAnvalidLeft = 0;
                double best_score = std::numeric_limits<double>::max();

                // 새로 갱신된 윈도우 중심
                const int window_center_x_left = s32_WindowCentorLeft;

                for (int idx = 0; idx < st_NonZeroPosition.total(); ++idx)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<cv::Point>(idx);

                    // 현재 윈도우 영역 내의 픽셀만 사용
                    if (st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >= s32_WindowMinWidthLeft && st_Position.x <= s32_WindowMaxWidthLeft)
                    {
                        // 윈도우 중심 + 이미지 중심 복합 스코어
                        double d_window = std::abs(st_Position.x - window_center_x_left);
                        double d_global = std::abs(st_Position.x - img_center_x);
                        double score    = 0.7 * d_window + 0.3 * d_global;

                        if (score < best_score)
                        {
                            best_score = score;
                            cv::Point ipmPt = st_Position;
                            ipmPt.y = ImgHeight - ipmPt.y; // 좌표계 반전(IPM 기준)
                            st_LaneInfoLeft.arst_LaneSample[st_LaneInfoLeft.s32_SampleCount] = ipmPt;
                        }
                    }
                }

                if (best_score < std::numeric_limits<double>::max())
                {
                    ++st_LaneInfoLeft.s32_SampleCount;
                }
            }

            st_Contours.clear();
        }

        // ======================= Right Lane =======================
        if (!b_NoLaneRight)
        {
            if (b_ValidWindowRight)
            {
                st_RightWindow = cv::Rect(
                    s32_WindowMinWidthRight,
                    s32_WindowMinHeight,
                    s32_WidthRight,
                    s32_MarginY);

                st_WindowMask = st_EdgeImage(st_RightWindow);
                cv::findContours(st_WindowMask, st_Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                for (s32_I = 0; s32_I < (int)st_Contours.size(); ++s32_I)
                {
                    cv::Moments M = cv::moments(st_Contours[s32_I]);
                    double m00 = M.m00, m10 = M.m10, m01 = M.m01;
                    if (m00 != 0.0)
                    {
                        s32_CentorX = static_cast<int>(m10 / m00);
                        s32_CentorY = static_cast<int>(m01 / m00);
                        s32_WindowCentorRight = s32_WindowMinWidthRight + s32_CentorX;
                        b_CheckValidWindowRight = true;
                    }
                }

                cv::rectangle(
                    st_EdgeImage,
                    cv::Point(s32_WindowMinWidthRight,  s32_WindowMinHeight),
                    cv::Point(s32_WindowMaxWidthRight,  s32_WindowHeight),
                    cv::Scalar(255, 255, 255),
                    2);
                
            }

            if (!b_CheckValidWindowRight)
            {
                ++s32_CountAnvalidRight;
                if (s32_CountAnvalidRight >= 7)
                {
                    b_ValidWindowRight = false;
                }
            }
            else
            {
                s32_CountAnvalidRight = 0;
                double best_score = std::numeric_limits<double>::max();

                const int window_center_x_right = s32_WindowCentorRight;

                for (int idx = 0; idx < st_NonZeroPosition.total(); ++idx)
                {
                    cv::Point st_Position = st_NonZeroPosition.at<cv::Point>(idx);

                    if (st_Position.y >= s32_WindowMinHeight && st_Position.y <= s32_WindowHeight &&
                        st_Position.x >= s32_WindowMinWidthRight && st_Position.x <= s32_WindowMaxWidthRight)
                    {

                        double d_window = std::abs(st_Position.x - window_center_x_right);
                        double d_global = std::abs(st_Position.x - img_center_x);
                        double score    = 0.7 * d_window + 0.3 * d_global;

                        if (score < best_score) // 한번은 무조건 들어감 
                        {
                            best_score = score;
                            cv::Point ipmPt = st_Position;
                            ipmPt.y = ImgHeight - ipmPt.y;
                            st_LaneInfoRight.arst_LaneSample[st_LaneInfoRight.s32_SampleCount] = ipmPt;
                        }
                    }
                }

                if (best_score < std::numeric_limits<double>::max())
                {
                    ++st_LaneInfoRight.s32_SampleCount;
                }
            }

            st_Contours.clear();
        }
        // ======================= 다음 윈도우로 이동 =======================
        s32_WindowHeight -= s32_MarginY;
    }

}

//###################################### DrawDrivingLane func ##################################################//

// 추정된 차선 계수로 결과 영상에 선을 그린다(직선)
void DrawDrivingLane(cv::Mat& st_ResultImage, const LANE_COEFFICIENT st_LaneCoef, cv::Scalar st_Color)
{
    if (std::abs(st_LaneCoef.f64_Slope) < 1e-6)
        return; // 혹은 x = const 직선 방식으로 따로 처리

    int rows = st_ResultImage.rows;
    int32_t x0 = int(-st_LaneCoef.f64_Intercept / st_LaneCoef.f64_Slope);
    int32_t x1 = int(((rows - 1) - st_LaneCoef.f64_Intercept) / st_LaneCoef.f64_Slope);

    cv::line(st_ResultImage,
             cv::Point(x0, rows - 1),
             cv::Point(x1, 0),
             st_Color, 2);
}

//###################################### FitModel func ##################################################//

// 두 포인트를 이용해 직선 모델을 구성
LANE_COEFFICIENT FitModel(const Point& st_Point1, const Point& st_Point2, bool& b_Flag)
{
    LANE_COEFFICIENT st_TmpModel; // return value
    
    if((st_Point2.x != st_Point1.x)) // 수직선이 아닌경우 == 기울기가 있는 경우 
    {
        // float64_t slope 계산 --> 로직 확인필요 
        st_TmpModel.f64_Slope = 
            static_cast<float64_t>(st_Point2.y - st_Point1.y) / static_cast<float64_t>(st_Point2.x - st_Point1.x);

        // y절편 계산 
        st_TmpModel.f64_Intercept = 
            static_cast<float64_t>(st_Point1.y) - st_TmpModel.f64_Slope * st_Point1.x;
    }
    else 
        b_Flag = false; // 수직선 

    return st_TmpModel;
}

//###################################### CalculateLaneCoefficient func ##################################################//

// 슬라이딩 윈도우로 수집한 점들에서 RANSAC으로 차선 계수를 산출
void CalculateLaneCoefficient(CAMERA_LANEINFO& st_LaneInfo, int32_t s32_Iteration, int64_t s64_Threshold)
{
    // 양쪽 차선 데이터 중 중앙선에 가장 가까운 안촉 포인트 기준으로 RANSAC을 활용한 기울기 계산
    int32_t s32_BestInlierCount = 0 ;
    int32_t s32_I, s32_J;
    int32_t s32_Idx1, s32_Idx2, s32_InlierCount ;
    int32_t N = st_LaneInfo.s32_SampleCount; // point의 개수 

    bool b_Flag = true ; // 수직선 판별 flag (flase : 수직선 )
    LANE_COEFFICIENT st_Temp ; // 기울기 , 절편 정보 저장 

    // Coef Reset (camera_laneinfo struct)
    st_LaneInfo.st_LaneCoefficient.f64_Slope = 0;
    st_LaneInfo.st_LaneCoefficient.f64_Intercept = 0;

    for(s32_I=0;s32_I<s32_Iteration;s32_I++) // Iteration 횟수만큼 반복 --> 적당한 값 찾기(위에 조합으로 최대 횟수 있음)
    {
        // 두 점 랜덤으로 선택 
        int32_t s32_Idx1 = rand() % N;
        int32_t s32_Idx2 = rand() % (N - 1);
        if (s32_Idx2 >= s32_Idx1) 
            s32_Idx2 += 1; // 같은 점 선택 방지 

        // 두개 랜덤포인트로 모델 피팅 
        st_Temp = FitModel(st_LaneInfo.arst_LaneSample[s32_Idx1],st_LaneInfo.arst_LaneSample[s32_Idx2], b_Flag);

        if (b_Flag) // 수직선이 아닌 경우 (기울기가 있는 경우)
        {
            // 모든 점에 대해서 inlinear count 계산 
            s32_InlierCount = 0;
            double denom = std::sqrt(st_Temp.f64_Slope * st_Temp.f64_Slope + 1.0); // 매번 sqrt하지 않게 

            // 모든 샘플포인트에 대해서 직선과의 거리 계산 
            for(s32_J = 0; s32_J < st_LaneInfo.s32_SampleCount; s32_J++)
            {
                if(abs(-st_Temp.f64_Slope * st_LaneInfo.arst_LaneSample[s32_J].x + st_LaneInfo.arst_LaneSample[s32_J].y - st_Temp.f64_Intercept)
                        / denom < s64_Threshold) // threshold 이내이면 inlinear로 간주(1픽셀)
                {
                    s32_InlierCount += 1;
                } 
            }

            // Best Model 갱신 (초깃값 0이라 무조건 갱신)
            if (s32_InlierCount > s32_BestInlierCount)
            {
                s32_BestInlierCount = s32_InlierCount; //베스트 모델 갱신 
                st_LaneInfo.st_LaneCoefficient = st_Temp; //기울기 절편 갱신 
            }
        }

        b_Flag = true; // 다음 계산을 위해 초기화
    }

}

//###################################### FindTop5MaxIndices func ##################################################//

// 히스토그램에서 가장 누적 픽셀이 높은  다섯 개의 열 인덱스를 구하기 
void FindTop5MaxIndices(const int32_t* ps32_Histogram, int32_t s32_MidPoint, int32_t ars32_resultIdxs[5], bool& b_NoLane) 
{

    int32_t s32_I, s32_Cnt=0;
    std::pair<int32_t, int32_t> topValues[5];                 // (value, index)
    std::fill_n(topValues, 5, std::make_pair(0, -1));         // 0으로 초기화

    //히스토그램 순회 하면서 픽셀이 가장 높은 5개 
    for (s32_I = 0; s32_I < s32_MidPoint; ++s32_I) {
        if (ps32_Histogram[s32_I] > topValues[4].first) { //0이면 값이 안 들어감 --> lane x
            topValues[4] = std::make_pair(ps32_Histogram[s32_I], s32_I);
            std::sort(topValues, topValues + 5, std::greater<>());
        }
    }

    //결과 인덱스를 복사 
    for (s32_I = 0; s32_I < 5; ++s32_I) {
        ars32_resultIdxs[s32_I] = topValues[s32_I].second;
        if (topValues[s32_I].second == -1) // 초깃값일 경우 
        {
            s32_Cnt += 1;
        }
    }
    
    if(s32_Cnt == 5)  //초깃값 5개 --> 차선이 없다
        b_NoLane = true;
}

//###################################### FindClosestToMidPoint func ##################################################//

// 인덱스 중 중앙선과 가장 가까운 값을 반환
int32_t FindClosestToMidPoint(const int32_t points[5], int32_t s32_MidPoint) 
{
    int32_t s32_MinDistance = std::abs(points[0] - s32_MidPoint); 
    int32_t s32_ClosestIndex = points[0];
    int32_t s32_I;
    
    for (s32_I = 1; s32_I < 5; ++s32_I) {
        if (points[s32_I] == -1) continue; // 유효하지 않은 인덱스 건너뛰기

        int32_t currentDistance = std::abs(points[s32_I] - s32_MidPoint);
        if (currentDistance < s32_MinDistance) {
            s32_MinDistance = currentDistance;
            s32_ClosestIndex = points[s32_I];
        }
    }

    return s32_ClosestIndex;
}

//###################################### FindLaneStartPositions func ##################################################//

// 히스토그램 분석으로 좌·우 슬라이딩 윈도 시작 위치를 계산 --> 중앙과 가까운 차선을 찾고있음. 
void FindLaneStartPositions(const cv::Mat& st_Edge,
                            int32_t& s32_WindowCentorLeft,
                            int32_t& s32_WindowCentorRight,
                            bool& b_NoLaneLeft,
                            bool& b_NoLaneRight,
                            int32_t prev_left_start,
                            bool has_prev_left,
                            int32_t prev_right_start,
                            bool has_prev_right)
{
    // 0으로 초기화된 히스토그램
    std::vector<int32_t> histogram(st_Edge.cols, 0);

    for (int32_t col = 0; col < st_Edge.cols; ++col) {
        for (int32_t row = st_Edge.rows * 0.7; row < st_Edge.rows; ++row) {
            histogram[col] += st_Edge.at<uchar>(row, col) > 0 ? 1 : 0;
        }
    }

    int32_t ars32_LeftCandidate[5], ars32_RightCandidate[5];

    FindTop5MaxIndices(histogram.data(), st_Edge.cols / 2, ars32_LeftCandidate, b_NoLaneLeft);
    if (!b_NoLaneLeft) {
        s32_WindowCentorLeft = FindClosestToMidPoint(ars32_LeftCandidate, st_Edge.cols / 2);
    }

    FindTop5MaxIndices(histogram.data() + st_Edge.cols / 2,
                       st_Edge.cols - st_Edge.cols / 2,
                       ars32_RightCandidate,
                       b_NoLaneRight);
    if (!b_NoLaneRight) {
        for (int i = 0; i < 5; ++i) {
            if (ars32_RightCandidate[i] != -1) {
                ars32_RightCandidate[i] += st_Edge.cols / 2;
            }
        }
        s32_WindowCentorRight = FindClosestToMidPoint(ars32_RightCandidate, st_Edge.cols / 2);
    }

    const int max_jump_px = 40;  // 프레임당 허용 이동량 (튜닝 포인트)

    if (!b_NoLaneLeft && has_prev_left) {
        int dx = s32_WindowCentorLeft - prev_left_start;
        if (std::abs(dx) > max_jump_px) {
            // 새 히스토그램 peak가 너무 멀리 떨어져 있으면,
            // 이전 위치 근처까지만 따라가도록 clamp
            s32_WindowCentorLeft = prev_left_start + (dx > 0 ? max_jump_px : -max_jump_px);
        }
    }

    if (!b_NoLaneRight && has_prev_right) {
        int dx = s32_WindowCentorRight - prev_right_start;
        if (std::abs(dx) > max_jump_px) {
            s32_WindowCentorRight = prev_right_start + (dx > 0 ? max_jump_px : -max_jump_px);
        }
    }

}


//###################################### Parameter loader ##################################################//

// YAML 카메라 설정을 로드하고 기본 상태를 초기화
void LoadParam(CAMERA_DATA *CameraData)
{
    YAML::Node st_CameraParam = YAML::LoadFile("src/Params/config.yaml");
    std::cout << "Loading Camera Parameter from YAML File..." << std::endl;

    CameraData->st_CameraParameter.s_IPMParameterX = st_CameraParam["IPMParameterX"].as<std::string>();
    CameraData->st_CameraParameter.s_IPMParameterY = st_CameraParam["IPMParameterY"].as<std::string>();
    CameraData->st_CameraParameter.s32_RemapHeight = st_CameraParam["RemapHeight"].as<int32_t>();
    CameraData->st_CameraParameter.s32_RemapWidth  = st_CameraParam["RemapWidth"].as<int32_t>();

    std::cout << "Sucess to Load Camera Parameter!" << std::endl;
    // Kalman Object InitialLize
    CameraData->s32_KalmanObjectNum = 0;
    CameraData->f32_LastDistanceLeft = 0;
    CameraData->f32_LastAngleLeft = 0;
    CameraData->f32_LastDistanceRight = 0;
    CameraData->f32_LastAngleRight = 0;
}  

// IPM 맵핑 테이블을 파일에서 읽어 cv::Mat으로 구성
void LoadMappingParam(CAMERA_DATA *pst_CameraData) 
{

    // cv::Mat에서 원하는 DataType이 있기 때문에 s64_Value는 float형으로 설정해야 함
    float s64_Value;
    int32_t s32_Columns, s32_Rows;

    std::ifstream st_IPMParameters(pst_CameraData->st_CameraParameter.s_IPMParameterX);
    if (!st_IPMParameters.is_open()) {
        std::cerr << "Failed to open file: " << pst_CameraData->st_CameraParameter.s_IPMParameterX << std::endl;
        return;
    }
    st_IPMX.create(pst_CameraData->st_CameraParameter.s32_RemapHeight, pst_CameraData->st_CameraParameter.s32_RemapWidth, CV_32FC1);
    for (s32_Columns = 0; s32_Columns < pst_CameraData->st_CameraParameter.s32_RemapHeight; ++s32_Columns) {
        for (s32_Rows = 0; s32_Rows < pst_CameraData->st_CameraParameter.s32_RemapWidth; ++s32_Rows) {
            st_IPMParameters >> s64_Value;
            st_IPMX.at<float>(s32_Columns, s32_Rows) = s64_Value;
        }
    }
    st_IPMParameters.close();

    st_IPMParameters.open(pst_CameraData->st_CameraParameter.s_IPMParameterY);
    st_IPMY.create(pst_CameraData->st_CameraParameter.s32_RemapHeight, pst_CameraData->st_CameraParameter.s32_RemapWidth, CV_32FC1);
    for (s32_Columns = 0; s32_Columns < pst_CameraData->st_CameraParameter.s32_RemapHeight; ++s32_Columns) {
        for (s32_Rows = 0; s32_Rows <  pst_CameraData->st_CameraParameter.s32_RemapWidth; ++s32_Rows) {
            st_IPMParameters >> s64_Value;
            st_IPMY.at<float>(s32_Columns, s32_Rows) = s64_Value;
        }
    }
    st_IPMParameters.close();
}

// RANSAC 구현 및 Coefficient 추출 완료
// Data를 다루는 구조를 다시한번 생각 후 Kalman Filter까지 결합 진행



//################################################## get_lane_coef_from_kalman function ##################################################//
// 칼만 필터에서 추정된 차선 계수를 가져오는 함수
bool get_lane_coef_from_kalman(const CAMERA_DATA& cam_data,
                               LANE_COEFFICIENT& left_coef,
                               LANE_COEFFICIENT& right_coef,
                               bool& has_left,
                               bool& has_right)
{
    has_left  = false;
    has_right = false;

    for (int i = 0; i < cam_data.s32_KalmanObjectNum; ++i)
    {
        const auto& obj = cam_data.arst_KalmanObject[i];

        if (obj.b_IsLeft) {
            left_coef  = obj.st_LaneCoefficient;
            has_left   = true;
        } else {
            right_coef = obj.st_LaneCoefficient;
            has_right  = true;
        }
    }

    return has_left || has_right;
}
// ======================= Lane pair consistency (anchor 기반) ======================= //
// 두 직선에서 폭/각도 계산
static bool ComputeLaneWidthAngle(const LANE_COEFFICIENT& left,
                                  const LANE_COEFFICIENT& right,
                                  int img_height,
                                  double& width_px,
                                  double& angle_diff_deg)
{
    if (std::abs(left.f64_Slope) < 1e-6 || std::abs(right.f64_Slope) < 1e-6) {
        return false;
    }

    // 바닥 쪽에서 폭 측정 (IPM에서 rows-1가 차량 가까운 쪽이라고 가정)
    double y_ref = img_height - 1;

    double xL = (y_ref - left.f64_Intercept)  / left.f64_Slope;
    double xR = (y_ref - right.f64_Intercept) / right.f64_Slope;

    if (!std::isfinite(xL) || !std::isfinite(xR)) return false;
    if (xR <= xL) return false;   // 왼쪽/오른쪽 뒤바뀐 이상 상황

    width_px = xR - xL;

    // 각도 (deg)
    double theta_left  = std::atan(left.f64_Slope)  * 180.0 / M_PI;
    double theta_right = std::atan(right.f64_Slope) * 180.0 / M_PI;
    angle_diff_deg     = std::abs(theta_left - theta_right);

    return std::isfinite(width_px) && std::isfinite(angle_diff_deg);
}
bool EnforceLaneConsistencyAnchor(LANE_COEFFICIENT& left,
                                  LANE_COEFFICIENT& right,
                                  int img_height,
                                  bool left_anchor,
                                  double target_width_px,
                                  double min_width_px,
                                  double max_width_px,
                                  double max_angle_diff_deg,
                                  double alpha_anchor_pos,
                                  double alpha_other_pos,
                                  double alpha_anchor_slope,
                                  double alpha_other_slope)
{
    double width_now = 0.0, angle_diff = 0.0;
    if (!ComputeLaneWidthAngle(left, right, img_height, width_now, angle_diff)) {
        return false;
    }

    if (width_now < min_width_px || width_now > max_width_px) return false;
    if (angle_diff > max_angle_diff_deg) return false;

    double y_ref = img_height - 1;

    double xL = (y_ref - left.f64_Intercept)  / left.f64_Slope;
    double xR = (y_ref - right.f64_Intercept) / right.f64_Slope;

    double x_mid    = 0.5 * (xL + xR);
    double w_target = (1.0 - 0.4) * width_now + 0.4 * target_width_px;
    double xL_sym   = x_mid - 0.5 * w_target;
    double xR_sym   = x_mid + 0.5 * w_target;

    double mL_old = left.f64_Slope;
    double mR_old = right.f64_Slope;
    double m_mean = 0.5 * (mL_old + mR_old);

    double mL_new, mR_new;
    double xL_new, xR_new;


    if (left_anchor) {
        xL_new = (1.0 - alpha_anchor_pos) * xL + alpha_anchor_pos * xL_sym;
        xR_new = (1.0 - alpha_other_pos ) * xR + alpha_other_pos  * xR_sym;

        mL_new = (1.0 - alpha_anchor_slope) * mL_old + alpha_anchor_slope * m_mean;
        mR_new = (1.0 - alpha_other_slope ) * mR_old + alpha_other_slope  * m_mean;
    } else {
        xL_new = (1.0 - alpha_other_pos ) * xL + alpha_other_pos  * xL_sym;
        xR_new = (1.0 - alpha_anchor_pos) * xR + alpha_anchor_pos * xR_sym;

        mL_new = (1.0 - alpha_other_slope ) * mL_old + alpha_other_slope  * m_mean;
        mR_new = (1.0 - alpha_anchor_slope) * mR_old + alpha_anchor_slope * m_mean;
    }

    double cL_new = y_ref - mL_new * xL_new;
    double cR_new = y_ref - mR_new * xR_new;

    if (!std::isfinite(mL_new) || !std::isfinite(mR_new) ||
        !std::isfinite(cL_new) || !std::isfinite(cR_new)) {
        return false;
    }

    left.f64_Slope      = mL_new;
    left.f64_Intercept  = cL_new;
    right.f64_Slope     = mR_new;
    right.f64_Intercept = cR_new;

    return true;
}

// ############################################# publish_lane_messages func #############################################//


void CameraProcessing::publish_lane_messages()
{
    LANE_COEFFICIENT left_coef, right_coef;
    bool has_left = false, has_right = false;

    get_lane_coef_from_kalman(static_camera_data,
                              left_coef, right_coef,
                              has_left, has_right);

    const int H = static_camera_data.st_CameraParameter.s32_RemapHeight;
    const int W = static_camera_data.st_CameraParameter.s32_RemapWidth;

    if (lane_left_pub_ && has_left)
    {
        auto lane_msg = build_lane_msg_from_coef(left_coef, W, H);
        if (!lane_msg.lane_points.empty())
        {
            lane_left_pub_->publish(lane_msg);
        }
    }

    if (lane_right_pub_ && has_right)
    {
        auto lane_msg = build_lane_msg_from_coef(right_coef, W, H);
        if (!lane_msg.lane_points.empty())
        {
            lane_right_pub_->publish(lane_msg);
        }
    }
}


// ############################################# build_lane_msg_from_coef func #############################################//

perception::msg::Lane build_lane_msg_from_coef(const LANE_COEFFICIENT& coef,
                                               int img_width,
                                               int img_height,
                                               int num_samples )
{
    perception::msg::Lane lane_msg;

    // 너무 수평이면 일단은 보내되, 나중에 필터에서 처리하게 두는게 좋음
    if (!std::isfinite(coef.f64_Slope)) {
        return lane_msg; // 완전 이상한 경우만 버리기
    }

    for (int i = 0; i < num_samples; ++i)
    {
        double y_img = (img_height - 1)
                     - (img_height - 1) * (static_cast<double>(i) / (num_samples - 1));
        double x_img = (y_img - coef.f64_Intercept) / coef.f64_Slope;

        // 화면 밖이면 skip
        if (x_img < 0 || x_img >= img_width) continue;

        perception::msg::LanePnt p;
        p.x = static_cast<float>(x_img);
        p.y = static_cast<float>((img_height - 1) - y_img);

        lane_msg.lane_points.push_back(p);
    }

    return lane_msg;
}

//################################################## build_lane_message function ##################################################//

perception::msg::Lane build_lane_message(const CAMERA_LANEINFO & lane_info)
{
    perception::msg::Lane lane_msg;
    const int32_t max_samples = static_cast<int32_t>(sizeof(lane_info.arst_LaneSample) /
                                                    sizeof(lane_info.arst_LaneSample[0]));
    const int32_t clamped_samples = std::min(lane_info.s32_SampleCount, max_samples);
    lane_msg.lane_points.reserve(clamped_samples);

    for (int32_t i = 0; i < clamped_samples; ++i)
    {
        const cv::Point & sample = lane_info.arst_LaneSample[i];
        perception::msg::LanePnt point_msg;
        point_msg.x = static_cast<float>(sample.x);
        point_msg.y = static_cast<float>(sample.y);
        lane_msg.lane_points.push_back(point_msg);
    }

    return lane_msg;
}

}  // namespace perception

//################################################## Camera node main function ##################################################//

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::CameraProcessing>()); // 객체 생성 및 spin(이벤트 루프 홀출)
  rclcpp::shutdown();
  return 0;
}
