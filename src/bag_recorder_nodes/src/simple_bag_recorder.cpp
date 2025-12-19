#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>

#include <sensor_msgs/msg/compressed_image.hpp>

#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_storage/storage_options.hpp>

using std::placeholders::_1;

class SimpleBagRecorder : public rclcpp::Node
{
public:
  SimpleBagRecorder()
  : Node("simple_bag_recorder")
  {
    // 1) StorageOptions / ConverterOptions 설정
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = "final_test1";   // 폴더 이름
    storage_options.storage_id = "sqlite3";

    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format  = "cdr";
    converter_options.output_serialization_format = "cdr";

    // 2) writer 생성 + open
    writer_ = std::make_unique<rosbag2_cpp::Writer>();
    writer_->open(storage_options, converter_options);

    // (선택) 미리 topic 등록 – 없어도 write()가 알아서 등록해주긴 함
    // writer_->create_topic({
    //   "/camera/camera/color/image_raw/compressed",
    //   "sensor_msgs/msg/CompressedImage",
    //   "cdr",
    //   ""
    // });

    // 3) /camera/camera/color/image_raw/compressed 구독
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/camera/camera/color/image_raw/compressed",
      10,  // 필요하면 rclcpp::SensorDataQoS() 로 변경 가능
      std::bind(&SimpleBagRecorder::topic_callback, this, _1));
  }

private:
  void topic_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    // 1) 메시지 → SerializedMessage로 직렬화
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer;
    auto serialized_msg = std::make_shared<rclcpp::SerializedMessage>();
    serializer.serialize_message(msg.get(), serialized_msg.get());

    // 2) 타임스탬프
    rclcpp::Time time_stamp = this->now();

    // 3) rosbag에 쓰기 (새 API: shared_ptr<SerializedMessage> 사용)
    writer_->write(
      serialized_msg,
      "/camera/camera/color/image_raw/compressed",
      "sensor_msgs/msg/CompressedImage",
      time_stamp);
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleBagRecorder>());
  rclcpp::shutdown();
  return 0;
}
