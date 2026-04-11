#pragma once
#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>


namespace rm_serial_driver {
class RMSerialDriver : public rclcpp::Node {
public:
    explicit RMSerialDriver(const rclcpp::NodeOptions &options);
    ~RMSerialDriver() override;

private:
    void getParams();
    void receiveData();
    void sendData();
    void joy_thread_func();
    void cmdvel_callback(geometry_msgs::msg::Twist::SharedPtr msg);
    void reopenPort();
    int  ConfigurePort(int);

    std::string device_name_;
    uint32_t    baud_rate_;
    int         serial_driver_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr                   cmd_vel_sub;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr base_encoder_pub;

    geometry_msgs::msg::Twist                      cmd_vel_msg;
    geometry_msgs::msg::TwistWithCovarianceStamped base_encoder_msg;

    struct JoyMsg {
        float left_x;
        float left_y;
        float right_x;
    };
    std::atomic<JoyMsg> joy_msg_;

    std::thread receive_thread_;
    std::thread send_thread_;
    std::thread joy_thread_;

    std::mutex serial_mtx;
};
} // namespace rm_serial_driver

#endif // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
