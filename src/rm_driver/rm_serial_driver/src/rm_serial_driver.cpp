#include "rm_serial_driver/rm_serial_driver.hpp"

#include <SDL2/SDL.h>
#include <cmath>
#include <limits>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>

#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "rm_serial_driver/crc.hpp"
#include "rm_serial_driver/packet.hpp"


namespace rm_serial_driver {

using namespace std::chrono_literals;

RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions &options)
    : Node("rm_serial_driver") {
    RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");

    getParams();

    // Create Publisher
    base_encoder_pub = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/base/twist0", rclcpp::SensorDataQoS());
    battery_state_pub = this->create_publisher<sensor_msgs::msg::BatteryState>("/battery_state", rclcpp::SensorDataQoS());

    // Create Subscription
    cmd_vel_sub = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel_nav", rclcpp::SensorDataQoS(),
        std::bind(&RMSerialDriver::cmdvel_callback, this, std::placeholders::_1));

    if (!access(device_name_.c_str(), F_OK)) {
        serial_driver_ = open(device_name_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (serial_driver_ < 0) {
            RCLCPP_ERROR(get_logger(), "no serial %s", device_name_.c_str());
            exit(-1);
        }
        fcntl(serial_driver_, F_SETFL, FNDELAY);
        RCLCPP_INFO(get_logger(), "%s port is open.", device_name_.c_str());
        ConfigurePort(serial_driver_);
    } else {
        RCLCPP_ERROR(get_logger(), "no serial %s", device_name_.c_str());
        exit(-1);
    }
    receive_thread_ = std::thread(&RMSerialDriver::receiveData, this);
    send_thread_    = std::thread(&RMSerialDriver::sendData, this);
    joy_thread_     = std::thread(&RMSerialDriver::joy_thread_func, this);
    RCLCPP_INFO(get_logger(), "Opening serial port: %s", device_name_.c_str());
    RCLCPP_INFO(get_logger(), "Successfully opened serial port: %s", device_name_.c_str());
}

RMSerialDriver::~RMSerialDriver() {
    // if (receive_thread_.joinable()) {
    //     receive_thread_.join();
    // }
    if (send_thread_.joinable()) {
        send_thread_.join();
    }
    SendPacket packet;
    packet.length = sizeof(packet);
    packet.Vx     = 0;
    packet.Vy     = 0;
    packet.Vw     = 0;
    CRC::appendCRC8CheckSum(reinterpret_cast<uint8_t *>(&packet), 3);
    CRC::appendCRC16CheckSum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));
    {
        std::lock_guard<std::mutex> locker(serial_mtx);
        write(serial_driver_, &packet, sizeof(SendPacket));
        tcdrain(serial_driver_);
    }
    close(serial_driver_);
}

void RMSerialDriver::cmdvel_callback(geometry_msgs::msg::Twist::SharedPtr msg) {
    cmd_vel_msg.linear.x  = msg->linear.x;
    cmd_vel_msg.linear.y  = msg->linear.y;
    cmd_vel_msg.angular.z = msg->angular.z;
}

int RMSerialDriver::detectBatteryCellCount(float total_voltage) const {
    constexpr float kMinCellVoltage     = 2.8f;
    constexpr float kMaxCellVoltage     = 4.35f;
    constexpr float kNominalCellVoltage = 3.7f;
    constexpr int   kCandidates[]       = {3, 4, 6};

    if (!std::isfinite(total_voltage) || total_voltage <= 0.0f) {
        return 0;
    }

    if (battery_cell_count_ > 0) {
        const float cell_voltage = total_voltage / static_cast<float>(battery_cell_count_);
        if (cell_voltage >= kMinCellVoltage && cell_voltage <= kMaxCellVoltage) {
            return battery_cell_count_;
        }
    }

    int   best_cell_count = 0;
    float best_score      = std::numeric_limits<float>::max();

    for (int cell_count : kCandidates) {
        const float cell_voltage = total_voltage / static_cast<float>(cell_count);
        if (cell_voltage < kMinCellVoltage || cell_voltage > kMaxCellVoltage) {
            continue;
        }

        const float score = std::fabs(cell_voltage - kNominalCellVoltage);
        if (score < best_score) {
            best_score      = score;
            best_cell_count = cell_count;
        }
    }

    return best_cell_count;
}

void RMSerialDriver::receiveData() {
    std::vector<uint8_t> data;
    ReceivePacket        packet;

    size_t  data_size;
    size_t  read_size;
    int     rx_fail_cnt = 0;
    int64_t time_offset_ns = 0;

    data.reserve(1000);

    while (rclcpp::ok()) {
        std::this_thread::sleep_for(10ms);
        ioctl(serial_driver_, FIONREAD, &read_size);
        int64_t ros_now_ns = this->now().nanoseconds();
        if (read_size + data.size() < sizeof(ReceivePacket)) {
            continue;
        }
        uint8_t *tail_ptr = data.data() + data.size();
        data.resize(data.size() + read_size);
        {
            std::lock_guard<std::mutex> locker(serial_mtx);
            read(serial_driver_, tail_ptr, read_size);
        }
        data_size = data.size();
        // if (data_size > 0) {
        //     printf("\t");
        //     for (size_t i = 0; i < data_size && rclcpp::ok(); ++i) {
        //         printf("%x ", data[i]);
        //     }
        //     printf("\n");
        // }

        bool rx_ok = false;
        for (int i = data_size - sizeof(ReceivePacket); i >= 0 && rclcpp::ok(); --i) {
            if (data[i] != serial_header || data[i + 1] != sizeof(ReceivePacket)) continue;

            bool crc_ok = CRC::verifyCRC16CheckSum(data.data() + i, sizeof(ReceivePacket));
            if (!crc_ok) {
                RCLCPP_WARN(get_logger(), "CRC ERROR");
                for (size_t ii = i; ii < i + sizeof(ReceivePacket); ++ii) {
                    printf("%x ", data[ii]);
                }
                printf("\n");
                for (size_t i = 0; i < data_size && rclcpp::ok(); ++i) {
                    printf("%x ", data[i]);
                }
                printf("\n");
                continue;
            }

            memcpy(&packet, data.data() + i, sizeof(ReceivePacket));

            int64_t stm32_now_ns = packet.time * 1000;
            time_offset_ns       = (ros_now_ns - stm32_now_ns) / 10 + time_offset_ns / 10 * 9;

            // std::cout << packet << std::endl;

            base_encoder_msg.header.stamp          = rclcpp::Time(stm32_now_ns + time_offset_ns);
            base_encoder_msg.header.frame_id       = "base_link";
            base_encoder_msg.twist.twist.linear.x  = -packet.v;
            base_encoder_msg.twist.twist.angular.z = packet.omega;

            // 填协方差
            for (int i = 0; i < 36; i++) base_encoder_msg.twist.covariance[i] = 0.0;
            base_encoder_msg.twist.covariance[0]  = 0.01; // vx variance
            base_encoder_msg.twist.covariance[35] = 0.05; // wz variance
            base_encoder_msg.twist.covariance[7]  = 1e6;  // vy (不可用)
            base_encoder_msg.twist.covariance[14] = 1e6;  // vz
            base_encoder_msg.twist.covariance[21] = 1e6;  // roll
            base_encoder_msg.twist.covariance[28] = 1e6;  // pitch

            base_encoder_pub->publish(base_encoder_msg);

            const int detected_cell_count = detectBatteryCellCount(packet.volt);
            if (detected_cell_count != battery_cell_count_) {
                battery_cell_count_ = detected_cell_count;
                if (battery_cell_count_ > 0) {
                    RCLCPP_INFO(
                        get_logger(), "Detected battery pack: %dS, total voltage: %.2f V",
                        battery_cell_count_, packet.volt);
                } else {
                    RCLCPP_WARN(
                        get_logger(), "Unable to determine battery cell count from %.2f V", packet.volt);
                }
            }

            battery_state_msg.header.stamp = base_encoder_msg.header.stamp;
            battery_state_msg.header.frame_id = "base_link";
            battery_state_msg.voltage = packet.volt;
            battery_state_msg.temperature = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.current = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.charge = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.capacity = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.design_capacity = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.percentage = std::numeric_limits<float>::quiet_NaN();
            battery_state_msg.power_supply_status = sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_UNKNOWN;
            battery_state_msg.power_supply_health = sensor_msgs::msg::BatteryState::POWER_SUPPLY_HEALTH_UNKNOWN;
            battery_state_msg.power_supply_technology = sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LIPO;
            battery_state_msg.present = packet.volt > 0.0f;
            battery_state_msg.location = "battery_pack";
            battery_state_msg.serial_number.clear();
            battery_state_msg.cell_voltage.clear();
            battery_state_msg.cell_temperature.clear();

            if (battery_cell_count_ > 0) {
                const float cell_voltage = packet.volt / static_cast<float>(battery_cell_count_);
                battery_state_msg.cell_voltage.assign(
                    static_cast<size_t>(battery_cell_count_), cell_voltage);
            }

            battery_state_pub->publish(battery_state_msg);

            size_t next_idx = i + sizeof(ReceivePacket);
            if (data_size > next_idx)
                memmove(data.data(), data.data() + next_idx, data_size - next_idx);
            data.resize(data_size - next_idx);

            rx_ok       = true;
            rx_fail_cnt = 0;
            break;
        }

        if (rx_ok == false) rx_fail_cnt++;
        if (rx_fail_cnt > 1) {
            RCLCPP_ERROR(get_logger(), "SERIAL FRAME ERROR");
            printf("\nerror serial data\n");
            for (size_t i = 0; i < data.size() && rclcpp::ok(); ++i) {
                printf("%x ", data[i]);
            }
            printf("\n\n");
        }

        if (data.size() >= 2 * sizeof(ReceivePacket)) {
            memcpy(data.data(), data.data() + data.size() - sizeof(ReceivePacket), sizeof(ReceivePacket));
            data.resize(sizeof(ReceivePacket));
        }
    }
}

float map(float value, float old_min, float old_max, float new_min, float new_max) {
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min);
}

void RMSerialDriver::sendData() {
    JoyMsg joy_msg;
    while (rclcpp::ok()) {
        try {
            SendPacket packet;
            packet.length = sizeof(packet);

            packet.Vx = float(cmd_vel_msg.linear.x);
            packet.Vy = float(cmd_vel_msg.linear.y);
            packet.Vw = float(cmd_vel_msg.angular.z);

            {
                std::lock_guard<std::mutex> locker(joy_mtx);
                joy_msg = joy_msg_;
            }
            if (std::abs(joy_msg.left_x) > 3000 || std::abs(joy_msg.left_y) > 3000 || std::abs(joy_msg.right_x) > 3000) {
                packet.Vx = map(-joy_msg.left_y, 0, 32678, 0, 0.8);
                // packet.Vy = map(-joy_msg.left_x, 0, 32678, 0, 1);
                packet.Vw = map(-joy_msg.right_x, 0, 32678, 0, 2.5);
            }
            packet.Vx = -packet.Vx;

            CRC::appendCRC16CheckSum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

            // printf("%.6f, %.6f, %.6f\n", packet.Vx, packet.Vy, packet.Vw);

            // for (size_t i = 0; i < sizeof(SendPacket); i++) {
            //     printf("%x ", std::bit_cast<uint8_t *>(&packet)[i]);
            // }
            // printf("\n");
            {
                std::lock_guard<std::mutex> locker(serial_mtx);
                write(serial_driver_, &packet, sizeof(SendPacket));
                tcdrain(serial_driver_);
            }
            std::this_thread::sleep_for(10ms);
        } catch (const std::exception &ex) {
            RCLCPP_ERROR(get_logger(), "Error while sending data: %s", ex.what());
            reopenPort();
        }
    }
}

void RMSerialDriver::joy_thread_func() {
    if (SDL_Init(SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS) < 0) {
        RCLCPP_ERROR(get_logger(), "SDL_Init failed: %s", SDL_GetError());
        return;
    }

    SDL_GameController *controller = nullptr;
    JoyMsg              joy_msg;

    RCLCPP_INFO(get_logger(), "SDL initialized. Waiting for controller...");

    SDL_Event event;

    while (rclcpp::ok()) {

        // 处理 SDL 事件（包括手柄插入/拔出）
        while (SDL_PollEvent(&event)) {

            if (event.type == SDL_CONTROLLERDEVICEADDED) {
                int index = event.cdevice.which;
                RCLCPP_INFO(get_logger(), "Controller added: index=%d", index);

                if (!controller) {
                    controller = SDL_GameControllerOpen(index);
                    if (controller) {
                        RCLCPP_INFO(get_logger(), "Controller opened!");
                    }
                }
            }

            else if (event.type == SDL_CONTROLLERDEVICEREMOVED) {
                RCLCPP_WARN(get_logger(), "Controller removed");

                if (controller) {
                    SDL_GameControllerClose(controller);
                    controller = nullptr;
                }
            }
        }

        // 手柄连接后才能读取
        if (controller) {
            joy_msg.left_x  = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTX);
            joy_msg.left_y  = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTY);
            joy_msg.right_x = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTX);

            // std::cout << joy_msg.left_x << " " << joy_msg.left_y << " " << joy_msg.right_x <<std::endl;

            {
                std::lock_guard<std::mutex> locker(joy_mtx);
                joy_msg_ = joy_msg;
            }
        }

        SDL_Delay(20);
    }

    if (controller) {
        SDL_GameControllerClose(controller);
    }

    SDL_Quit();
}

void RMSerialDriver::getParams() {
    try {
        device_name_ = declare_parameter<std::string>("device_name", "");
    } catch (rclcpp::ParameterTypeException &ex) {
        RCLCPP_ERROR(get_logger(), "The device name provided was invalid");
        throw ex;
    }
}

void RMSerialDriver::reopenPort() {
    RCLCPP_WARN(get_logger(), "Attempting to reopen port");
}

int RMSerialDriver::ConfigurePort(int fd) {
    struct termios port_settings; // structure to store the port settings in
    memset(&port_settings, 0, sizeof(port_settings));
    cfsetispeed(&port_settings, B921600); // set baud rates B460800 B921600
    cfsetospeed(&port_settings, B921600);
    /* Enable the receiver and set local mode...*/
    port_settings.c_cflag |= (CLOCAL | CREAD);
    /* Set c_cflag options.*/
    port_settings.c_cflag &= ~PARENB; // set no parity, stop bits, data bits
    port_settings.c_cflag &= ~PARODD;
    port_settings.c_cflag &= ~CSTOPB;
    port_settings.c_cflag &= ~CSIZE;
    port_settings.c_cflag |= CS8;
    // port_settings.c_cflag &= ~CRTSCTS;
    port_settings.c_iflag &= ~(IXON | IXOFF | IXANY);
    port_settings.c_iflag &= ~(INLCR | IGNCR | ICRNL);
    port_settings.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    port_settings.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP);
    /* Set c_oflag output options */
    port_settings.c_oflag &= ~OPOST;
    /* Set the timeout options */
    port_settings.c_cc[VTIME] = 0;
    port_settings.c_cc[VMIN]  = 0;
    tcsetattr(fd, TCSANOW, &port_settings); // apply the settings to the port
    return fd;
}
} // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)
