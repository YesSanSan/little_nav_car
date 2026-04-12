/*
 * @Author: 张博文
 * @Date: 2023-12-02 23:47:11
 * @Last Modified by:   张博文
 * @Last Modified time: 2023-12-02 23:47:11
 */

#pragma once
#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <vector>


namespace rm_serial_driver {
constexpr uint8_t serial_header = 0x7E;

// 通信接收结构体
struct ReceivePacket {
    uint8_t  header = serial_header;
    uint8_t  length;
    float    v;     // 线速度
    float    omega; // 角速度
    float    volt;  // 电压
    uint64_t time;  // 时间戳 (us)
    uint16_t crc16;

    // 友元：让 cout << obj 可用
    friend std::ostream &operator<<(std::ostream &os, const ReceivePacket &p) {
        os << "ReceivePacket{"
           << "header=" << +p.header << ", "
           << "length=" << +p.length << ", "
           << "v=" << p.v << ", "
           << "omega=" << p.omega << ", "
           << "time=" << p.time << ", "
           << "crc16=0x" << std::hex << p.crc16 << std::dec
           << "}";
        return os;
    }
} __attribute__((packed));

// 数据发送结构体
struct SendPacket {
    uint8_t header = 0xD4;
    uint8_t length;

    float Vx; // x速度
    float Vy; // y速度
    float Vw;

    uint16_t check_num16;
} __attribute__((packed));

} // namespace rm_serial_driver

#endif // RM_SERIAL_DRIVER__PACKET_HPP_
