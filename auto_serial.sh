#!/bin/bash
cmds=(
        #"ros2 launch livox_ros_driver2 msg_MID360_launch.py"
        #"ros2 launch fast_lio mapping_mid360.launch.py"
        #"ros2 launch sentry localization.launch.py"
        #"ros2 launch linefit_ground_segmentation_ros segmentation.launch.py"
        #"ros2 launch pointcloud_to_laserscan pointcloud_to_laserscan_launch.py"
        "ros2 launch rm_serial_driver serial_driver.launch.py "
        #"ros2 launch rm_navigation bringup_launch.py"
        #"ros2 launch rm_decision rm_decision.launch.py"
        #"ros2 launch rm_navigation rviz_launch.py"
)

for cmd in "${cmds[@]}";
do
        #gnome-terminal -- bash -c "cd /home/navigation/evolution-sentry;source install/setup.bash;$cmd;exec bash;"
        bash -c "cd /home/navigation/evolution-sentry;source install/setup.bash;$cmd;exec bash;"
        sleep 0.2
done
