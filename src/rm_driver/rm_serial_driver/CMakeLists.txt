cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME rm_serial_driver)
project(${PROJECT_NAME})

## Use C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(SDL2 REQUIRED)
include_directories(${PROJECT_NAME} ${SDL2_INCLUDE_DIRS})

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
        src/rm_serial_driver.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
        PLUGIN rm_serial_driver::RMSerialDriver
        EXECUTABLE ${PROJECT_NAME}_node
)

target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} atomic)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
          ament_cmake_cppcheck
          ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()
endif()

#############
## Install ##
#############


ament_auto_package(
        INSTALL_TO_SHARE
        config
        launch
)
