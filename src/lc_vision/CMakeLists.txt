cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME lc_vision)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-Wall)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  DEFAULT_ASTRA_SDK_ROOT
  "/home/cmls/sanwu/AstraSDK-v2.1.3-Ubuntu-x86_64/AstraSDK-v2.1.3-94bca0f52e-20210608T062039Z-Ubuntu18.04-x86_64"
)
if(DEFINED ENV{ASTRA_SDK_ROOT} AND NOT "$ENV{ASTRA_SDK_ROOT}" STREQUAL "")
  set(DEFAULT_ASTRA_SDK_ROOT "$ENV{ASTRA_SDK_ROOT}")
endif()
set(ASTRA_SDK_ROOT "${DEFAULT_ASTRA_SDK_ROOT}" CACHE PATH "Path to the Astra SDK root")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(PkgConfig REQUIRED)

set(DEFAULT_OPENVINO_ROOT "$ENV{HOME}/intel/openvino_2026")
if(DEFINED ENV{OPENVINO_ROOT} AND NOT "$ENV{OPENVINO_ROOT}" STREQUAL "")
  set(DEFAULT_OPENVINO_ROOT "$ENV{OPENVINO_ROOT}")
endif()
set(OPENVINO_ROOT "${DEFAULT_OPENVINO_ROOT}" CACHE PATH "Path to the OpenVINO installation root")

if(NOT OpenVINO_DIR AND EXISTS "${OPENVINO_ROOT}/runtime/cmake/OpenVINOConfig.cmake")
  set(OpenVINO_DIR "${OPENVINO_ROOT}/runtime/cmake")
endif()

if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|X86_64|amd64|AMD64)$")
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  else()
    set(VCPKG_TARGET_TRIPLET "arm64-linux")
  endif()
endif()

if(NOT ncnn_DIR AND EXISTS "$ENV{HOME}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/share/ncnn/ncnnConfig.cmake")
  set(ncnn_DIR "$ENV{HOME}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/share/ncnn")
endif()

if(EXISTS "$ENV{HOME}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/share/glslang/glslang-config.cmake")
  set(glslang_DIR "$ENV{HOME}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/share/glslang" CACHE PATH "Path to glslang config" FORCE)
endif()

find_package(ncnn CONFIG QUIET)
find_package(OpenVINO QUIET)

set(LC_VISION_HAVE_NCNN OFF)
if(ncnn_FOUND)
  set(LC_VISION_HAVE_NCNN ON)
endif()

set(LC_VISION_HAVE_OPENVINO OFF)
if(OpenVINO_FOUND)
  set(LC_VISION_HAVE_OPENVINO ON)
endif()

set(LC_VISION_DETECTOR_FRAMEWORK "none")
if(LC_VISION_HAVE_OPENVINO)
  set(LC_VISION_DETECTOR_FRAMEWORK "openvino")
elseif(LC_VISION_HAVE_NCNN)
  set(LC_VISION_DETECTOR_FRAMEWORK "ncnn")
endif()

message(STATUS "lc_vision detector framework selection: ${LC_VISION_DETECTOR_FRAMEWORK}")
if(LC_VISION_HAVE_OPENVINO)
  message(STATUS "OpenVINO detected at ${OPENVINO_ROOT}")
endif()
if(LC_VISION_HAVE_NCNN)
  message(STATUS "NCNN detected via CMake package")
endif()
if(NOT LC_VISION_HAVE_OPENVINO AND NOT LC_VISION_HAVE_NCNN)
  message(WARNING "Neither OpenVINO nor NCNN was found; lc_vision will build without detector support")
endif()

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libswscale)

find_library(ASTRA_LIBRARY NAMES astra HINTS "${ASTRA_SDK_ROOT}/lib" REQUIRED)
find_library(ASTRA_CORE_LIBRARY NAMES astra_core HINTS "${ASTRA_SDK_ROOT}/lib" REQUIRED)
find_library(ASTRA_CORE_API_LIBRARY NAMES astra_core_api HINTS "${ASTRA_SDK_ROOT}/lib" REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/lc_vision.cpp
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASTRA_SDK_ROOT}/include
)

target_link_directories(${PROJECT_NAME}
  PRIVATE
    "${ASTRA_SDK_ROOT}/lib"
)

target_link_libraries(${PROJECT_NAME}
  ${ASTRA_LIBRARY}
  ${ASTRA_CORE_LIBRARY}
  ${ASTRA_CORE_API_LIBRARY}
  PkgConfig::FFMPEG
  ${OpenCV_LIBRARIES}
  pthread
)

if(LC_VISION_DETECTOR_FRAMEWORK STREQUAL "openvino")
  target_link_libraries(${PROJECT_NAME} openvino::runtime)
elseif(LC_VISION_DETECTOR_FRAMEWORK STREQUAL "ncnn")
  target_link_libraries(${PROJECT_NAME} ncnn)
endif()

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    LC_VISION_HAVE_NCNN=$<IF:$<BOOL:${LC_VISION_HAVE_NCNN}>,1,0>
    LC_VISION_HAVE_OPENVINO=$<IF:$<BOOL:${LC_VISION_HAVE_OPENVINO}>,1,0>
    LC_VISION_DETECTOR_FRAMEWORK_NCNN=$<IF:$<STREQUAL:${LC_VISION_DETECTOR_FRAMEWORK},ncnn>,1,0>
    LC_VISION_DETECTOR_FRAMEWORK_OPENVINO=$<IF:$<STREQUAL:${LC_VISION_DETECTOR_FRAMEWORK},openvino>,1,0>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${ASTRA_SDK_ROOT}/lib;${OPENVINO_ROOT}/runtime/lib/intel64"
  INSTALL_RPATH "${ASTRA_SDK_ROOT}/lib;${OPENVINO_ROOT}/runtime/lib/intel64"
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN lc_vision::LCVision
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_cppcheck
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()
endif()

install(FILES
  README.md
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  FILES_MATCHING
  PATTERN ".ultralytics-venv" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pt" EXCLUDE
  PATTERN "*.py" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.xml"
    PATTERN "*.param"
    PATTERN "*.bin"
    PATTERN "*.mapping"
    PATTERN "metadata.yaml"
)

ament_auto_package(
  INSTALL_TO_SHARE
    config
    launch
    scripts
)
