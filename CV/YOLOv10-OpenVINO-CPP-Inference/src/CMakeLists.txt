# 配置和构建一个使用OpenCV、yaml-cpp和OpenVINO的YOLOv10项目
cmake_minimum_required(VERSION 3.10.2)			# 设置CMake最低版本
project(YOLOv10_OpenVINO)						# 定义了项目的名称为YOLOv10_OpenVINO

set(CMAKE_CXX_STANDARD 14)						# 指定了项目使用的C++标准为C++14。

find_package(OpenCV REQUIRED)					# 查找并加载OpenCV和yaml-cpp库。
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /opt/intel/openvino/runtime/include)		# 添加OpenCV和OpenVINO的头文件搜索路径。

set(SOURCES										# 定义了两个源文件（utils/inference）
	utils.cc
	inference.cc
)

add_library(inference_lib ${SOURCES})			# 并将它们编译成一个名为inference_lib的库。
target_link_libraries(inference_lib ${OpenCV_LIBS} /opt/intel/openvino/runtime/lib/intel64/libopenvino.so yaml-cpp)		# 为inference_lib库链接OpenCV、OpenVINO和yaml-cpp库。

add_executable(camera main_camera.cc)			# 定义了三个可执行文件：camera、detect和video。
add_executable(detect main_detect.cc)
add_executable(video main_video.cc)

target_link_libraries(camera inference_lib)		# 将inference_lib库链接到三个可执行文件上。
target_link_libraries(detect inference_lib)
target_link_libraries(video inference_lib)
