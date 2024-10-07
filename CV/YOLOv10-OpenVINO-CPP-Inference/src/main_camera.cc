#include "inference.h"
#include "utils.h"

// C++11标准引入的一个头文件，它定义了一组固定宽度的整数类型。这些类型包括：int8_t：8位带符号整数，uint8_t：8位无符号整数等
#include <cstdint>
#include <iostream>
#include <opencv2/highgui.hpp>

int main(const int argc, const char **argv) {
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <camera_index>" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const uint8_t camera_index = std::stoi(argv[2]);

  	const std::size_t pos = model_path.find_last_of("/");
	const std::string metadata_path = model_path.substr(0, pos + 1) + "metadata.yaml";
	const std::vector<std::string> class_names = GetClassNameFromMetadata(metadata_path);

	// 使用OpenCV的 VideoCapture 类创建一个视频捕获对象，传入摄像头索引。检查摄像头是否成功打开。
	cv::VideoCapture capture(camera_index);

	if (!capture.isOpened()) {
		std::cerr << "ERROR: Could not open the camera" << std::endl;
		return 1;
	}

	const float confidence_threshold = 0.5;

	yolo::Inference inference(model_path, confidence_threshold);

	// 声明一个 cv::Mat 对象用于存储从摄像头捕获的帧。不同于detect从本地cv::Mat image = cv::imread(image_path);
	cv::Mat frame;

	const char escape_key = 27;

	while (true) {
		// 从视频捕获设备读取一帧，并检查捕获的帧是否为空。
		capture >> frame;
		if (frame.empty()) {
			std::cerr << "ERROR: Frame is empty" << std::endl;
			break;
		}

		// 调用 RunInference 函数进行对象检测，并获取检测结果，然后绘制展示，并判断是否退出
		std::vector<yolo::Detection> detections = inference.RunInference(frame);

		DrawDetectedObject(frame, detections, class_names);

		cv::imshow("camera", frame);

		if (cv::waitKey(1) == escape_key) {
			break;
		}
	}
	// 释放视频捕获设备。
	capture.release();
	cv::destroyAllWindows();

	return 0;
}
