// 加载本地一个图像，运行单次检测任务并显示结果（不保存）
// 包含推理和工具两个本地头文件
#include "inference.h"
#include "utils.h"

// 基本的输入输出流功能(cin)
#include <iostream>
// OpenCV库中的一个头文件，提供了高级GUI（图形用户界面）功能, highgui 模块主要用于图像窗口的显示、图像的读取和保存、滑动条的创建等。
#include <opencv2/highgui.hpp>

// 主函数的开始，argc 表示命令行参数的数量，argv 是一个指向参数字符串数组的指针。
int main(const int argc, const char **argv) {
	// 检查程序是否接收到了正确的参数数量。这里期望的是3个参数：程序名、模型路径和图像路径。
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
		return 1; // 返回1表示程序因为错误而终止
	}

	// 将第二个参数（模型路径）转换为字符串并存储。将第三个参数（图像路径）转换为字符串并存储。
	const std::string model_path = argv[1];
	const std::string image_path = argv[2];

	// 构建了元数据文件的路径。它查找模型路径中最后一个斜杠的位置，并构建包含 metadata.yaml 的完整路径。
	// 要求metadata.yaml和模型在同一目录，其中names字段是class_names，获取所有类名的向量
    const std::size_t pos = model_path.find_last_of("/");
	const std::string metadata_path = model_path.substr(0, pos + 1) + "metadata.yaml";
	const std::vector<std::string> class_names = GetClassNameFromMetadata(metadata_path);

	// 使用OpenCV的 imread 函数读取图像，并检查图像是否成功加载
	cv::Mat image = cv::imread(image_path);
	if (image.empty()) {
		std::cerr << "ERROR: image is empty" << std::endl;
		return 1;
	}

	// 设置置信度阈值，只有当检测对象的置信度高于这个值时，才会被认为是有效的从而被显示出来
	const float confidence_threshold = 0.5;

	// 实例化一个 yolo::Inference 对象，用于加载模型并设置置信度阈值。
	yolo::Inference inference(model_path, confidence_threshold);
	// 调用 RunInference 函数进行对象检测，并获取检测结果。
	std::vector<yolo::Detection> detections = inference.RunInference(image);

	// 调用 DrawDetectedObject 函数在图像上绘制检测到的对象。
	DrawDetectedObject(image, detections, class_names);
	// 使用OpenCV的 imshow 函数显示图像。
	cv::imshow("image", image);

	// 定义退出键（ESC键），其ASCII值为27。
	const char escape_key = 27;
	// 等待用户按键，如果按下的是ESC键，则退出循环。
	while (cv::waitKey(0) != escape_key);
	// 销毁所有OpenCV创建的窗口。
	cv::destroyAllWindows();

	return 0;
}
