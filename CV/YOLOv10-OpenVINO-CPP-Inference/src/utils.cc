#include "utils.h"

// 对文件输入输出流的支持。它允许程序读写文件，例如创建文件、打开文件、读取文件内容、写入文件等。
#include <fstream>
#include <random>
// 第三方库的头文件，用于解析和生成YAML（YAML Ain't Markup Language）格式的数据。
#include <yaml-cpp/yaml.h>


/*
在图像帧上绘制检测到的对象。它接受三个参数：
cv::Mat &frame：图像帧，用于绘制检测框和文本。
const std::vector<yolo::Detection> &detections：包含检测到的对象的向量。
const std::vector<std::string> &class_names：包含类名的向量。
*/
void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections, const std::vector<std::string> &class_names) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(120, 255);
	
	// 遍历每个检测到的对象
	for (const auto &detection : detections) {
		// 获取检测框（cv::Rect）、置信度（confidence）和类ID（class_id）。
		const cv::Rect &box = detection.box;
		const float &confidence = detection.confidence;
		const int &class_id = detection.class_id;
		
		// 根据随机颜色绘制检测框。
		const cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
		cv::rectangle(frame, box, color, 3);

		// 根据是否提供了类名，构建一个包含类名和置信度的字符串。
		std::string class_string;

		if (class_names.empty())
			class_string = "id[" + std::to_string(class_id) + "] " + std::to_string(confidence).substr(0, 4);
		else
			class_string = class_names[class_id] + " " + std::to_string(confidence).substr(0, 4);
		
		// 计算文本大小，以确定文本框的位置和大小。
		const cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, 0);
		const cv::Rect text_box(box.x - 2, box.y - 27, text_size.width + 10, text_size.height + 15);
		// 在图像帧上绘制文本框，并在其中放置文本。
		cv::rectangle(frame, text_box, color, cv::FILLED);
		cv::putText(frame, class_string, cv::Point(box.x + 5, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, 0);
	}
}

/*
从元数据文件中读取类名。它接受一个参数：
const std::string &metadata_path：元数据文件的路径。
*/
std::vector<std::string> GetClassNameFromMetadata(const std::string &metadata_path) {
	std::ifstream check_file(metadata_path);
	// 尝试打开指定路径的文件，如果失败则输出错误信息并返回空向量。
	if (!check_file.is_open()) {
		std::cerr << "Unable to open file: " << metadata_path << std::endl;
		return {};
	}

	check_file.close();

	// 使用 YAML::LoadFile 加载 YAML 文件
	YAML::Node metadata = YAML::LoadFile(metadata_path);
	std::vector<std::string> class_names;

	// 检查 YAML 文件中是否存在 names 节点，如果不存在则输出错误信息并返回空向量。
	if (!metadata["names"]) {
		std::cerr << "ERROR: 'names' node not found in the YAML file" << std::endl;
		return {};
	}
	// 遍历 names 节点中的每个条目，将其作为字符串读取并添加到类名向量中。
	for (int i = 0; i < metadata["names"].size(); ++i) {
		std::string class_name = metadata["names"][std::to_string(i)].as<std::string>();
		class_names.push_back(class_name);
	}

	// 返回所有class_name
	return class_names;
}
