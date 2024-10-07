// 预处理指令，用于防止头文件被多次包含
#ifndef UTILS_H_
#define UTILS_H_

// 包含另一个名为 inference.h 的头文件
#include "inference.h"

/*
（1）接受一个 cv::Mat 对象，一个 yolo::Detection对象的向量（检测结果框），以及一个包含类名的字符串向量。函数的目的是将检测到的对象绘制到图像帧上。
（2）从metadata中获取类名
*/
void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections, const std::vector<std::string> &class_names);
std::vector<std::string> GetClassNameFromMetadata(const std::string &metadata_path);

#endif // UTILS_H_
