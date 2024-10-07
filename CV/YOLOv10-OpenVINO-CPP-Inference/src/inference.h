/*
定义inference类，但不实现，实现在.cc文件中。这里核心是
（1）include相关库
（2）定义yolo命名空间，并在这个空间内定义类和结构体
*/
/*
在C++中，宏#ifndef YOLO_INFERENCE_H_和#define YOLO_INFERENCE_H_以及#endif通常被用来防止头文件被多次包含，
这是一种常见的编程模式，称为“包含保护”（Include Guard）。
如果头文件inference.h已经被包含过一次，那么预处理器会跳过该头文件的整个内容，从#ifndef到#endif之间的所有代码。
即，在新的文件中，如果YOLO_INFERENCE_H_没有定义（说明.h头文件还没有被include），那么定义它，并包含以下内容
*/
#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

// 包含C++标准库中的<string>和<vector>头文件，这些头文件分别提供了字符串和动态数组的支持。
#include <string>
#include <vector>

// 包含OpenCV的图像处理库和OpenVINO库的头文件，这些库用于图像处理和深度学习模型推理。
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

// 定义一个名为yolo的命名空间，用于封装相关的类和结构体。
namespace yolo {
// 定义一个Detection结构体，用于存储检测结果，并直接服务于显示（在显示器上只显示框、类别、置信度）
struct Detection {
	short class_id;			// 16位int，类别ID一般不超过百
	float confidence;		// 置信度，为这个类的可能
	cv::Rect box;			// 边界框，为cv命名空间中的Rect(rectanular矩形)
};

class Inference {
// 定义Inference类的公共部分，三个可重载的构造函数，根据不同的传参需求进行构造
 public:
	Inference() {}																											// 提供一个默认构造函数。
	Inference(const std::string &model_path, const float &model_confidence_threshold);										// 提供一个构造函数，接收模型路径和置信度阈值作为参数
	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold);	// 提供一个构造函数，接收模型路径、输入尺寸和置信度阈值作为参数。

	std::vector<Detection> RunInference(const cv::Mat &frame);			// 定义一个公共成员函数RunInference，用于执行推理并返回检测结果。

// 定义Inference类的私有部分
 private:
	void InitialModel(const std::string &model_path);		// 定义一个私有成员函数InitialModel，用于初始化模型。
	void Preprocessing(const cv::Mat &frame);				// 定义一个私有成员函数Preprocessing，用于执行预处理。
	void PostProcessing();									// 定义一个私有成员函数PostProcessing，用于执行后处理
	cv::Rect GetBoundingBox(const cv::Rect &src) const;		// 定义一个私有成员函数GetBoundingBox，用于获取实际的检测框。
	
	// 定义一些私有成员变量（cv命名空间的数据类型），用于存储缩放因子、模型输入和输出尺寸。
	cv::Point2f scale_factor_;
	cv::Size2f model_input_shape_;
	cv::Size model_output_shape_;

	// 定义一些私有成员变量（ov命名空间的数据类型），用于存储推理请求和编译后的模型。
	ov::InferRequest inference_request_;
	ov::CompiledModel compiled_model_;

	// 定义一个私有成员变量（std标准命名空间的数据类型），用于存储检测结果。
	std::vector<Detection> detections_;

	// 定义一个私有成员变量，用于存储模型的置信度阈值。
	float model_confidence_threshold_;
};
} // namespace yolo结束yolo命名空间。

#endif // YOLO_INFERENCE_H_结束预处理器指令，防止头文件被多次包含
