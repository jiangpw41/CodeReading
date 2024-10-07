#include "inference.h"			// 包含自定义的头文件inference.h，该文件定义了Inference类。

#include <memory>				// 包含C++标准库中的<memory>头文件，用于使用智能指针。

namespace yolo {				// 声明为yolo的命名空间（在.h中定义），下面逐个实现其中的函数
// 实现公有部分两个参数输入的构造函数
Inference::Inference(const std::string &model_path, const float &model_confidence_threshold) {
	model_input_shape_ = cv::Size(640, 640); 					// 设置模型输入尺寸为640x640像素，这是为动态形状模型设置的默认尺寸。
	model_confidence_threshold_ = model_confidence_threshold;	// 将成员变量model_confidence_threshold_设置为传入的置信度阈值。
	InitialModel(model_path);									// 调用InitialModel函数来初始化模型。
}

// 实现公有部分三个参数输入的构造函数：它允许用户指定模型输入尺寸。
Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold) {
	model_input_shape_ = model_input_shape;						// 将成员变量model_input_shape_设置为传入的尺寸。其他一样
	model_confidence_threshold_ = model_confidence_threshold;

	InitialModel(model_path);
}

// 定义InitialModel函数，用于加载和初始化模型。
// 输入字符串模型路径
void Inference::InitialModel(const std::string &model_path) {
	ov::Core core;														// 创建一个OpenVINO运行时核心对象。
	std::shared_ptr<ov::Model> model = core.read_model(model_path);		// 用core读取模型文件并创建一个模型对象。

	if (model->is_dynamic()) {											// 如果模型是动态形状的，则调整其形状以匹配输入尺寸。
		model->reshape({1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)});
	}

	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);	// 创建一个OpenVINO中的预处理和后处理处理器，ppp

	// 设置输入张量的元素格式：类型为无符号8位整数（通常是图像数据），布局为NHWC（批大小、高度、宽度、通道数），颜色格式为BGR。
  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // 在预处理阶段，将元素类型转换为浮点数，颜色格式转换为RGB，并进行缩放。
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
	// 设置模型输入的布局为NCHW。
	ppp.input().model().set_layout("NCHW");
	// 设置输出张量的元素类型为浮点数。
  ppp.output().tensor().set_element_type(ov::element::f32);

	// 构建处理后的模型。
  model = ppp.build();
  	// 编译模型，使用自动选择的设备。
	compiled_model_ = core.compile_model(model, "AUTO");
	// 创建一个推理请求对象
	inference_request_ = compiled_model_.create_infer_request();
	// 定义宽度和高度变量。
	short width, height;
	// 获取模型的输入并获取其形状：
  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  const ov::Shape input_shape = inputs[0].get_shape();
	// 将shape分派给width和height，更新模型输入尺寸。
	height = input_shape[1];
	width = input_shape[2];
	model_input_shape_ = cv::Size2f(width, height);
	// 获取模型的输出并获取其形状。
  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  const ov::Shape output_shape = outputs[0].get_shape();
	// 更新模型输出尺寸。
	height = output_shape[1];
	width = output_shape[2];
	model_output_shape_ = cv::Size(width, height);
}

// 定义RunInference函数，用于执行推理。
std::vector<Detection> Inference::RunInference(const cv::Mat &frame) {
	Preprocessing(frame);			// 执行预处理。
	inference_request_.infer();		// 执行推理。
	PostProcessing();				// 执行后处理。

	return detections_;				// 返回检测结果。
}

// 定义Preprocessing函数，用于执行预处理。
void Inference::Preprocessing(const cv::Mat &frame) {
	// 将输入帧调整到模型的输入尺寸。
	cv::Mat resized_frame;
	cv::resize(frame, resized_frame, model_input_shape_, 0, 0, cv::INTER_AREA);
	// 计算缩放因子。
	scale_factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
	scale_factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);
	// 获取调整大小后的帧数据。
	float *input_data = (float *)resized_frame.data;
	// 创建一个输入张量。
	const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data);
	// 设置推理请求的输入张量。
	inference_request_.set_input_tensor(input_tensor);
}

// 定义PostProcessing函数，用于执行后处理。
void Inference::PostProcessing() {
	// 获取输出张量的数据。
	const float *detections = inference_request_.get_output_tensor().data<const float>();
	// 清除之前的检测结果。
	detections_.clear();

	/*
	* 0  1  2  3      4          5
	* x, y, w. h, confidence, class_id
	*/
	
	for (unsigned int i = 0; i < model_output_shape_.height; ++i) {
		// 遍历输出张量
		const unsigned int index = i * model_output_shape_.width;
		// 获取置信度。
		const float &confidence = detections[index + 4];
		// 如果置信度高于阈值，则处理检测。
		if (confidence > model_confidence_threshold_) {
			// 获取检测框的坐标和尺寸。
			const float &x = detections[index + 0];
			const float &y = detections[index + 1];
			const float &w = detections[index + 2];
			const float &h = detections[index + 3];
			// 创建一个检测结果对象。
			Detection result;
			// 设置类别ID、置信度和检测框。
			result.class_id = static_cast<const short>(detections[index + 5]);
			result.confidence = confidence;
			result.box = GetBoundingBox(cv::Rect(x, y, w, h));
			// 将检测结果添加到结果列表。
			detections_.push_back(result);
		}
	}
}

// 定义GetBoundingBox函数，用于获取实际的检测框。
cv::Rect Inference::GetBoundingBox(const cv::Rect &src) const {
	// 创建一个检测框。
	cv::Rect box = src;
	// 根据缩放因子调整检测框的宽度和高度。
	box.width = (box.width - box.x) * scale_factor_.x;
	box.height = (box.height - box.y) * scale_factor_.y;
	// 根据缩放因子调整检测框的x和y坐标。
	box.x *= scale_factor_.x;
	box.y *= scale_factor_.y;
	// 返回调整后的检测框。
	return box;
}
} // namespace yolo
