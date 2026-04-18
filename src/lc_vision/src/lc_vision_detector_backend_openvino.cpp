#include "lc_vision_internal.hpp"

#include <openvino/openvino.hpp>

namespace lc_vision {

namespace {

class OpenVINODetectorBackend final : public DetectorBackend {
public:
    DetectorFramework framework() const override {
        return DetectorFramework::OpenVINO;
    }

    std::string defaultModelDir() const override {
        return "models/yolo26n_openvino_model";
    }

    std::string modelFileExtension() const override {
        return ".xml";
    }

    DetectorPreparationResult prepare(
        const DetectorConfig &config, const fs::path &model_dir, const ParsedMetadata &metadata,
        const rclcpp::Logger &logger) override {
        reset();

        if (!backendValueIsKnown(config.backend)) {
            RCLCPP_WARN(
                logger, "Unknown detector.backend value '%s', defaulting to AUTO for OpenVINO.",
                config.backend.c_str());
        }

        fs::path model_xml;
        for (const auto &entry : fs::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                model_xml = entry.path();
                break;
            }
        }
        if (model_xml.empty()) {
            throw std::runtime_error("No .xml file found in model directory: " + model_dir.string());
        }

        const fs::path model_bin = model_xml.parent_path() / (model_xml.stem().string() + ".bin");
        if (!fs::exists(model_bin)) {
            throw std::runtime_error("OpenVINO .bin file not found next to " + model_xml.string());
        }

        detector_device_name_ = requestedDetectorDevice(config.backend);
        detector_model_ = detector_core_.read_model(model_xml.string());
        detector_compiled_model_ = detector_core_.compile_model(detector_model_, detector_device_name_);
        detector_infer_request_ = detector_compiled_model_.create_infer_request();

        const auto inputs = detector_compiled_model_.inputs();
        const auto outputs = detector_compiled_model_.outputs();
        if (inputs.empty()) {
            throw std::runtime_error("OpenVINO model exposes no input tensors");
        }
        if (outputs.empty()) {
            throw std::runtime_error("OpenVINO model exposes no output tensors");
        }

        DetectorPreparationResult result;
        result.active_backend = "openvino:" + toLower(detector_device_name_);
        const auto input_shape = inputs.front().get_shape();
        if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
            result.input_size = static_cast<int>(input_shape[2]);
        } else if (metadata.input_size > 0) {
            result.input_size = metadata.input_size;
        }

        if (config.log_backend_info) {
            RCLCPP_INFO(
                logger, "OpenVINO detector configured: backend=%s gpu_count=%d gpu_index=%d cpu_threads=%d",
                result.active_backend.c_str(), result.detected_gpu_count, result.active_gpu_index,
                config.cpu_num_threads);
        }

        return result;
    }

    std::vector<Detection> infer(
        const std::vector<uint8_t> &rgb_data, const int width, const int height, const int input_size,
        const DetectorConfig &config) override {
        if (rgb_data.empty() || width <= 0 || height <= 0) {
            return {};
        }

        cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(rgb_data.data()));

        const float scale = std::min(
            static_cast<float>(input_size) / static_cast<float>(width),
            static_cast<float>(input_size) / static_cast<float>(height));
        const int resized_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(width) * scale)));
        const int resized_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(height) * scale)));
        const int pad_w = std::max(0, input_size - resized_w);
        const int pad_h = std::max(0, input_size - resized_h);
        const int pad_left = pad_w / 2;
        const int pad_top = pad_h / 2;

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat padded(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_left, pad_top, resized_w, resized_h)));

        ov::Tensor input_tensor(
            ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(input_size), static_cast<size_t>(input_size)});
        float       *input_data = input_tensor.data<float>();
        const size_t channel_stride = static_cast<size_t>(input_size) * static_cast<size_t>(input_size);

        for (int y = 0; y < input_size; ++y) {
            const auto *row = padded.ptr<uint8_t>(y);
            for (int x = 0; x < input_size; ++x) {
                const size_t index = static_cast<size_t>(y) * static_cast<size_t>(input_size) + static_cast<size_t>(x);
                const auto  *pixel = row + (static_cast<size_t>(x) * 3U);
                input_data[index] = static_cast<float>(pixel[0]) / 255.0f;
                input_data[channel_stride + index] = static_cast<float>(pixel[1]) / 255.0f;
                input_data[channel_stride * 2U + index] = static_cast<float>(pixel[2]) / 255.0f;
            }
        }

        detector_infer_request_.set_input_tensor(input_tensor);
        detector_infer_request_.infer();

        const ov::Tensor output = detector_infer_request_.get_output_tensor(0);
        auto             shape = output.get_shape();
        while (shape.size() > 2 && !shape.empty() && shape.front() == 1U) {
            shape.erase(shape.begin());
        }

        const auto         type = output.get_element_type();
        const float       *data_f32 = type == ov::element::f32 ? output.data<const float>() : nullptr;
        const ov::float16 *data_f16 = type == ov::element::f16 ? output.data<const ov::float16>() : nullptr;

        if (data_f32 == nullptr && data_f16 == nullptr) {
            throw std::runtime_error("Unsupported OpenVINO output element type: " + type.to_string());
        }

        if (shape.size() != 2 || static_cast<int>(shape[1]) != 6) {
            throw std::runtime_error(
                "Unsupported OpenVINO detection tensor shape for Ultralytics export: rank=" +
                std::to_string(shape.size()) +
                (shape.empty() ? std::string() : " first_dim=" + std::to_string(shape[0])));
        }

        const auto read_at = [&](const int flat_index) -> float {
            if (data_f32 != nullptr) {
                return data_f32[flat_index];
            }
            return static_cast<float>(data_f16[flat_index]);
        };

        const int   num_boxes = static_cast<int>(shape[0]);
        const float inv_scale = 1.0f / std::max(scale, 1.0e-6f);
        std::vector<Detection> detections;
        detections.reserve(static_cast<size_t>(num_boxes));

        for (int row = 0; row < num_boxes; ++row) {
            const int   base = row * 6;
            const float score = read_at(base + 4);
            if (score < config.score_threshold) {
                continue;
            }

            const int class_id = static_cast<int>(std::lround(read_at(base + 5)));
            if (!config.target_class_ids.empty() && !config.target_class_ids.contains(class_id)) {
                continue;
            }

            const float raw_x1 = read_at(base + 0);
            const float raw_y1 = read_at(base + 1);
            const float raw_x2 = read_at(base + 2);
            const float raw_y2 = read_at(base + 3);

            const int x1 = std::clamp(
                static_cast<int>(std::floor((raw_x1 - static_cast<float>(pad_left)) * inv_scale)), 0, width - 1);
            const int y1 = std::clamp(
                static_cast<int>(std::floor((raw_y1 - static_cast<float>(pad_top)) * inv_scale)), 0, height - 1);
            const int x2 = std::clamp(
                static_cast<int>(std::ceil((raw_x2 - static_cast<float>(pad_left)) * inv_scale)), x1 + 1, width);
            const int y2 = std::clamp(
                static_cast<int>(std::ceil((raw_y2 - static_cast<float>(pad_top)) * inv_scale)), y1 + 1, height);

            Detection detection;
            detection.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            detection.score = score;
            detection.class_id = class_id;
            detection.label = class_id >= 0 && class_id < static_cast<int>(config.class_names.size())
                                  ? config.class_names[static_cast<size_t>(class_id)]
                                  : std::to_string(class_id);
            detections.push_back(std::move(detection));
        }

        applyNms(detections, config.nms_threshold);
        return detections;
    }

    void reset() override {
        detector_infer_request_ = {};
        detector_compiled_model_ = {};
        detector_model_.reset();
        detector_device_name_ = "AUTO";
    }

private:
    static bool backendValueIsKnown(const std::string &backend) {
        const auto normalized = toLower(backend);
        return normalized == "auto" || normalized == "cpu" || normalized == "gpu";
    }

    static std::string requestedDetectorDevice(const std::string &backend) {
        const auto normalized = toLower(backend);
        if (normalized == "cpu") {
            return "CPU";
        }
        if (normalized == "gpu") {
            return "GPU";
        }
        return "AUTO";
    }

    ov::Core                   detector_core_;
    std::shared_ptr<ov::Model> detector_model_;
    ov::CompiledModel          detector_compiled_model_;
    ov::InferRequest           detector_infer_request_;
    std::string                detector_device_name_ = "AUTO";
};

} // namespace

std::unique_ptr<DetectorBackend> createOpenVINOBackend() {
    return std::make_unique<OpenVINODetectorBackend>();
}

} // namespace lc_vision
