#include "lc_vision_internal.hpp"

#include <gpu.h>
#include <net.h>

namespace lc_vision {

namespace {

class NCNNDetectorBackend final : public DetectorBackend {
public:
    DetectorFramework framework() const override {
        return DetectorFramework::NCNN;
    }

    std::string defaultModelDir() const override {
        return "models/yolo26n_ncnn_model";
    }

    std::string modelFileExtension() const override {
        return ".param";
    }

    DetectorPreparationResult prepare(
        const DetectorConfig &config, const fs::path &model_dir, const ParsedMetadata &metadata,
        const rclcpp::Logger &logger) override {
        reset();

        if (!backendValueIsKnown(config.backend)) {
            RCLCPP_WARN(
                logger, "Unknown detector.backend value '%s', defaulting to CPU fallback behavior.",
                config.backend.c_str());
        }

        fs::path model_param;
        for (const auto &entry : fs::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".param") {
                model_param = entry.path();
                break;
            }
        }
        if (model_param.empty()) {
            throw std::runtime_error("No .param file found in model directory: " + model_dir.string());
        }

        const fs::path model_bin = model_param.parent_path() / (model_param.stem().string() + ".bin");
        if (!fs::exists(model_bin)) {
            throw std::runtime_error("NCNN .bin file not found next to " + model_param.string());
        }

        detector_net_.opt.num_threads = std::max(1, config.cpu_num_threads);
        detector_net_.opt.use_vulkan_compute = false;

        DetectorPreparationResult result;
        result.active_backend = "ncnn:cpu";

#if NCNN_VULKAN
        if (!forcesCpuOnly(config.backend) && prefersVulkan(config.backend)) {
            if (ncnn::create_gpu_instance() != 0) {
                RCLCPP_WARN(logger, "Failed to create NCNN Vulkan instance, falling back to CPU.");
            } else {
                gpu_instance_created_ = true;
                result.detected_gpu_count = ncnn::get_gpu_count();
                if (result.detected_gpu_count <= 0) {
                    RCLCPP_WARN(logger, "No Vulkan-capable NCNN GPU found, falling back to CPU.");
                    reset();
                    detector_net_.opt.num_threads = std::max(1, config.cpu_num_threads);
                } else {
                    result.active_gpu_index =
                        std::clamp(config.vulkan_device_index, 0, result.detected_gpu_count - 1);
                    detector_net_.opt.use_vulkan_compute = true;
                    detector_net_.set_vulkan_device(result.active_gpu_index);
                    result.active_backend = "ncnn:vulkan";
                }
            }
        }
#else
        if (prefersVulkan(config.backend)) {
            RCLCPP_WARN(
                logger, "This NCNN build does not include Vulkan support, falling back to CPU backend.");
        }
#endif

        if (detector_net_.load_param(model_param.c_str()) != 0) {
            throw std::runtime_error("Failed to load NCNN param file: " + model_param.string());
        }
        if (detector_net_.load_model(model_bin.c_str()) != 0) {
            throw std::runtime_error("Failed to load NCNN model weights: " + model_bin.string());
        }

        const auto input_indexes = detector_net_.input_indexes();
        const auto output_indexes = detector_net_.output_indexes();
        if (input_indexes.empty()) {
            throw std::runtime_error("NCNN model exposes no input blobs");
        }
        if (output_indexes.empty()) {
            throw std::runtime_error("NCNN model exposes no output blobs");
        }

        detector_input_index_ = input_indexes.front();
        detector_output_indexes_.assign(output_indexes.begin(), output_indexes.end());

#if NCNN_STRING
        const auto input_names = detector_net_.input_names();
        const auto output_names = detector_net_.output_names();
        detector_input_name_ = input_names.empty() ? std::string() : std::string(input_names.front());
        detector_output_names_.clear();
        detector_output_names_.reserve(output_names.size());
        for (const char *name : output_names) {
            detector_output_names_.emplace_back(name);
        }
        std::sort(detector_output_names_.begin(), detector_output_names_.end());
#else
        detector_input_name_.clear();
        detector_output_names_.clear();
#endif

        if (metadata.input_size > 0) {
            result.input_size = metadata.input_size;
        }

        if (config.log_backend_info) {
            RCLCPP_INFO(
                logger, "NCNN detector configured: backend=%s gpu_count=%d gpu_index=%d cpu_threads=%d",
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

        ncnn::Mat input = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_RGB, input_size, input_size);
        const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        input.substract_mean_normalize(nullptr, norm_vals);

        ncnn::Extractor extractor = detector_net_.create_extractor();
        extractor.set_light_mode(true);

        int input_result = -1;
#if NCNN_STRING
        if (!detector_input_name_.empty()) {
            input_result = extractor.input(detector_input_name_.c_str(), input);
        } else
#endif
        {
            input_result = extractor.input(detector_input_index_, input);
        }
        if (input_result != 0) {
            throw std::runtime_error("Failed to feed detector input into NCNN extractor");
        }

        if (detector_output_names_.empty() && detector_output_indexes_.empty()) {
            throw std::runtime_error("NCNN model has no output blobs");
        }

        ncnn::Mat output;
        int       extract_result = -1;
#if NCNN_STRING
        if (!detector_output_names_.empty()) {
            extract_result = extractor.extract(detector_output_names_.front().c_str(), output);
        } else
#endif
        {
            extract_result = extractor.extract(detector_output_indexes_.front(), output);
        }
        if (extract_result != 0) {
            throw std::runtime_error("Failed to extract NCNN detector output");
        }

        const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
        const int attr_count = class_count + 4;

        enum class Layout {
            AttrByBoxes,
            BoxesByAttr,
            ChannelsByBoxes,
        };

        Layout layout;
        int    num_boxes = 0;

        if (output.dims == 2) {
            if (output.h == attr_count) {
                layout = Layout::AttrByBoxes;
                num_boxes = output.w;
            } else if (output.w == attr_count) {
                layout = Layout::BoxesByAttr;
                num_boxes = output.h;
            } else {
                throw std::runtime_error(
                    "Unsupported NCNN detection tensor shape " + std::to_string(output.w) + "x" +
                    std::to_string(output.h));
            }
        } else if (output.dims == 3) {
            if (output.c == 1 && output.h == attr_count) {
                layout = Layout::AttrByBoxes;
                num_boxes = output.w;
            } else if (output.c == 1 && output.w == attr_count) {
                layout = Layout::BoxesByAttr;
                num_boxes = output.h;
            } else if (output.c == attr_count) {
                layout = Layout::ChannelsByBoxes;
                num_boxes = output.w * output.h;
            } else {
                throw std::runtime_error(
                    "Unsupported NCNN detection tensor dims c=" + std::to_string(output.c) + " h=" +
                    std::to_string(output.h) + " w=" + std::to_string(output.w));
            }
        } else {
            throw std::runtime_error("Unsupported NCNN output dims: " + std::to_string(output.dims));
        }

        const auto value_at = [&](const int attr_index, const int box_index) -> float {
            switch (layout) {
            case Layout::AttrByBoxes: {
                const float *data = static_cast<const float *>(output.data);
                return data[attr_index * num_boxes + box_index];
            }
            case Layout::BoxesByAttr: {
                const float *data = static_cast<const float *>(output.data);
                return data[box_index * attr_count + attr_index];
            }
            case Layout::ChannelsByBoxes: {
                const float *channel = output.channel(attr_index);
                return channel[box_index];
            }
            }
            return 0.0f;
        };

        return decodeDetections(
            width, height, pad_left, pad_top, scale, config.class_names, config.target_class_ids,
            config.score_threshold, config.nms_threshold, num_boxes, value_at);
    }

    void reset() override {
        detector_net_.clear();
#if NCNN_VULKAN
        if (gpu_instance_created_) {
            ncnn::destroy_gpu_instance();
            gpu_instance_created_ = false;
        }
#endif
        detector_input_index_ = 0;
        detector_input_name_.clear();
        detector_output_indexes_.clear();
        detector_output_names_.clear();
    }

private:
    static bool prefersVulkan(const std::string &backend) {
        const auto normalized = toLower(backend);
        return normalized == "auto" || normalized == "vulkan";
    }

    static bool forcesCpuOnly(const std::string &backend) {
        return toLower(backend) == "cpu";
    }

    static bool backendValueIsKnown(const std::string &backend) {
        const auto normalized = toLower(backend);
        return normalized == "auto" || normalized == "vulkan" || normalized == "cpu";
    }

    ncnn::Net                detector_net_;
    bool                     gpu_instance_created_ = false;
    int                      detector_input_index_ = 0;
    std::string              detector_input_name_;
    std::vector<int>         detector_output_indexes_;
    std::vector<std::string> detector_output_names_;
};

} // namespace

std::unique_ptr<DetectorBackend> createNCNNBackend() {
    return std::make_unique<NCNNDetectorBackend>();
}

} // namespace lc_vision
