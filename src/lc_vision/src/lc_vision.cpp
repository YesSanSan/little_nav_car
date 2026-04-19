#include "lc_vision_internal.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace lc_vision {

LCVision::LCVision(const rclcpp::NodeOptions &options)
    : Node("lc_vision", options),
      impl_(std::make_unique<Impl>(*this)) {
    RCLCPP_INFO(get_logger(), "Start LCVision!");

    try {
        getParams();
        prepareModel();

        nav_to_pose_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
        impl_->tracking_runtime_enabled.store(impl_->tracking.enable);
        impl_->goal_runtime_enabled.store(impl_->goal.enable);
        impl_->parameter_callback_handle =
            this->add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &parameters) {
                return handleRuntimeParameters(parameters);
            });

        if (!impl_->rgb.enable && !impl_->depth.enable) {
            throw std::runtime_error("Both rgb.enable and depth.enable are false; nothing to publish");
        }

        astra::initialize();
        impl_->astra_initialized = true;
        impl_->stream_set = astra::StreamSet();
        impl_->reader = impl_->stream_set.create_reader();

        if (!impl_->reader.is_valid()) {
            throw std::runtime_error("Failed to create Astra stream reader");
        }

        impl_->reader.add_listener(*impl_->capture_listener);

        auto detection_qos = rclcpp::QoS(rclcpp::KeepLast(10));
        detection_qos.reliable();
        impl_->detections_depth_pub =
            create_publisher<lc_vision::msg::DetectionDepthArray>("~/detections_depth", detection_qos);

        if (impl_->depth.debug_marker.enable) {
            impl_->depth_debug_marker_pub = create_publisher<visualization_msgs::msg::MarkerArray>(
                impl_->depth.debug_marker.topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
        }

        if (impl_->goal.debug_marker.enable) {
            impl_->goal_debug_marker_pub = create_publisher<visualization_msgs::msg::MarkerArray>(
                impl_->goal.debug_marker.topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
        }

        if (impl_->goal.publish_debug_topics) {
            impl_->selected_target_camera_point_pub =
                create_publisher<geometry_msgs::msg::PointStamped>("~/selected_target_camera_point", detection_qos);
            impl_->selected_target_map_point_pub =
                create_publisher<geometry_msgs::msg::PointStamped>("~/selected_target_map_point", detection_qos);
            impl_->selected_goal_pose_pub =
                create_publisher<geometry_msgs::msg::PoseStamped>("~/selected_goal_pose", detection_qos);
        }

        if (!impl_->goal.costmap_topic.empty()) {
            impl_->costmap_sub = create_subscription<nav2_msgs::msg::Costmap>(
                impl_->goal.costmap_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
                [impl = impl_.get()](const nav2_msgs::msg::Costmap::SharedPtr msg) {
                    impl->storeCostmap(msg);
                });
        }

        if (impl_->rgb.enable) {
            auto color_stream = impl_->reader.stream<astra::ColorStream>();
            if (!color_stream.is_available()) {
                throw std::runtime_error("Astra color stream is not available");
            }

            color_stream.enable_mirroring(false);
            const auto rgb_mode = selectBestMode(
                color_stream, impl_->rgb.width, impl_->rgb.height, impl_->rgb.capture_fps, ASTRA_PIXEL_FORMAT_RGB888);
            color_stream.set_mode(rgb_mode);
            color_stream.start();

            const auto active_mode = color_stream.mode();
            RCLCPP_INFO(
                get_logger(), "Configured RGB stream: %ux%u @ %u fps (mirroring disabled)", active_mode.width(),
                active_mode.height(), active_mode.fps());

            auto video_qos = rclcpp::QoS(rclcpp::KeepLast(5));
            video_qos.reliable();
            impl_->rgb_video_pub = create_publisher<foxglove_msgs::msg::CompressedVideo>("~/rgb/video", video_qos);
        }

        if (impl_->depth.enable) {
            impl_->depth_stream = impl_->reader.stream<astra::DepthStream>();
            if (!impl_->depth_stream->is_available()) {
                throw std::runtime_error("Astra depth stream is not available");
            }

            impl_->depth_stream->enable_mirroring(false);
            const auto depth_mode = selectBestMode(
                *impl_->depth_stream, impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
                ASTRA_PIXEL_FORMAT_DEPTH_MM);
            impl_->depth_stream->set_mode(depth_mode);
            if (impl_->depth.enable_registration && impl_->rgb.enable) {
                try {
                    impl_->depth_stream->enable_registration(true);
                    RCLCPP_INFO(get_logger(), "Enabled Astra depth-to-color registration");
                } catch (const std::exception &ex) {
                    RCLCPP_WARN(get_logger(), "Failed to enable Astra depth registration: %s", ex.what());
                }
            }
            impl_->depth_stream->start();

            const auto active_mode = impl_->depth_stream->mode();
            RCLCPP_INFO(
                get_logger(), "Configured depth stream: %ux%u @ %u fps (mirroring disabled)", active_mode.width(),
                active_mode.height(), active_mode.fps());

            auto video_qos = rclcpp::QoS(rclcpp::KeepLast(5));
            video_qos.reliable();
            impl_->depth_video_pub =
                create_publisher<foxglove_msgs::msg::CompressedVideo>("~/depth/video", video_qos);

            if (impl_->depth.debug_video.histogram.enable) {
                impl_->depth_histogram_video_pub = create_publisher<foxglove_msgs::msg::CompressedVideo>(
                    impl_->depth.debug_video.histogram.topic, video_qos);
            }
            if (impl_->depth.debug_video.peak_mask.enable) {
                impl_->depth_peak_mask_video_pub = create_publisher<foxglove_msgs::msg::CompressedVideo>(
                    impl_->depth.debug_video.peak_mask.topic, video_qos);
            }
        }

        impl_->running.store(true);

        impl_->capture_thread = std::thread([this]() {
            while (rclcpp::ok() && impl_->running.load()) {
                try {
                    astra_update();
                    std::this_thread::sleep_for(1ms);
                } catch (const std::exception &ex) {
                    RCLCPP_ERROR(get_logger(), "Astra capture loop failed: %s", ex.what());
                    impl_->running.store(false);
                }
            }
        });

        if (impl_->detector_runtime_enabled.load()) {
            impl_->detection_thread = std::thread([this]() {
                impl_->detectionLoop();
            });
        }

        impl_->publish_thread = std::thread([this]() {
            auto       next_rgb_publish_time = std::chrono::steady_clock::now();
            auto       next_depth_publish_time = std::chrono::steady_clock::now();
            const auto rgb_period =
                std::chrono::milliseconds(std::max(1, 1000 / std::max(1, impl_->rgb.publish_fps)));
            const auto depth_period =
                std::chrono::milliseconds(std::max(1, 1000 / std::max(1, impl_->depth.publish_fps)));

            while (rclcpp::ok() && impl_->running.load()) {
                const auto now = std::chrono::steady_clock::now();

                if (impl_->rgb.enable && now >= next_rgb_publish_time) {
                    next_rgb_publish_time = now + rgb_period;
                    try {
                        const auto msg = impl_->makeRgbMessage();
                        if (msg.has_value()) {
                            impl_->rgb_video_pub->publish(*msg);
                        }
                    } catch (const std::exception &ex) {
                        RCLCPP_ERROR(get_logger(), "RGB video encoding failed: %s", ex.what());
                    }
                }

                if (impl_->depth.enable && now >= next_depth_publish_time) {
                    next_depth_publish_time = now + depth_period;
                    try {
                        const auto msg = impl_->makeDepthMessage();
                        if (msg.has_value()) {
                            impl_->depth_video_pub->publish(*msg);
                        }
                        if (impl_->depth_histogram_video_pub != nullptr) {
                            const auto histogram_msg = impl_->makeDepthHistogramMessage();
                            if (histogram_msg.has_value()) {
                                impl_->depth_histogram_video_pub->publish(*histogram_msg);
                            }
                        }
                        if (impl_->depth_peak_mask_video_pub != nullptr) {
                            const auto peak_mask_msg = impl_->makeDepthPeakMaskMessage();
                            if (peak_mask_msg.has_value()) {
                                impl_->depth_peak_mask_video_pub->publish(*peak_mask_msg);
                            }
                        }
                    } catch (const std::exception &ex) {
                        RCLCPP_ERROR(get_logger(), "Depth video encoding failed: %s", ex.what());
                    }
                }

                std::this_thread::sleep_for(2ms);
            }
        });

        RCLCPP_INFO(get_logger(), "Astra SDK root: %s", impl_->sdk_root.c_str());
    } catch (...) {
        impl_->shutdown();
        throw;
    }
}

LCVision::~LCVision() {
    impl_->shutdown();
}

void LCVision::getParams() {
    impl_->sdk_root = expandHomeDirectory(this->declare_parameter<std::string>("sdk_root", defaultAstraSdkRoot()));

    impl_->rgb.enable = this->declare_parameter<bool>("rgb.enable", true);
    impl_->rgb.width = this->declare_parameter<int>("rgb.width", 640);
    impl_->rgb.height = this->declare_parameter<int>("rgb.height", 480);
    impl_->rgb.capture_fps = this->declare_parameter<int>("rgb.capture_fps", 15);
    impl_->rgb.publish_fps = this->declare_parameter<int>("rgb.publish_fps", 10);
    impl_->rgb.bitrate_kbps = this->declare_parameter<int>("rgb.bitrate_kbps", 1200);
    impl_->rgb.gop_size = this->declare_parameter<int>("rgb.gop_size", 30);
    impl_->rgb.frame_id = this->declare_parameter<std::string>("rgb.frame_id", "cam_optical_frame");

    impl_->depth.enable = this->declare_parameter<bool>("depth.enable", true);
    impl_->depth.width = this->declare_parameter<int>("depth.width", 640);
    impl_->depth.height = this->declare_parameter<int>("depth.height", 480);
    impl_->depth.capture_fps = this->declare_parameter<int>("depth.capture_fps", 15);
    impl_->depth.publish_fps = this->declare_parameter<int>("depth.publish_fps", 8);
    impl_->depth.bitrate_kbps = this->declare_parameter<int>("depth.bitrate_kbps", 500);
    impl_->depth.gop_size = this->declare_parameter<int>("depth.gop_size", 24);
    impl_->depth.frame_id = this->declare_parameter<std::string>("depth.frame_id", "cam_optical_frame");
    impl_->depth.visualization_min_mm = this->declare_parameter<int>("depth.visualization_min_mm", 300);
    impl_->depth.visualization_max_mm = this->declare_parameter<int>("depth.visualization_max_mm", 5000);
    impl_->depth.roi_min_valid_mm = this->declare_parameter<int>("depth.roi_min_valid_mm", 300);
    impl_->depth.roi_max_valid_mm = this->declare_parameter<int>("depth.roi_max_valid_mm", 5000);
    impl_->depth.invalid_as_black = this->declare_parameter<bool>("depth.invalid_as_black", true);
    impl_->depth.enable_registration = this->declare_parameter<bool>("depth.enable_registration", true);
    impl_->depth.estimation.min_valid_pixels =
        this->declare_parameter<int>("depth.estimation.min_valid_pixels", 50);
    impl_->depth.estimation.histogram_bin_size_mm =
        this->declare_parameter<int>("depth.estimation.histogram_bin_size_mm", 50);
    impl_->depth.estimation.peak_min_ratio =
        static_cast<float>(this->declare_parameter<double>("depth.estimation.peak_min_ratio", 0.25));
    impl_->depth.estimation.peak_min_count = this->declare_parameter<int>("depth.estimation.peak_min_count", 5);
    impl_->depth.estimation.min_peak_pixels = this->declare_parameter<int>("depth.estimation.min_peak_pixels", 20);
    impl_->depth.estimation.trim_ratio =
        static_cast<float>(this->declare_parameter<double>("depth.estimation.trim_ratio", 0.1));
    impl_->depth.estimation.annotate_depth_on_rgb =
        this->declare_parameter<bool>("depth.estimation.annotate_depth_on_rgb", true);
    impl_->depth.estimation.annotate_depth_on_depth =
        this->declare_parameter<bool>("depth.estimation.annotate_depth_on_depth", true);
    impl_->depth.debug_marker.enable = this->declare_parameter<bool>("depth.debug_marker.enable", false);
    impl_->depth.debug_marker.topic =
        this->declare_parameter<std::string>("depth.debug_marker.topic", "~/depth_debug_marker");
    impl_->depth.debug_marker.frame_id =
        this->declare_parameter<std::string>("depth.debug_marker.frame_id", "base_link");
    impl_->depth.debug_marker.z =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.z", 0.05));
    impl_->depth.debug_marker.line_width =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.line_width", 0.02));
    impl_->depth.debug_marker.alpha =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.alpha", 0.9));
    impl_->depth.debug_marker.use_detection_label_ns =
        this->declare_parameter<bool>("depth.debug_marker.use_detection_label_ns", true);
    impl_->depth.debug_video.histogram.enable =
        this->declare_parameter<bool>("depth.debug_video.histogram.enable", false);
    impl_->depth.debug_video.histogram.topic =
        this->declare_parameter<std::string>("depth.debug_video.histogram.topic", "~/depth_histogram/video");
    impl_->depth.debug_video.peak_mask.enable =
        this->declare_parameter<bool>("depth.debug_video.peak_mask.enable", false);
    impl_->depth.debug_video.peak_mask.topic =
        this->declare_parameter<std::string>("depth.debug_video.peak_mask.topic", "~/depth_peak_mask/video");
    impl_->image.rotation_degrees =
        normalizeRotationDegrees(this->declare_parameter<int>("image.rotation_degrees", 0));
    rotationDegreesToQuarterTurns(impl_->image.rotation_degrees);

    impl_->detector.enable = this->declare_parameter<bool>("detector.enable", true);
    impl_->detector.framework = this->declare_parameter<std::string>("detector.framework", "auto");
    impl_->detector.backend = this->declare_parameter<std::string>("detector.backend", "auto");
    impl_->detector.model_dir = this->declare_parameter<std::string>("detector.model_dir", std::string());
    impl_->detector.input_size = this->declare_parameter<int>("detector.input_size", 640);
    impl_->detector.score_threshold =
        static_cast<float>(this->declare_parameter<double>("detector.score_threshold", 0.25));
    impl_->detector.nms_threshold =
        static_cast<float>(this->declare_parameter<double>("detector.nms_threshold", 0.45));
    impl_->detector.fps = this->declare_parameter<int>("detector.fps", 5);
    impl_->detector.vulkan_device_index = this->declare_parameter<int>("detector.vulkan_device_index", 0);
    impl_->detector.cpu_num_threads =
        this->declare_parameter<int>("detector.cpu_num_threads", defaultDetectorThreadCount());
    impl_->detector.log_backend_info = this->declare_parameter<bool>("detector.log_backend_info", true);
    impl_->detector.target_classes = this->declare_parameter<std::vector<std::string>>(
        "detector.target_classes", std::vector<std::string>{"person"});
    impl_->detector.class_names = defaultCocoClassNames();

    impl_->tracking.enable = this->declare_parameter<bool>("tracking.enable", true);
    impl_->tracking.initial_center_gate_ratio =
        static_cast<float>(this->declare_parameter<double>("tracking.initial_center_gate_ratio", 0.2));
    impl_->tracking.max_lost_frames = this->declare_parameter<int>("tracking.max_lost_frames", 10);
    impl_->tracking.min_iou_for_match =
        static_cast<float>(this->declare_parameter<double>("tracking.min_iou_for_match", 0.05));
    impl_->tracking.max_center_distance_px =
        static_cast<float>(this->declare_parameter<double>("tracking.max_center_distance_px", 120.0));
    impl_->tracking.max_depth_delta_mm =
        static_cast<float>(this->declare_parameter<double>("tracking.max_depth_delta_mm", 800.0));

    impl_->goal.enable = this->declare_parameter<bool>("goal.enable", true);
    impl_->goal.global_frame_id = this->declare_parameter<std::string>("goal.global_frame_id", "map");
    impl_->goal.robot_frame_id = this->declare_parameter<std::string>("goal.robot_frame_id", "base_link");
    impl_->goal.costmap_topic =
        this->declare_parameter<std::string>("goal.costmap_topic", "/global_costmap/costmap_raw");
    impl_->goal.standoff_distance_m =
        static_cast<float>(this->declare_parameter<double>("goal.standoff_distance_m", 0.5));
    impl_->goal.max_target_distance_m =
        static_cast<float>(this->declare_parameter<double>("goal.max_target_distance_m", 1.0));
    impl_->goal.min_stable_frames = this->declare_parameter<int>("goal.min_stable_frames", 3);
    impl_->goal.clearance_radius_m =
        static_cast<float>(this->declare_parameter<double>("goal.clearance_radius_m", 0.25));
    impl_->goal.max_cell_cost = this->declare_parameter<int>("goal.max_cell_cost", 80);
    impl_->goal.longitudinal_search_step_m =
        static_cast<float>(this->declare_parameter<double>("goal.longitudinal_search_step_m", 0.05));
    impl_->goal.max_backoff_distance_m =
        static_cast<float>(this->declare_parameter<double>("goal.max_backoff_distance_m", 1.5));
    impl_->goal.lateral_search_step_m =
        static_cast<float>(this->declare_parameter<double>("goal.lateral_search_step_m", 0.1));
    impl_->goal.max_lateral_offset_m =
        static_cast<float>(this->declare_parameter<double>("goal.max_lateral_offset_m", 0.4));
    impl_->goal.publish_debug_topics = this->declare_parameter<bool>("goal.publish_debug_topics", true);
    impl_->goal.debug_marker.enable = this->declare_parameter<bool>("goal.debug_marker.enable", false);
    impl_->goal.debug_marker.topic =
        this->declare_parameter<std::string>("goal.debug_marker.topic", "~/target_map_debug_marker");
    impl_->goal.debug_marker.frame_id =
        this->declare_parameter<std::string>("goal.debug_marker.frame_id", "map");
    impl_->goal.debug_marker.point_scale =
        static_cast<float>(this->declare_parameter<double>("goal.debug_marker.point_scale", 0.18));
    impl_->goal.debug_marker.text_scale =
        static_cast<float>(this->declare_parameter<double>("goal.debug_marker.text_scale", 0.16));
    impl_->goal.debug_marker.alpha =
        static_cast<float>(this->declare_parameter<double>("goal.debug_marker.alpha", 0.9));
    impl_->goal.debug_marker.z =
        static_cast<float>(this->declare_parameter<double>("goal.debug_marker.z", 0.08));
    impl_->goal.debug_marker.text_z_offset =
        static_cast<float>(this->declare_parameter<double>("goal.debug_marker.text_z_offset", 0.18));

    RCLCPP_INFO(
        get_logger(), "RGB config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d",
        impl_->rgb.enable ? "true" : "false", impl_->rgb.width, impl_->rgb.height, impl_->rgb.capture_fps,
        impl_->rgb.publish_fps, impl_->rgb.bitrate_kbps, impl_->rgb.gop_size);
    RCLCPP_INFO(
        get_logger(),
        "Depth config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d vis=[%d,%d]mm roi_valid=[%d,%d]mm",
        impl_->depth.enable ? "true" : "false", impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
        impl_->depth.publish_fps, impl_->depth.bitrate_kbps, impl_->depth.gop_size,
        impl_->depth.visualization_min_mm, impl_->depth.visualization_max_mm, impl_->depth.roi_min_valid_mm,
        impl_->depth.roi_max_valid_mm);
    RCLCPP_INFO(get_logger(), "Image config: rotation=%d degrees", impl_->image.rotation_degrees);
    RCLCPP_INFO(
        get_logger(),
        "Detector config: enable=%s framework=%s backend=%s available=[%s] model_dir=%s input=%d score=%.2f nms=%.2f fps=%d vulkan_device=%d cpu_threads=%d",
        impl_->detector.enable ? "true" : "false", impl_->detector.framework.c_str(), impl_->detector.backend.c_str(),
        availableDetectorFrameworksDescription().c_str(),
        impl_->detector.model_dir.empty() ? "<auto>" : impl_->detector.model_dir.c_str(), impl_->detector.input_size,
        impl_->detector.score_threshold, impl_->detector.nms_threshold, impl_->detector.fps,
        impl_->detector.vulkan_device_index, impl_->detector.cpu_num_threads);
    RCLCPP_INFO(
        get_logger(),
        "Depth estimation config: min_valid=%d bin=%dmm peak_ratio=%.2f peak_min=%d min_peak=%d trim=%.2f annotate_rgb=%s annotate_depth=%s marker=%s debug_hist=%s debug_peak=%s",
        impl_->depth.estimation.min_valid_pixels, impl_->depth.estimation.histogram_bin_size_mm,
        impl_->depth.estimation.peak_min_ratio, impl_->depth.estimation.peak_min_count,
        impl_->depth.estimation.min_peak_pixels, impl_->depth.estimation.trim_ratio,
        impl_->depth.estimation.annotate_depth_on_rgb ? "true" : "false",
        impl_->depth.estimation.annotate_depth_on_depth ? "true" : "false",
        impl_->depth.debug_marker.enable ? "true" : "false",
        impl_->depth.debug_video.histogram.enable ? "true" : "false",
        impl_->depth.debug_video.peak_mask.enable ? "true" : "false");
    RCLCPP_INFO(
        get_logger(),
        "Tracking config: enable=%s center_gate=%.2f lost_frames=%d min_iou=%.2f max_center=%.1f max_depth_delta=%.1f",
        impl_->tracking.enable ? "true" : "false", impl_->tracking.initial_center_gate_ratio,
        impl_->tracking.max_lost_frames, impl_->tracking.min_iou_for_match,
        impl_->tracking.max_center_distance_px, impl_->tracking.max_depth_delta_mm);
    RCLCPP_INFO(
        get_logger(),
        "Goal config: enable=%s global_frame=%s robot_frame=%s costmap=%s standoff=%.2fm max_target_distance=%.2fm stable_frames=%d clearance=%.2fm debug_marker=%s",
        impl_->goal.enable ? "true" : "false", impl_->goal.global_frame_id.c_str(),
        impl_->goal.robot_frame_id.c_str(), impl_->goal.costmap_topic.c_str(), impl_->goal.standoff_distance_m,
        impl_->goal.max_target_distance_m, impl_->goal.min_stable_frames, impl_->goal.clearance_radius_m,
        impl_->goal.debug_marker.enable ? "true" : "false");
}

void LCVision::prepareModel() {
    impl_->resetDetectorState();

    if (!impl_->detector.enable) {
        RCLCPP_INFO(get_logger(), "Detector disabled by parameter.");
        return;
    }

    if (!impl_->rgb.enable) {
        RCLCPP_WARN(get_logger(), "Detector requires RGB input; detector disabled because rgb.enable=false.");
        return;
    }

    try {
        DetectorFramework resolved_framework = DetectorFramework::None;
        impl_->detector_backend = createDetectorBackend(impl_->detector.framework, resolved_framework, get_logger());
        impl_->resolved_detector_framework = resolved_framework;
        if (impl_->detector_backend == nullptr) {
            RCLCPP_WARN(
                get_logger(), "No detector backend is available. Available frameworks in this build: %s",
                availableDetectorFrameworksDescription().c_str());
            return;
        }

        const fs::path model_dir = impl_->resolveModelDirectory(*impl_->detector_backend);
        if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
            throw std::runtime_error("Model directory does not exist: " + model_dir.string());
        }

        const fs::path    metadata_path = model_dir / "metadata.yaml";
        const ParsedMetadata metadata = parseMetadataFile(metadata_path);
        if (!metadata.class_names.empty()) {
            impl_->detector.class_names = metadata.class_names;
        }

        impl_->updateTargetClassIds();
        const auto preparation =
            impl_->detector_backend->prepare(impl_->detector, model_dir, metadata, get_logger());
        if (preparation.input_size.has_value()) {
            impl_->detector.input_size = *preparation.input_size;
        }

        impl_->active_backend = preparation.active_backend;
        impl_->detected_gpu_count = preparation.detected_gpu_count;
        impl_->active_gpu_index = preparation.active_gpu_index;
        impl_->detector_ready = true;
        impl_->detector_runtime_enabled.store(true);

        RCLCPP_INFO(
            get_logger(), "Loaded %s detector from %s on backend '%s' with input size %d",
            detectorFrameworkToString(impl_->resolved_detector_framework).c_str(), model_dir.string().c_str(),
            impl_->active_backend.c_str(), impl_->detector.input_size);
    } catch (const std::exception &ex) {
        RCLCPP_ERROR(get_logger(), "Failed to prepare detector, overlays disabled: %s", ex.what());
        impl_->resetDetectorState();
    }
}

void LCVision::sendGoal(const geometry_msgs::msg::PoseStamped &goal) {
    if (!nav_to_pose_->wait_for_action_server(500ms)) {
        RCLCPP_WARN(get_logger(), "navigate_to_pose action server is not available yet");
        impl_->navigation_goal_active.store(false);
        return;
    }

    NavigateToPose::Goal nav_goal;
    nav_goal.pose = goal;
    impl_->navigation_goal_active.store(true);

    using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;
    rclcpp_action::Client<NavigateToPose>::SendGoalOptions options;
    options.goal_response_callback = [this](const GoalHandleNavigateToPose::SharedPtr &goal_handle) {
        if (!goal_handle) {
            impl_->navigation_goal_active.store(false);
            RCLCPP_WARN(get_logger(), "Navigation goal was rejected by navigate_to_pose");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(impl_->current_nav_goal_mutex);
            impl_->current_nav_goal_handle = goal_handle;
        }
        RCLCPP_INFO(get_logger(), "Navigation goal accepted");
    };
    options.result_callback = [this](const GoalHandleNavigateToPose::WrappedResult &result) {
        impl_->navigation_goal_active.store(false);
        {
            std::lock_guard<std::mutex> lock(impl_->current_nav_goal_mutex);
            impl_->current_nav_goal_handle.reset();
        }
        switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(get_logger(), "Navigation goal succeeded");
            break;
        case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_WARN(get_logger(), "Navigation goal aborted");
            break;
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(get_logger(), "Navigation goal canceled");
            break;
        default:
            RCLCPP_WARN(get_logger(), "Navigation goal returned an unknown result code");
            break;
        }
    };

    nav_to_pose_->async_send_goal(nav_goal, options);
}

void LCVision::cancelCurrentNavigationGoal(const std::string &reason) {
    rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr goal_handle;
    {
        std::lock_guard<std::mutex> lock(impl_->current_nav_goal_mutex);
        goal_handle = impl_->current_nav_goal_handle;
    }

    if (!goal_handle) {
        impl_->navigation_goal_active.store(false);
        return;
    }

    RCLCPP_INFO(get_logger(), "Canceling current navigation goal: %s", reason.c_str());
    nav_to_pose_->async_cancel_goal(goal_handle);
}

rcl_interfaces::msg::SetParametersResult
LCVision::handleRuntimeParameters(const std::vector<rclcpp::Parameter> &parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    bool tracking_updated = false;
    bool goal_updated = false;

    for (const auto &parameter : parameters) {
        if (parameter.get_name() == "tracking.enable") {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "tracking.enable must be a boolean";
                return result;
            }
            impl_->tracking_runtime_enabled.store(parameter.as_bool());
            tracking_updated = true;
            continue;
        }

        if (parameter.get_name() == "goal.enable") {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "goal.enable must be a boolean";
                return result;
            }
            impl_->goal_runtime_enabled.store(parameter.as_bool());
            goal_updated = true;
        }
    }

    if (tracking_updated || goal_updated) {
        const bool tracking_enabled = impl_->tracking_runtime_enabled.load(std::memory_order_relaxed);
        const bool goal_enabled = impl_->goal_runtime_enabled.load(std::memory_order_relaxed);
        RCLCPP_INFO(
            get_logger(), "Runtime vision control updated: tracking.enable=%s goal.enable=%s",
            tracking_enabled ? "true" : "false", goal_enabled ? "true" : "false");

        if (!tracking_enabled || !goal_enabled) {
            cancelCurrentNavigationGoal("vision tracking disabled");
        }
    }

    return result;
}

} // namespace lc_vision

RCLCPP_COMPONENTS_REGISTER_NODE(lc_vision::LCVision)
