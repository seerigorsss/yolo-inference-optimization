// yolo_video — распознавание объектов в видео сетью YOLO (ONNX) на C++.
//
// Два бэкенда инференса:
//   --backend opencv  : OpenCV DNN. --device cpu | cuda | cuda-fp16
//                       (CUDA работает в OpenCV 4.x, собранном с WITH_CUDA/OPENCV_DNN_CUDA;
//                        в OpenCV 5.x новый DNN-движок пока только CPU).
//   --backend ort     : ONNX Runtime (если собрано с -DWITH_ORT=ON).
//                       --device cpu | cuda | trt (TensorRT EP) | coreml | openvino
//                       Понимает и INT8-модель (раздел 4 ноутбука), и модель «со сжатием внутри» (раздел 3)
//                       (вход uint8 [1,H,W,3] — кадр передаётся как есть, без препроцессинга).
//
// Пример:
//   yolo_video ../models/yolo26n.onnx ../data/demo.mp4 --backend opencv --out out.mp4
//   yolo_video ../models/yolo26n_int8.onnx ../data/demo.mp4 --backend ort
//   yolo_video ../models/yolo26n_fp16.onnx 0 --backend ort --device trt --show (0 = веб-камера)

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <opencv2/core.hpp>
#if CV_VERSION_MAJOR < 5
#include <opencv2/core/cuda.hpp>  // cv::cuda::getCudaEnabledDeviceCount()
#endif
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ------------------------------------------------------------------ параметры
struct Args {
    std::string model, source, backend = "opencv", device = "cpu", out = "out_cpp.mp4";
    int imgsz = 640, max_frames = 0;
    float conf = 0.25f, iou = 0.45f;
    bool show = false;
};

static void usage() {
    std::cout <<
        "Использование: yolo_video <model.onnx> <video.mp4 | номер камеры> [опции]\n"
        "  --backend opencv|ort      движок инференса (по умолчанию opencv)\n"
        "  --device  cpu|cuda|cuda-fp16|trt|coreml|openvino\n"
        "  --imgsz N                 размер входа сети (640)\n"
        "  --conf X --iou X          пороги (0.25 / 0.45)\n"
        "  --out file.mp4            видео с разметкой (out_cpp.mp4; \"\" — не сохранять)\n"
        "  --max-frames N            ограничить число кадров\n"
        "  --show                    показывать окно (q — выход)\n";
}

static Args parse(int argc, char** argv) {
    Args a;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("нет значения для " + k);
            return argv[++i];
        };
        if (k == "--backend") a.backend = next();
        else if (k == "--device") a.device = next();
        else if (k == "--imgsz") a.imgsz = std::stoi(next());
        else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--iou") a.iou = std::stof(next());
        else if (k == "--out") a.out = next();
        else if (k == "--max-frames") a.max_frames = std::stoi(next());
        else if (k == "--show") a.show = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else pos.push_back(k);
    }
    if (pos.size() != 2) { usage(); std::exit(1); }
    a.model = pos[0];
    a.source = pos[1];
    return a;
}

static const std::vector<std::string> COCO = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

// ------------------------------------------------------------------ pre/post-обработка
struct Letterbox {
    float r = 1.f;
    int padx = 0, pady = 0;
};

struct Detection {
    cv::Rect2f box;
    float score;
    int cls;
};

// Масштабирование с сохранением пропорций + серые поля до size x size; на выходе — blob NCHW float32 RGB 0..1
static cv::Mat make_blob(const cv::Mat& frame, int size, Letterbox& lb) {
    lb.r = std::min(size / float(frame.rows), size / float(frame.cols));
    int nw = int(std::round(frame.cols * lb.r)), nh = int(std::round(frame.rows * lb.r));
    cv::Mat resized, padded;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    lb.pady = (size - nh) / 2;
    lb.padx = (size - nw) / 2;
    cv::copyMakeBorder(resized, padded, lb.pady, size - nh - lb.pady, lb.padx, size - nw - lb.padx,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(), cv::Scalar(), /*swapRB=*/true, /*crop=*/false,
                                  CV_32F);
}

// Выход сети: (1, 4+nc, N) [cx,cy,w,h,scores...] -> нужен NMS; или (1, N, 6) [x1,y1,x2,y2,score,cls] — готово.
static std::vector<Detection> postprocess(const cv::Mat& out, float conf, float iou, const Letterbox& lb,
                                          cv::Size frame) {
    const int d1 = out.size[1], d2 = out.size[2];
    const float* p = out.ptr<float>();
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classes;

    if (d2 == 6) {  // end-to-end (NMS-free)
        for (int i = 0; i < d1; ++i) {
            const float* q = p + i * 6;
            if (q[4] < conf) continue;
            boxes.emplace_back(q[0], q[1], q[2] - q[0], q[3] - q[1]);
            scores.push_back(q[4]);
            classes.push_back(int(q[5]));
        }
    } else {
        const int nc = d1 - 4, n = d2;
        for (int i = 0; i < n; ++i) {
            int best = -1;
            float bs = conf;
            for (int c = 0; c < nc; ++c) {
                float s = p[(4 + c) * n + i];  // данные лежат «по строкам»: сначала все cx, потом все cy, ...
                if (s > bs) { bs = s; best = c; }
            }
            if (best < 0) continue;
            float cx = p[i], cy = p[n + i], w = p[2 * n + i], h = p[3 * n + i];
            boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
            scores.push_back(bs);
            classes.push_back(best);
        }
        // NMS по классам: сдвигаем рамки разных классов далеко друг от друга и делаем один общий NMS
        std::vector<cv::Rect> shifted;
        shifted.reserve(boxes.size());
        for (size_t k = 0; k < boxes.size(); ++k)
            shifted.emplace_back(int(boxes[k].x + classes[k] * 7680), int(boxes[k].y), int(boxes[k].width),
                                 int(boxes[k].height));
        std::vector<int> keep;
        cv::dnn::NMSBoxes(shifted, scores, conf, iou, keep);
        std::vector<cv::Rect2f> b2;
        std::vector<float> s2;
        std::vector<int> c2;
        for (int k : keep) { b2.push_back(boxes[k]); s2.push_back(scores[k]); c2.push_back(classes[k]); }
        boxes.swap(b2); scores.swap(s2); classes.swap(c2);
    }

    std::vector<Detection> dets;
    const cv::Rect2f frame_rect(0, 0, float(frame.width), float(frame.height));
    for (size_t k = 0; k < boxes.size(); ++k) {
        cv::Rect2f b((boxes[k].x - lb.padx) / lb.r, (boxes[k].y - lb.pady) / lb.r, boxes[k].width / lb.r,
                     boxes[k].height / lb.r);
        dets.push_back({b & frame_rect, scores[k], classes[k]});
    }
    return dets;
}

static void draw(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Scalar col((d.cls * 67) % 255, (d.cls * 151 + 80) % 255, (d.cls * 37 + 160) % 255);
        cv::rectangle(img, d.box, col, 2);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s %.2f", d.cls < int(COCO.size()) ? COCO[d.cls].c_str() : "?", d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::Point tl(int(d.box.x), std::max(int(d.box.y) - ts.height - 6, 0));
        cv::rectangle(img, cv::Rect(tl, cv::Size(ts.width + 4, ts.height + 6)), col, cv::FILLED);
        cv::putText(img, buf, tl + cv::Point(2, ts.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1,
                    cv::LINE_AA);
    }
}

// ------------------------------------------------------------------ бэкенды
class Backend {
public:
    virtual ~Backend() = default;
    // подготовить вход сети из кадра (для «модели со сжатием внутри» — просто сам кадр)
    virtual cv::Mat prepare(const cv::Mat& frame, Letterbox& lb, int imgsz) { return make_blob(frame, imgsz, lb); }
    virtual cv::Mat run(const cv::Mat& input) = 0;
    std::string name;
};

class OpenCVBackend : public Backend {
    cv::dnn::Net net;

public:
    OpenCVBackend(const std::string& path, const std::string& device) {
        net = cv::dnn::readNetFromONNX(path);
        name = "OpenCV " CV_VERSION " DNN CPU";
        if (device == "cuda" || device == "cuda-fp16") {
#if CV_VERSION_MAJOR >= 5
            std::cerr << "[!] В OpenCV 5.x новый DNN-движок пока работает только на CPU.\n"
                         "    Для GPU: OpenCV 4.x, собранный с CUDA, или --backend ort --device cuda|trt\n";
#else
            if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
                std::cerr << "[!] OpenCV собран без CUDA (или нет NVIDIA GPU) — работаю на CPU.\n"
                             "    Нужна сборка OpenCV с WITH_CUDA=ON, WITH_CUDNN=ON, OPENCV_DNN_CUDA=ON.\n";
            } else {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(device == "cuda-fp16" ? cv::dnn::DNN_TARGET_CUDA_FP16
                                                              : cv::dnn::DNN_TARGET_CUDA);
                name = std::string("OpenCV " CV_VERSION " DNN ") + (device == "cuda-fp16" ? "CUDA FP16" : "CUDA");
            }
#endif
        }
    }
    cv::Mat run(const cv::Mat& blob) override {
        net.setInput(blob);
        return net.forward().clone();
    }
};

#ifdef WITH_ORT
class OrtBackend : public Backend {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolo_video"};
    Ort::Session session{nullptr};
    std::string in_name, out_name;
    std::vector<int64_t> in_shape;
    bool raw_frame = false;  // модель из раздела 3 ноутбука: вход uint8 [1,H,W,3]
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

public:
    OrtBackend(const std::string& path, const std::string& device, int imgsz) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (device == "cuda") {
            OrtCUDAProviderOptions cuda;
            so.AppendExecutionProvider_CUDA(cuda);
        } else if (device == "trt") {
            const OrtApi& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* trt = nullptr;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
            // Точность задаётся моделью: yolo26n_fp16.onnx -> FP16, yolo26n_int8.onnx (Q/DQ) -> INT8.
            // Так работает и с TensorRT 10, и с TensorRT 11 (в 11.x флаги точности у сборщика убраны).
            std::vector<const char*> keys{"device_id", "trt_engine_cache_enable", "trt_engine_cache_path"};
            std::vector<const char*> vals{"0", "1", "trt_cache"};
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(), vals.data(), keys.size()));
            so.AppendExecutionProvider_TensorRT_V2(*trt);
            api.ReleaseTensorRTProviderOptions(trt);
            OrtCUDAProviderOptions cuda;  // запасной вариант для операций, которые TensorRT не поддерживает
            so.AppendExecutionProvider_CUDA(cuda);
        } else if (device == "coreml") {
            so.AppendExecutionProvider("CoreML", {{"MLComputeUnits", "ALL"}});
        } else if (device == "openvino") {
            so.AppendExecutionProvider_OpenVINO_V2({{"device_type", "CPU"}});
        } else if (device != "cpu") {
            throw std::runtime_error("неизвестное устройство для ORT: " + device);
        }
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), so);
#else
        session = Ort::Session(env, path.c_str(), so);
#endif
        Ort::AllocatorWithDefaultOptions alloc;
        in_name = session.GetInputNameAllocated(0, alloc).get();
        out_name = session.GetOutputNameAllocated(0, alloc).get();
        Ort::TypeInfo type_info = session.GetInputTypeInfo(0);  // держим объект живым: info ниже — «вид» на него
        auto info = type_info.GetTensorTypeAndShapeInfo();
        in_shape = info.GetShape();
        for (auto& d : in_shape)
            if (d < 0) d = (&d == &in_shape[0]) ? 1 : imgsz;  // динамические размеры -> 1 x 3 x imgsz x imgsz
        auto type = info.GetElementType();
        raw_frame = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        if (!raw_frame && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("поддерживаются модели с входом float32 (FP32/INT8 QDQ) или uint8");
        name = std::string("ONNX Runtime ") + Ort::GetVersionString() + " " + device +
               (raw_frame ? " (сжатие внутри модели)" : "");
    }

    cv::Mat prepare(const cv::Mat& frame, Letterbox& lb, int imgsz) override {
        if (!raw_frame) return make_blob(frame, imgsz, lb);
        if (frame.rows != in_shape[1] || frame.cols != in_shape[2])
            throw std::runtime_error("модель экспортирована для кадра " + std::to_string(in_shape[2]) + "x" +
                                     std::to_string(in_shape[1]) + " — пересоздайте её в разделе 3 ноутбука для этого видео");
        lb = Letterbox{};  // рамки уже в координатах кадра
        return frame.isContinuous() ? frame : frame.clone();
    }

    cv::Mat run(const cv::Mat& input) override {
        Ort::Value tensor =
            raw_frame ? Ort::Value::CreateTensor<uint8_t>(mem, const_cast<uint8_t*>(input.ptr<uint8_t>()), input.total() * 3,
                                                          in_shape.data(), in_shape.size())
                      : Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input.ptr<float>()), input.total(),
                                                        in_shape.data(), in_shape.size());
        const char* in_names[] = {in_name.c_str()};
        const char* out_names[] = {out_name.c_str()};
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int> sz(shape.begin(), shape.end());
        return cv::Mat(int(sz.size()), sz.data(), CV_32F, outs[0].GetTensorMutableData<float>()).clone();
    }
};
#endif

// ------------------------------------------------------------------ main
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        Args a = parse(argc, argv);

        std::unique_ptr<Backend> net;
        if (a.backend == "opencv") {
            net = std::make_unique<OpenCVBackend>(a.model, a.device);
        } else if (a.backend == "ort") {
#ifdef WITH_ORT
            net = std::make_unique<OrtBackend>(a.model, a.device, a.imgsz);
#else
            throw std::runtime_error("программа собрана без ONNX Runtime: cmake -DWITH_ORT=ON -DONNXRUNTIME_DIR=...");
#endif
        } else {
            throw std::runtime_error("неизвестный бэкенд: " + a.backend);
        }

        cv::VideoCapture cap;
        if (std::all_of(a.source.begin(), a.source.end(), ::isdigit)) cap.open(std::stoi(a.source));
        else cap.open(a.source);
        if (!cap.isOpened()) throw std::runtime_error("не удалось открыть видео: " + a.source);
        const int W = int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), H = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps_in = cap.get(cv::CAP_PROP_FPS);
        if (fps_in <= 0) fps_in = 25;
        cv::VideoWriter writer;
        if (!a.out.empty()) writer.open(a.out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_in, cv::Size(W, H));

        std::cout << "Модель:  " << a.model << "\nБэкенд:  " << net->name << "\nВидео:   " << a.source << " (" << W
                  << "x" << H << ", " << fps_in << " к/с)\n";

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) throw std::runtime_error("видео пустое");
        Letterbox lb;
        for (int i = 0; i < 3; ++i) net->run(net->prepare(frame, lb, a.imgsz));  // прогрев
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        double t_read = 0, t_pre = 0, t_inf = 0, t_post = 0, t_draw = 0;
        int n = 0;
        auto t_start = Clock::now();
        while (true) {
            auto t0 = Clock::now();
            if (!cap.read(frame) || frame.empty()) break;
            t_read += ms_since(t0);

            t0 = Clock::now();
            cv::Mat input = net->prepare(frame, lb, a.imgsz);
            t_pre += ms_since(t0);

            t0 = Clock::now();
            cv::Mat out = net->run(input);
            double inf = ms_since(t0);
            t_inf += inf;

            t0 = Clock::now();
            auto dets = postprocess(out, a.conf, a.iou, lb, frame.size());
            t_post += ms_since(t0);

            t0 = Clock::now();
            draw(frame, dets);
            char info[96];
            std::snprintf(info, sizeof(info), "C++ | %s | %.0f ms", a.backend.c_str(), inf);
            cv::putText(frame, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2, cv::LINE_AA);
            if (writer.isOpened()) writer.write(frame);
            t_draw += ms_since(t0);

            ++n;
            if (a.show) {
                cv::imshow("yolo_video", frame);
                if ((cv::waitKey(1) & 0xFF) == 'q') break;
            }
            if (a.max_frames && n >= a.max_frames) break;
        }
        double wall = ms_since(t_start);
        if (n == 0) throw std::runtime_error("не обработано ни одного кадра");

        const double model_ms = (t_pre + t_inf + t_post) / n;
        std::cout << std::fixed << std::setprecision(2) << "\nКадров: " << n << "\nСреднее время на кадр, мс:\n"
                  << "  read         " << t_read / n << "\n"
                  << "  preprocess   " << t_pre / n << "\n"
                  << "  inference    " << t_inf / n << "\n"
                  << "  postprocess  " << t_post / n << "\n"
                  << "  draw+write   " << t_draw / n << "\n"
                  << std::setprecision(1) << "FPS модели (pre+inference+post): " << 1000.0 / model_ms << "\n"
                  << "FPS всего конвейера: " << n * 1000.0 / wall << "\n";
        if (writer.isOpened()) std::cout << "Видео с разметкой: " << a.out << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
