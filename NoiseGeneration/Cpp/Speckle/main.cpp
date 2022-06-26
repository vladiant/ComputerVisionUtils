#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

constexpr auto inputName = "Initial";
constexpr auto noiseName = "Speckle";

constexpr auto meanLabel = "mean";
constexpr auto varLabel = "var";

std::random_device rd{};
std::mt19937 gen{rd()};

cv::Mat speckle(const cv::Mat& image, float mean, float var,
                unsigned int seed) {
  gen.seed(seed);
  std::normal_distribution<float> d{mean, sqrt(var)};

  cv::Mat out(image.size(), image.type());

  float* in_pixel = reinterpret_cast<float*>(image.data);
  float* out_pixel = reinterpret_cast<float*>(out.data);
  for (int i = 0; i < image.rows * image.cols * image.channels(); i++) {
    *out_pixel++ = (d(gen) + 1.0) * (*in_pixel++);
  }

  cv::threshold(out, out, 1.0, 1.0, cv::THRESH_TRUNC);
  cv::threshold(out, out, 0.0, 0.0, cv::THRESH_TOZERO);

  return out;
}

void update(int, void* data) {
  const float mean = cv::getTrackbarPos(meanLabel, noiseName) / 100.0 - 0.5;
  const float var = cv::getTrackbarPos(varLabel, noiseName) / 2000.0;

  if (!data) {
    std::cout << "Invalid pointer to callback data!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto& input_img = *static_cast<cv::Mat*>(data);
  const auto noised = speckle(input_img, mean, var, 42);

  cv::imshow(noiseName, noised);
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " image_name" << '\n';
    return EXIT_SUCCESS;
  }

  std::string image_name = argv[1];

  cv::Mat source = cv::imread(image_name, cv::IMREAD_COLOR);
  if (source.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  // convert to float point data
  cv::Mat img;
  source.convertTo(img, CV_32F);
  img /= 255.0;

  cv::namedWindow(inputName);
  cv::imshow(inputName, img);

  cv::namedWindow(noiseName);

  // Initial trackbar values
  int mean = 50;
  int var = 20;

  cv::createTrackbar(meanLabel, noiseName, &mean, 100, update, &img);
  cv::createTrackbar(varLabel, noiseName, &var, 100, update, &img);

  update(0, &img);

  while (true) {
    const char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
    if (ch == ' ') {
      update(0, &img);
    }
  }

  return EXIT_SUCCESS;
}
