#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_map>

constexpr auto inputName = "Initial";
constexpr auto noiseName = "Poisson";

std::random_device rd{};
std::mt19937 gen{rd()};

// Pre-generate table
// std::poisson_distribution is slow
auto distributions = []() {
  std::array<std::poisson_distribution<uint8_t>, 255> result;
  for (int i = 0; i < result.size(); i++) {
    result[i] = std::poisson_distribution<uint8_t>(i);
  }
  return result;
}();

auto create_lut(const cv::Mat& image) {
  std::unordered_map<uint8_t, std::poisson_distribution<uint8_t>> result;

  uint8_t* in_pixel = reinterpret_cast<uint8_t*>(image.data);
  for (int i = 0; i < image.rows * image.cols * image.channels(); i++) {
    const auto value = *in_pixel++;
    if (result.end() == result.find(value)) {
      result[value] = std::poisson_distribution<uint8_t>(value);
    }
  }

  return result;
}

cv::Mat poisson(const cv::Mat& image, unsigned int seed) {
  gen.seed(seed);

  cv::Mat out(image.size(), image.type());

  // auto distributions = create_lut(image);

  uint8_t* in_pixel = reinterpret_cast<uint8_t*>(image.data);
  uint8_t* out_pixel = reinterpret_cast<uint8_t*>(out.data);
  for (int i = 0; i < image.rows * image.cols * image.channels(); i++) {
    const auto value = *in_pixel++;
    *out_pixel++ = distributions[value](gen);
  }

  return out;
}

void update(int, void* data) {
  if (!data) {
    std::cout << "Invalid pointer to callback data!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto& input_img = *static_cast<cv::Mat*>(data);
  const auto noised = poisson(input_img, 42);

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

  cv::namedWindow(inputName);
  cv::imshow(inputName, source);

  cv::namedWindow(noiseName);

  update(0, &source);

  while (true) {
    const char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
    if (ch == ' ') {
      update(0, &source);
    }
  }

  return EXIT_SUCCESS;
}
