#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

constexpr auto inputName = "Initial";
constexpr auto noiseName = "Salt&Pepper";

constexpr auto amountLabel = "amount";
constexpr auto saltVsPepperLabel = "salt_vs_pepper";

std::random_device rd{};
std::mt19937 gen{rd()};

cv::Mat snp(const cv::Mat& image, float amount, float salt_vs_pepper,
            unsigned int seed) {
  gen.seed(seed);
  std::bernoulli_distribution d{amount};
  std::bernoulli_distribution svp{salt_vs_pepper};

  cv::Mat out(image.size(), image.type());

  uint8_t* in_pixel = reinterpret_cast<uint8_t*>(image.data);
  uint8_t* out_pixel = reinterpret_cast<uint8_t*>(out.data);
  for (int i = 0; i < image.rows * image.cols * image.channels();
       i++, in_pixel++, out_pixel++) {
    if (!d(gen)) {
      *out_pixel = *in_pixel;
    } else {
      if (!svp(gen)) {
        *out_pixel = 255;
      } else {
        *out_pixel = 0;
      }
    }
  }

  return out;
}

void update(int, void* data) {
  const float amount = cv::getTrackbarPos(amountLabel, noiseName) / 100.0;
  const float salt_vs_pepper =
      cv::getTrackbarPos(saltVsPepperLabel, noiseName) / 100.0;

  if (!data) {
    std::cout << "Invalid pointer to callback data!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto& input_img = *static_cast<cv::Mat*>(data);
  const auto noised = snp(input_img, amount, salt_vs_pepper, 42);

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

  // Initial trackbar values
  int amount = 50;
  int salt_vs_pepper = 50;

  cv::createTrackbar(amountLabel, noiseName, &amount, 100, update, &source);
  cv::createTrackbar(saltVsPepperLabel, noiseName, &salt_vs_pepper, 100, update,
                     &source);

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
