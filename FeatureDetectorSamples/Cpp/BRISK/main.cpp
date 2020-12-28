#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

constexpr auto kWindowName = "BRISK";
constexpr auto kThresholdTaskbar = "Threshold";
constexpr auto kOctavesTaskbar = "Octaves";
constexpr auto kPatternScaleTaskbar = "Pattern Scale";

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    std::cout << "Format to call: " << argv[0] << " image name" << '\n';
    return EXIT_FAILURE;
  }

  const std::string image_filename = argv[1];

  auto img = cv::imread(image_filename);

  if (img.empty()) {
    std::cout << "Error reading: " << image_filename << '\n';
    return EXIT_FAILURE;
  }

  cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

  // Detector parameters
  int current_threshold = 30;
  int current_octaves = 3;
  int current_scale = 3;

  // Detector parameters modifiers
  cv::createTrackbar(kThresholdTaskbar, kWindowName, &current_threshold, 128);
  cv::createTrackbar(kOctavesTaskbar, kWindowName, &current_octaves, 9);
  cv::createTrackbar(kPatternScaleTaskbar, kWindowName, &current_scale, 9);

  while (true) {
    // Update detector parameters
    current_threshold = cv::getTrackbarPos(kThresholdTaskbar, kWindowName);
    current_octaves = cv::getTrackbarPos(kOctavesTaskbar, kWindowName);
    current_scale =
        (cv::getTrackbarPos(kPatternScaleTaskbar, kWindowName) + 1) / 3;

    auto detector =
        cv::BRISK::create(current_threshold, current_octaves, current_scale);
    if (!detector) {
      std::cout << "Error creating feature detector\n";
      return EXIT_FAILURE;
    }

    std::vector<cv::KeyPoint> kp;
    detector->detect(img, kp);

    cv::Mat out_img;
    cv::drawKeypoints(img, kp, out_img, cv::Scalar(0, 255, 0));

    cv::imshow(kWindowName, out_img);

    // Press ESC to leave
    if (cv::waitKey(10) == 27) {
      break;
    }
  }

  return EXIT_SUCCESS;
}