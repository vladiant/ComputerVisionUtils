#include <cstddef>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char* argv[]) {
  cv::VideoCapture cap;

  if (!cap.open(0)) {
    std::cout << "Error opening camera" << '\n';
    return EXIT_FAILURE;
  }

  const auto window_name = "Frame";
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);

  cv::Mat frame;

  for (;;) {
    cap >> frame;

    if (frame.empty()) {
      break;
    }

    // Do processing here

    cv::imshow(window_name, frame);

    // Press ESC to leave
    if (cv::waitKey(10) == 27) {
      break;
    }
  }

  cap.release();

  std::cout << "Done.\n";

  return EXIT_SUCCESS;
}
