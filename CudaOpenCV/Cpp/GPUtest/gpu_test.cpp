#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>

// https://learnopencv.com/getting-started-opencv-cuda-modul/

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Program usage: " << argv[0] << " image_filename\n";
    return -1;
  }

  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::cuda::GpuMat dst;
  cv::cuda::GpuMat src;
  src.upload(img);

  cv::Ptr<cv::cuda::CLAHE> ptr_clahe =
      cv::cuda::createCLAHE(5.0, cv::Size(8, 8));
  ptr_clahe->apply(src, dst);

  cv::Mat result;
  dst.download(result);

  cv::imshow("result", result);
  cv::waitKey();

  return 0;
}
