# Default OpenCV include dir: /usr/local/include
# Default OpenCV libraries dir: /usr/local/lib

CPPFLAGS=-std=c++11 -I/usr/local/include/opencv4
LDFLAGS=-Wall -Wextra
LDLIBS=-L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
    -lopencv_cudaarithm -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaoptflow

all: gpu_pipeline

gpu_pipeline: gpu_pipeline.o
	 g++ $(LDFLAGS) -o gpu_pipeline gpu_pipeline.o $(LDLIBS)

gpu_pipeline.o: gpu_pipeline.cpp
	 g++ $(CPPFLAGS) -c gpu_pipeline.cpp
     
clean:
	 rm gpu_pipeline.o gpu_pipeline
