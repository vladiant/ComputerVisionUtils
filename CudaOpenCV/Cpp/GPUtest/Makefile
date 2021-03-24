# Default OpenCV include dir: /usr/local/include
# Default OpenCV libraries dir: /usr/local/lib

CPPFLAGS=-std=c++11 -I/usr/local/include/opencv4
LDFLAGS=-Wall -Wextra
LDLIBS=-L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc

all: gpu_test

gpu_test: gpu_test.o
	 g++ $(LDFLAGS) -o gpu_test gpu_test.o $(LDLIBS)

gpu_test.o: gpu_test.cpp
	 g++ $(CPPFLAGS) -c gpu_test.cpp
     
clean:
	 rm gpu_test.o gpu_test