# Default OpenCV include dir: /usr/local/include
# Default OpenCV libraries dir: usr/local/lib

CPPFLAGS=-std=c++11 -I/usr/local/include/opencv4
LDFLAGS=-Wall -Wextra
LDLIBS=-L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video

all: cpu_pipeline

cpu_pipeline: cpu_pipeline.o
	 g++ $(LDFLAGS) -o cpu_pipeline cpu_pipeline.o $(LDLIBS)

cpu_pipeline.o: cpu_pipeline.cpp
	 g++ $(CPPFLAGS) -c cpu_pipeline.cpp
     
clean:
	 rm cpu_pipeline.o cpu_pipeline
