FROM spmallick/opencv-docker:opencv

COPY main.cpp CMakeLists.txt /src/

WORKDIR /src/build

RUN cmake .. && make

CMD ["./CameraStreamOpenCV"]

