# Install OpenCV from source with CUDA support

## Minimal Commands

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/path_to/opencv_contrib/modules -DWITH_CUDA=ON ..
make
sudo make install
sudo ldconfig
```

## Add specific compilers

To solve the probled descibed in <https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version> add:
```bash
-DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8
```

## Use different than default installation directories

```bash
make DESTDIR=/path_to/opencv_install install
export LD_LIBRARY_PATH=/path_to/opencv_install/usr/local/lib
```

## Run Docker OpenCV
Image: <https://hub.docker.com/r/spmallick/opencv-docker>

```bash
docker run --device=/dev/video0:/dev/video0 \
           -v /tmp/.X11-unix:/tmp/.X11-unix \ 
           -e DISPLAY=$DISPLAY \
           -p 5000:5000 -p 8888:8888 -it \
           spmallick/opencv-docker:opencv \
           /bin/bash
```

## References
* <https://github.com/NeerajGulia/python-opencv-cuda>
* <https://hub.docker.com/r/jjanzic/docker-python3-opencv>
* <https://github.com/janza/docker-python3-opencv/blob/master/Dockerfile>
* <https://learnopencv.com/install-opencv-docker-image-ubuntu-macos-windows/>
* <https://stackoverflow.com/questions/3239343/make-install-but-not-to-default-directories>
