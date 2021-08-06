## Build for Linux
```bash
docker build -t pythoncv .
```

## Run in Linux
```bash
xhost +
docker run -it --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pythoncv
```

## Dev Run in Linux
```
xhost +
docker run -it --net=host --ipc=host --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pythoncv /bin/bash
```

## VSCode run
* See the `.devcontainer/devcontainer.json`

## References
* <https://learnopencv.com/install-opencv-docker-image-ubuntu-macos-windows/>
* <https://bobcares.com/blog/activate-python-virtualenv-in-dockerfile/>
* [Getting Started with Videos](https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html)
