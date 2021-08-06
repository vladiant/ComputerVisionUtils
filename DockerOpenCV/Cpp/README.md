## Build for Linux
```bash
docker build -t cppcv .
```

## Run in Linux
```bash
xhost +
docker run -it --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cppcv
```

## Dev Run in Linux
```
xhost +
docker run -it --net=host --ipc=host --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cppcv /bin/bash
```

## VSCode run
* See the `.devcontainer/devcontainer.json`

## References
* <https://learnopencv.com/install-opencv-docker-image-ubuntu-macos-windows/>
