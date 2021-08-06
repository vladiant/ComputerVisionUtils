FROM spmallick/opencv-docker:opencv

COPY camera_test.py /src/

WORKDIR /src/

# OpenCV-3.4.3-py3
# OpenCV-3.4.4-py3
# OpenCV-master-py3
CMD . /root/.virtualenvs/OpenCV-master-py3/bin/activate && exec python3 camera_test.py

