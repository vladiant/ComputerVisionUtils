# Run C++ OpenCV samples using CUDA

## Compile OpenCV with CUDA support

TBD

## Export path to C++ libraries

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

where `/usr/local/lib` is the default path to OpenCV libraries

## Compile program

* Check in `Makefile` that paths to OpenCV includes and libraries are correct
* Run:

```
make
```
