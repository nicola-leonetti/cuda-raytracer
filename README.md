# SDM-raytracer

Simple raytracer written in CUDA, based on the book [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Compiling and running locally
The following compiles and runs the program, creating the image file `img.ppm` in the current directory.

```bash
nvcc main.cu -o main -lcurand
./main > img.ppm
```
