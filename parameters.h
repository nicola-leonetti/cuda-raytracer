#ifndef PARAMETERS_H
#define PARAMETERS_H

#define my_decimal double

#define RNG_SEED 1

// CUDA block size
dim3 block(32, 32); // 1024 threads per block

// Minimum value that is safe to square without underflow
#define MY_DECIMAL_UNDERFLOW_LIMIT 1e-18

#define NEAR_ZERO_TRESHOLD 1e-8

#define ASPECT_RATIO 16.0/9.0
#define VIEWPORT_WIDTH 1400

// Points from thich the camera looks from and towards which it looks at
#define LOOK_FROM {20, 4, 20}
#define LOOK_AT {0, 0, 0}

// What the camera considers to be "up"
#define UP_DIRECTION {0, 1, 0}

// Variation angle of rays through each pixel
#define DEFOCUS_ANGLE 0

// How distant a plane with perfect focus is from the lookfrom point
#define FOCUS_DISTANCE 20.0

// Vertical field of view, aka the angle between the image top and bottom.
// Expressed in degrees
#define VERTICAL_FOV_DEGREES 20

// How many rays to send inside each square pixel in order to obtain a more 
// precise color (antialiasing).
#define SAMPLES_PER_PIXEL 1

// Limit to how many times a ray can bounce from a surface to another before 
// determining its color. This is done in order to avoid stack overflow
#define MAX_RAY_BOUNCES 10

#endif