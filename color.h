#ifndef COLOR_H
#define COLOR_H

#include <stdio.h>

#include "common.h"
#include "vec3.h"

// A color is represented as a vector with three values normalized in the range
// [0, 1]
#define t_color t_vec3
#define color_new vec3_new

#define COLOR_BLACK  (t_color)   {0.0, 0.0, 0.0}
#define COLOR_GRAY   (t_color)   {0.5, 0.5, 0.5}
#define COLOR_WHITE  (t_color)   {1.0, 1.0, 1.0}
#define COLOR_RED    (t_color)   {1.0, 0.0, 0.0}
#define COLOR_BLUE   (t_color)   {0.0, 0.0, 1.0}
#define COLOR_GREEN  (t_color)   {0.0, 1.0, 0.0}
#define COLOR_SKY    (t_color)   {0.5, 0.7, 1.0}

// Return a blend (lerp) going  from  color1 and color2 based on blend factor a
__device__ inline t_color blend(float a, t_color color1, t_color color2) {
    return sum(
        scale(color1, (0.5 * (1.0 - a))), 
        scale(color2, (0.5 * (1.0 + a)))
    );
}

// Prints color to stderr in readable format
__host__ void color_print(t_color c) {
    fprintf(stderr, "%d %d %d\n", 
            (int) (255.999 * c.x), 
            (int) (255.999 * c.y), 
            (int) (255.999 * c.z));
}

#endif 