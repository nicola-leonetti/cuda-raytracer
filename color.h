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
#define BLEND(a, color1, color2) \
        sum(scale(color1, (0.5 * (1.0 - a))), \
        scale(color2, (0.5 * (1.0 + a))))

#define TO_GAMMA(a) \
        (a > 0) ? sqrt(a) : a

// Prints color to stderr in readable format
void color_print(t_color c) {
    fprintf(stderr, "%d %d %d\n", 
            (int) (255.999 * c.x), 
            (int) (255.999 * c.y), 
            (int) (255.999 * c.z));
}

// Prints the color to stdout in PPM format
void color_print_PPM(t_color c) {
    // Clamp color RGB components to interval [0, 0.999]
    c.x = (c.x > 0.999) ? 0.999 : c.x;
    c.y = (c.y > 0.999) ? 0.999 : c.y;
    c.z = (c.z > 0.999) ? 0.999 : c.z;
    c.x = (c.x < 0) ? 0 : c.x;
    c.y = (c.y < 0) ? 0 : c.y;
    c.z = (c.z < 0) ? 0 : c.z;

    my_decimal r = TO_GAMMA(c.x);
    my_decimal g = TO_GAMMA(c.y);
    my_decimal b = TO_GAMMA(c.z);

    printf("%d %d %d\n", 
            (int) (255.999 * r), 
            (int) (255.999 * g), 
            (int) (255.999 * b));
}

#endif 