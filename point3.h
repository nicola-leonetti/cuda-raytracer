#ifndef POINT3_H
#define POINT3_H

#include "common.h"
#include "vec3.h"

#define t_point3 t_vec3

#define point3_new vec3_new

__host__ void point3_print(const t_point3 v) {
    fprintf(stderr, "Point(%f, %f, %f)\n", v.x, v.y, v.z);
}

#endif