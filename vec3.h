#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdio.h>

#include "common.h"

typedef struct { my_decimal x, y, z; } t_vec3;

// Constructor
t_vec3 vec3_new(my_decimal e0, my_decimal e1, my_decimal e2) {
    return (t_vec3) {e0, e1, e2};
}

t_vec3 sum(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)+(v.x), (u.y)+(v.y), (u.z)+(v.z));
}

t_vec3 subtract(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)-(v.x), (u.y)-(v.y), (u.z)-(v.z));
}
    
my_decimal squared_length(t_vec3 v) {
    return (v.x)*(v.x) + (v.y)*(v.y) + (v.z)*(v.z);
}

my_decimal length(t_vec3 v) {
    return sqrt(squared_length(v));
}

t_vec3 scale(t_vec3 v, my_decimal t) {
    return vec3_new(t*(v.x), t*(v.y), t*(v.z));
}

t_vec3 divide(t_vec3 v, my_decimal t) {
    return scale(v, 1.0/t);
}

t_vec3 negate(const t_vec3 v) {
    return scale(v, -1);
}

// Returns vector with same direction and unitary length
t_vec3 vec3_unit(t_vec3 v) {
    return divide(v, length(v));
}

// Element-wise multiplication
t_vec3 mul(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)*(v.x), (u.y)*(v.y), (u.z)*(v.z));
}

// Dot product
my_decimal dot(t_vec3 u, t_vec3 v) {
    return (u.x)*(v.x) + (u.y)*(v.y) + (u.z)*(v.z);
}

// Cross product
t_vec3 cross(t_vec3 u, t_vec3 v) {
    return (t_vec3) {
        (u.y)*(v.z) - (u.z)*(v.y), 
        (u.z)*(v.x) - (u.x)*(v.z),
        (u.x)*(v.y) - (u.y)*(v.x)
    };
}

void vec3_print(t_vec3 v) {
    fprintf(stderr, "Vector(%f, %f, %f)\n", v.x, v.y, v.z);
}

// Random vector with each component in [0, 1)
t_vec3 vec3_random() {
    return (t_vec3) {
        random_my_decimal(), 
        random_my_decimal(), 
        random_my_decimal()
    };
}

// Random vector with each component in [min, max)
t_vec3 vec3_random_in(my_decimal min, my_decimal max) {
    return (t_vec3) {
        random_my_decimal_in(min, max), 
        random_my_decimal_in(min, max), 
        random_my_decimal_in(min, max) 
    };
}

// TODO Optimize
// Random unit vector
t_vec3 vec3_random_unit() {
    while (true) {
        t_vec3 point = vec3_random();
        my_decimal length_squared = squared_length(point);
        if (MY_DECIMAL_UNDERFLOW_LIMIT < length_squared && length_squared <= 1) {
            return divide(point, sqrt(length_squared));
        }
    }
}

// Random unit vector on a surface with given surface normal
t_vec3 vec3_random_on_hemisphere(t_vec3 normal) {
    t_vec3 v = vec3_random_unit();
    return (dot(v, normal) > 0) ? v : scale(v, -1.0); 
}

// Random unit vector in a unit disk
t_vec3 random_in_unit_disk() {
    while (true) {
        t_vec3 p = vec3_new(random_my_decimal_in(-1, 1), random_my_decimal_in(-1, 1), 0);
        if (squared_length(p) < 1) {
            return p;
        }
    }
}

#define NEAR_ZERO(v) \
    (fabs(v.x) < NEAR_ZERO_TRESHOLD) && \
    (fabs(v.y) < NEAR_ZERO_TRESHOLD) && \
    (fabs(v.z) < NEAR_ZERO_TRESHOLD)

// Calculate reflected ray over a surface with given normal
#define REFLECT(v, n) \
    subtract(v, scale(n, 2*dot(v, n)))

// Calculate refracted ray over a surface with given normal and a given ratio 
// of refractive indexes
t_vec3 refract(t_vec3 v, t_vec3 n, my_decimal ratio) {
    my_decimal cos_theta = fmin(dot(negate(v), n), 1.0);
    t_vec3 out_perpendicular = scale(sum(v, scale(n, cos_theta)), ratio);
    t_vec3 out_parallel = scale(
        n, -sqrt(fabs(1 - squared_length(out_perpendicular))));
    return(sum(out_perpendicular, out_parallel));
}

#endif
