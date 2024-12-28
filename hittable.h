#ifndef HITTABLE_H
#define HITTABLE_H

#include "common.h"
#include "material.h"
#include "point3.h"
#include "vec3.h"

// Stores some information about a ray hitting a hittable object
typedef struct {
    bool did_hit;
    my_decimal t;
    t_point3 p;

    // Always points towards outside the surface.
    // Not normalized to length 1 here, but later in the geometry code
    t_vec3 normal;
    
    // Set to true if the ray hits the object on its front face
    bool front_face;

    material_type surface_material;
    t_color albedo;
    my_decimal fuzz;
    my_decimal refraction_index;
} t_hit_result;

#endif