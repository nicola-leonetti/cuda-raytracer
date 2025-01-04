#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "common.h"
#include "ray.h"

typedef enum {
    LAMBERTIAN, 
    METAL,
    DIELECTRIC
} material_type;

typedef struct {
    material_type type;

    // For lambertian and metal
    t_color albedo;

    // For metal
    my_decimal fuzz;

    // For dielectric
    my_decimal refraction_index;
} t_material;

__host__ t_material new_lambertian(t_color albedo) {
    t_material material;
    material.type = LAMBERTIAN;
    material.albedo = albedo;
    material.fuzz = -1;
    material.refraction_index = -1;
    return material;
}

__host__ t_material new_metal(t_color albedo, my_decimal fuzz) {
    t_material material;
    material.type = METAL;
    material.albedo = albedo;
    material.fuzz = (fuzz < 1) ? fuzz : 1;
    material.refraction_index = -1;
    return material;
}

__host__ t_material new_dielectric(my_decimal refraction_index) {
    t_material material;
    material.type = DIELECTRIC;
    material.albedo = (t_color){-1, -1, -1};
    material.fuzz = -1;
    material.refraction_index = refraction_index;
    return material;
}

// Reflectance of a dielectric
__device__ inline my_decimal get_reflectance(my_decimal cosine, my_decimal refraction_index) {
    my_decimal r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine), 5);
}


#endif 