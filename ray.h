#ifndef RAY_H
#define RAY_H

#include <math.h>

#include "common.h"
#include "hittable.h"
#include "material.h"
#include "point3.h"
#include "vec3.h"

typedef struct {
    t_point3 origin;
    t_vec3 direction;
} t_ray;

// Constructor
t_ray ray_new(const t_point3 origin, const t_vec3 direction) {
    t_ray r = {origin, direction};
    return r;
}

// Function to get the point at a given time t along the ray
t_point3 ray_at(const t_ray r, my_decimal t) {
    return sum(r.origin, scale(r.direction, t));
}

void ray_print(t_ray r) {
    fprintf(stderr, "Ray");
    fprintf(stderr, "\n  direction=");
    vec3_print(r.direction);
    fprintf(stderr, "  origin=");
    point3_print(r.origin); 
}

// Scatters the incident ray based on the material hit
bool scatter(t_hit_result *hit_result,
             t_ray *ray_in, 
             t_color *attenuation, 
             t_ray *ray_scattered) {

    // Declare variables beforehand to stop CUDA from complaining about them 
    // being initialized in every branch of the switch statement
    t_vec3 scatter_direction;
    my_decimal refraction_index, cos_theta, sin_theta;
    bool cannot_refract;
    switch (hit_result->surface_material) {

        case LAMBERTIAN:
            scatter_direction = sum(hit_result->normal, vec3_random_unit());
            scatter_direction = NEAR_ZERO(scatter_direction) ?
                                    hit_result->normal : scatter_direction;
            *attenuation = hit_result->albedo;
            break;

        case METAL:
            scatter_direction = REFLECT(ray_in->direction, hit_result->normal);
            scatter_direction = sum(scatter_direction, scale(vec3_random_unit(), hit_result->fuzz));
            *attenuation = hit_result->albedo;
            break;

        case DIELECTRIC:
            refraction_index = hit_result->front_face ? 
                ( 1.0/(hit_result->refraction_index) ) :
                hit_result->refraction_index;
            cos_theta = fmin(dot(negate(vec3_unit(ray_in->direction)), hit_result->normal), 1.0);
            sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        
            // Cannot refract, so it reflects (total internal reflection)
            // Reflectivity varying based on the angle is given by Shlick's 
            // Approximation
            cannot_refract = ((refraction_index*sin_theta) > 1.0);
                scatter_direction = (cannot_refract || reflectance(cos_theta, refraction_index) > random_my_decimal()) ?
                    REFLECT(vec3_unit(ray_in->direction), hit_result->normal) :
                    refract(vec3_unit(ray_in->direction), 
                                vec3_unit(hit_result->normal), 
                                refraction_index);

            *attenuation = color_new(1.0, 1.0, 1.0);
            break;

        default:
            fprintf(stderr, "Material not yet implemented %d! Aborting", 
                hit_result->surface_material);
            exit(1);
        }

        *ray_scattered = ray_new(hit_result->p, scatter_direction);
        // TODO Understand why
        return true;
}

#endif
