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
__device__ inline t_ray ray_new(const t_point3 origin, const t_vec3 direction) {
    t_ray r = {origin, direction};
    return r;
}

// Function to get the point at a given time t along the ray
__device__ void ray_at(t_point3 *point, const t_ray r, my_decimal t) {
    *point = sum(r.origin, scale(r.direction, t));
}

__host__ void ray_print(t_ray r) {
    fprintf(stderr, "Ray");
    fprintf(stderr, "\n  direction=");
    vec3_print(r.direction);
    fprintf(stderr, "  origin=");
    point3_print(r.origin); 
}

// Scatters the incident ray based on the material hit
__device__ void scatter(
    t_ray *scattered_ray,
    t_hit_result *hit_result, 
    t_ray *ray_in, 
    t_color *attenuation, 
    curandState *random_state
) {

    t_vec3 scatter_direction;
    my_decimal cos_theta, sin_theta, ri, reflectance;
    bool cannot_refract, til;
    switch (hit_result->surface_material) {

    case LAMBERTIAN:
        scatter_direction = sum(hit_result->normal, d_vec3_random_unit(random_state));
        scatter_direction = NEAR_ZERO(scatter_direction) ?
                                hit_result->normal : scatter_direction;
        *attenuation = hit_result->albedo;
        break;

    case METAL:
        scatter_direction = reflect(ray_in->direction, hit_result->normal);
        scatter_direction = sum(scatter_direction, scale(d_vec3_random_unit(random_state), hit_result->fuzz));
        *attenuation = hit_result->albedo;
        break;

    case DIELECTRIC:
        ri = hit_result->front_face ? 
            ( 1.0/(hit_result->refraction_index) ) :
            hit_result->refraction_index;
        cos_theta = fmin(
            dot(
                negate(vec3_unit(ray_in->direction)), 
                hit_result->normal
            ), 
            1.0
        );
        sin_theta = sqrt(1.0 - cos_theta*cos_theta);
    
        // // Cannot refract, so it reflects (total internal reflection)
        // // Reflectivity varying based on the angle is given by Shlick's 
        // // Approximation
        cannot_refract = ((ri*sin_theta) > 1.0);
        reflectance = get_reflectance(cos_theta, ri);
        til = reflectance > d_random_my_decimal(random_state);
        
        scatter_direction = (cannot_refract || til) ?
            reflect(
                vec3_unit(ray_in->direction), 
                hit_result->normal
            ) :
            refract(
                vec3_unit(ray_in->direction), 
                vec3_unit(hit_result->normal), 
                ri
            );

        *attenuation = color_new(1.0, 1.0, 1.0);
        break;

    default:
        // TODO Trovare un modo per fare error handling senza scrivere su 
        // stdout
        printf("ERROR while scattered: material %d not yet implemented", 
            hit_result->surface_material );  
        break;
    }
        
    t_point3 origin = hit_result->p;
    printf("Origin: (%f %f %f), scatter direction: (%f %f %f)", origin.x,origin.y,origin.z,scatter_direction.x, scatter_direction.y, scatter_direction.z); 
    // *scattered_ray = (t_ray) {origin, scatter_direction};
}

#endif
