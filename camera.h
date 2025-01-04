#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "common.h"
#include "point3.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

#define RAY_T_MAX 9999

typedef struct {
    int image_width, image_height;

    // Point of space from which rays originate
    t_point3 center;
    
    // The viewport is a virtual rectangle in our 3D world containing the 
    // square pixels that make up the rendered image.
    // One or more rays are sent from the camera center through each pixel of 
    // the viewport and the intersection point with the objects in the scene is 
    // determined, which in turn determines the pixel's color.
    // Vectors used to navigate the viewpoer through its width and down its 
    // height
    t_vec3 pixel_delta_u, pixel_delta_v;

    // Center of upper-left pixel in the viewport
    t_point3 pixel00;

    // Used later to determine ray origin
    bool defocus_angle_is_negative;

    // Horizontal and vertical radius of defocus disk
    t_vec3 defocus_disk_u, defocus_disk_v;
} t_camera;

// Constructor
__host__ t_camera camera_new(my_decimal aspect_ratio, 
                    int image_width, 
                    my_decimal vertical_fov,
                    t_point3 look_from, 
                    t_point3 look_at,
                    my_decimal defocus_angle,
                    my_decimal focus_distance) {
    t_camera cam;

    cam.image_width = image_width;
    cam.image_height = (int) (image_width/aspect_ratio);
    cam.image_height = (cam.image_height < 1) ? 1 : cam.image_height;
    
    cam.center = look_from;

    // pixel_delta_u, pixel_delta_v
    t_vec3 w = vec3_unit(subtract(look_from, look_at));
    t_vec3 u = vec3_unit(cross((t_vec3) UP_DIRECTION, w));
    t_vec3 v = cross(w, u);
    my_decimal viewport_height = \
        2.0 * tan(degrees_to_radians(vertical_fov)/2.0) * focus_distance;
    my_decimal viewport_width = viewport_height * image_width/cam.image_height;
    t_vec3 viewport_u = scale(u, viewport_width);
    t_vec3 viewport_v = scale(v, -viewport_height);
    cam.pixel_delta_u = divide(viewport_u, cam.image_width);
    cam.pixel_delta_v = divide(viewport_v, cam.image_height);

    cam.defocus_angle_is_negative = (defocus_angle <= 0);

    cam.pixel00 = cam.center;
    cam.pixel00 = subtract(cam.pixel00, scale(w, focus_distance));
    cam.pixel00 = subtract(cam.pixel00, divide(viewport_u, 2));
    cam.pixel00 = subtract(cam.pixel00, divide(viewport_v, 2));
    cam.pixel00 = sum(cam.pixel00, scale(
        sum(cam.pixel_delta_u, cam.pixel_delta_v), 0.5));

    // defocus_disk_u, defocus_disk_v
    my_decimal defocus_radius = \
        focus_distance * tan(degrees_to_radians(defocus_angle / 2));
    cam.defocus_disk_u = scale(u, defocus_radius);
    cam.defocus_disk_v = scale(v, defocus_radius);

    return cam;
}

// Return a random ray with the end in a random point inside the (i, j) pixel.
__device__ t_ray get_random_ray(
    t_camera *cam, 
    int i, 
    int j, 
    curandState *random_state
) {
    // Offset from the center of the vector generated in the unit square
    // [-0.5, 0.5]x[-0.5, 0.5]
    t_vec3 offset  = vec3_new(d_random_my_decimal(random_state) - 0.5, 
                            d_random_my_decimal(random_state) - 0.5, 0.0);

    // Use the offset to select a random point inside the (i, j) pixel
    t_point3 pixel_sample = cam->pixel00;
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_u, i+offset.x));
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_v, j+offset.y));

    t_point3 p = random_in_unit_disk(random_state);
    t_point3 defocus_disk_sample = cam->center;
    defocus_disk_sample = \
        sum(defocus_disk_sample, scale(cam->defocus_disk_u, p.x));
    defocus_disk_sample = \
        sum(defocus_disk_sample, scale(cam->defocus_disk_v, p.y));
    t_point3 ray_origin = (cam->defocus_angle_is_negative) ? 
                            cam->center : defocus_disk_sample;
    t_vec3 ray_direction = subtract(pixel_sample, ray_origin);

    return ray_new(ray_origin, ray_direction);
}

// Determining a different color for each of the pixels of the viewport by 
// sending one or more rays from the camera center to each pixel 
__device__ t_color ray_color(
    t_ray *r,
    t_sphere world[], 
    int number_of_spheres, 
    int max_bounces, 
    curandState *random_state
) {
    
    t_color accumulated_color = COLOR_WHITE;
    t_ray current_ray = *r;
    int bounces_remaining = max_bounces;

    while (bounces_remaining > 0) {
        bool hit_anything = false;
    //     t_hit_result temp, closest_hit_result;
    //     my_decimal closest_hit = RAY_T_MAX;

    //     // Check the first intersection with the spheres
    //     // (0.001 lower bound is used to fix "shadow acne")
    //     for (int i = 0; i < number_of_spheres; i++) {
    //         sphere_hit(&temp, &current_ray, world[i], 0.001, closest_hit);
    //         if (temp.did_hit) {
    //             hit_anything = true;
    //             closest_hit = temp.t;
    //             closest_hit_result = temp;
    //         }
    //     }

    //     // If no object is hit, return a blend between blue and white based on the 
    //     // y coordinate, so going vertically from white all the way to blue
        if (!hit_anything) {
            float unit_direction_y = vec3_unit(current_ray.direction).y;
            return blend(abs(unit_direction_y), COLOR_WHITE, COLOR_SKY);
        } 

    //     t_vec3 scatter_direction;
    //     my_decimal cos_theta, sin_theta, ri, reflectance;
    //     bool cannot_refract, til;
    //     switch (closest_hit_result.surface_material) {
            
    //     case LAMBERTIAN:
    //         scatter_direction = sum(
    //             closest_hit_result.normal, 
    //             d_vec3_random_unit(random_state)
    //         );
    //         if (NEAR_ZERO(scatter_direction)) {
    //             scatter_direction = closest_hit_result.normal;
    //         }
    //         accumulated_color = mul(
    //             accumulated_color,
    //             closest_hit_result.albedo
    //         );
    //         break;
        
    //     case METAL:
    //         scatter_direction = reflect(
    //             r->direction, 
    //             closest_hit_result.normal
    //         );
    //         scatter_direction = sum(
    //             scatter_direction, 
    //             scale(
    //                 d_vec3_random_unit(random_state), 
    //                 closest_hit_result.fuzz)
    //             );
    //         accumulated_color = mul(
    //             accumulated_color, 
    //             closest_hit_result.albedo
    //         );
    //         break;

    //     case DIELECTRIC:
    //         ri = closest_hit_result.front_face ? 
    //                 1.0/(closest_hit_result.refraction_index):
    //                 closest_hit_result.refraction_index;
    //         cos_theta = fmin(
    //             dot(
    //                 negate(vec3_unit(r->direction)), 
    //                 closest_hit_result.normal
    //             ), 
    //             1.0
    //         );
    //         sin_theta = sqrt(1.0 - cos_theta*cos_theta);
    
    //         // // Cannot refract, so it reflects (total internal reflection)
    //         // // Reflectivity varying based on the angle is given by Shlick's 
    //         // // Approximation
    //         cannot_refract = ((ri*sin_theta) > 1.0);
    //         reflectance = get_reflectance(cos_theta, ri);
    //         til = reflectance > d_random_my_decimal(random_state);

    //         if (cannot_refract || til) {
    //             scatter_direction = reflect(
    //                 vec3_unit(r->direction), 
    //                 closest_hit_result.normal
    //             );
    //         }
    //         else {
    //             scatter_direction = refract(
    //                 vec3_unit(r->direction), 
    //                 vec3_unit(closest_hit_result.normal), 
    //                 ri
    //             );
    //         }
    //         break;

    //     default:
    //         // TODO Trovare un modo per fare error handling senza scrivere su 
    //         // stdout
    //         printf("ERROR while scattered: material %d not yet implemented", 
    //             closest_hit_result.surface_material );  
    //         break;
    //     }

    //     current_ray = ray_new(closest_hit_result.p, scatter_direction);
        bounces_remaining--;
    }

    // Limit the amount of bounces
    return accumulated_color;
}


__global__ void camera_render(
    t_camera *cam, 
    t_sphere world[],
    int number_of_spheres, 
    unsigned char *result_img,
    curandState random_states[]
) {

    int max_ray_bounces = MAX_RAY_BOUNCES;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i < (cam->image_width)) && (j < (cam->image_height))) {

        long long rgb_offset = (j*(cam->image_width) + i)*3;
        curandState state = random_states[j*(cam->image_width) + i];

        // Antialiasing: sample SAMPLE_PER_PIXEL colors and average them to
        // obtain pixel color
        t_color pixel_color = color_new(0, 0, 0);
        for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
            t_ray random_ray = get_random_ray(cam, i, j, &state);
            t_color sampled_color = ray_color(
                &random_ray, 
                world, 
                number_of_spheres, 
                max_ray_bounces, 
                &state
            );
            pixel_color = sum(pixel_color, sampled_color);
        } 
        pixel_color = divide(pixel_color, SAMPLES_PER_PIXEL);
        
        // // Clamp color RGB components to interval [0, 0.999]
        pixel_color.x = (pixel_color.x > 0.999) ? 0.999 : pixel_color.x;
        pixel_color.y = (pixel_color.y > 0.999) ? 0.999 : pixel_color.y;
        pixel_color.z = (pixel_color.z > 0.999) ? 0.999 : pixel_color.z;
        pixel_color.x = (pixel_color.x < 0) ? 0 : pixel_color.x;
        pixel_color.y = (pixel_color.y < 0) ? 0 : pixel_color.y;
        pixel_color.z = (pixel_color.z < 0) ? 0 : pixel_color.z;

        // sqrt performs gamma correction
        result_img[rgb_offset] = \
            (unsigned char) (255.999 * sqrt(pixel_color.x));
        result_img[rgb_offset+1] = \
            (unsigned char) (255.999 * sqrt(pixel_color.y));
        result_img[rgb_offset+2] = \
            (unsigned char) (255.999 * sqrt(pixel_color.z));
    }
    // TODO Add progress bar
}

#endif