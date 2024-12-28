#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

#define NUMBER_OF_SPHERES 489
t_sphere world[NUMBER_OF_SPHERES];

void init_world(t_sphere *world) {
    // Ground sphere (Lambertian material)
    world[0] = sphere_new(
        point3_new(0, -1000, 0), 1000, new_lambertian(COLOR_GRAY));

    world[1] = sphere_new(point3_new(0, 1, 0), 1.0, new_dielectric(1.5));
    world[2] = \
        sphere_new(point3_new(-4, 1, 0), 1.0, new_lambertian(COLOR_BLUE));
    world[3] = \
        sphere_new(point3_new(4, 1, 0), 1.0, new_metal(COLOR_GREEN, 0.0));

    // Create a grid of random spheres
    int index = 4;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            // Randomize material choice
            my_decimal choose_mat = random_my_decimal();
            t_point3 center = point3_new(a + 0.9 * random_my_decimal(), 0.2, b + 0.9 * random_my_decimal());
            t_material sphere_material;

            if (choose_mat < 0.8) {
                // Lambertian (diffuse)
                sphere_material = new_lambertian(
                    color_new(random_my_decimal()*random_my_decimal(), 
                              random_my_decimal()*random_my_decimal(), 
                              random_my_decimal()*random_my_decimal()
                            ));
                world[index++] = sphere_new(center, 0.2, sphere_material);
            } 
            else if (choose_mat < 0.95) {
                // Metal
                t_color color = color_new(random_my_decimal_in(0.5, 1), random_my_decimal_in(0.5, 1), random_my_decimal_in(0.5, 1));
                sphere_material = new_metal(color, random_my_decimal_in(0, 0.5));
                world[index++] = sphere_new(center, 0.2, sphere_material);
            } 
            else {
                // Dielectric (glass)
                world[index++] = sphere_new(center, 0.2, new_dielectric(1.5));
            }
            
        }
    }

}

int main() {
    // Initialize RNG
    srand((unsigned int) time(NULL));

    fprintf(stderr, "Initializing spheres...");
    init_world(world);
    fprintf(stderr, "\r                            \r");
    fprintf(stderr, "Sfere inizializzate\n");

    clock_t start = clock();

    t_camera cam = camera_new(ASPECT_RATIO, 
                              VIEWPORT_WIDTH, 
                              VERTICAL_FOV_DEGREES,
                              (t_point3) LOOK_FROM, 
                              (t_point3) LOOK_AT,
                              DEFOCUS_ANGLE,
                              FOCUS_DISTANCE);
    camera_render(&cam, world, NUMBER_OF_SPHERES);

    clock_t end  = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr, "Elapsed time: %.6fs\n", elapsed_time);
}
