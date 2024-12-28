#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

#define NUMBER_OF_SPHERES 489
t_sphere world[NUMBER_OF_SPHERES];

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9);
}

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

void write_PPM_img_to_stdout(unsigned char *img, int width, int height) {
    // PPM header
    printf("P3\n%d %d\n255\n", width, height);
    for (int pixel = 0; pixel < width*height*3; pixel+=3) {
        printf("%d %d %d\n", img[pixel], img[pixel+1], img[pixel+2]);
    }
}

int main() {
    // Initialize RNG
    srand((unsigned int) time(NULL));

    fprintf(stderr, "Initializing spheres...");
    init_world(world);
    fprintf(stderr, "\r                            \r");
    fprintf(stderr, "Sfere inizializzate\n");

    double start = cpuSecond();

    t_camera cam = camera_new(ASPECT_RATIO, 
                              VIEWPORT_WIDTH, 
                              VERTICAL_FOV_DEGREES,
                              (t_point3) LOOK_FROM, 
                              (t_point3) LOOK_AT,
                              DEFOCUS_ANGLE,
                              FOCUS_DISTANCE);

    // Allocating the necessary memory on the heap for the image
    unsigned char *result_img = (unsigned char*) \
        malloc(cam.image_height*cam.image_width*sizeof(unsigned char) * 3);
    
    camera_render(&cam, world, NUMBER_OF_SPHERES, result_img);

    double end  = cpuSecond();
    fprintf(stderr, "Elapsed time: %.6fs\n", end - start);

    write_PPM_img_to_stdout(result_img, cam.image_width, cam.image_height);
    free(result_img);
}
