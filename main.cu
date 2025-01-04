#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

#define NUMBER_OF_SPHERES 489

__host__ inline void print_device_info(int id) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    fprintf(stderr, "\n-----------------Device properties-----------------\n");
    fprintf(stderr, "GPU name: %s \n", properties.name);
    fprintf(
        stderr, 
        "Compute capability: %d.%d\n", 
        properties.major,
        properties.minor
    );
    fprintf(
        stderr, 
        "Max number of threads per block: %d \n", 
        properties.maxThreadsPerBlock
    );
    fprintf(
        stderr, 
        "Max size of a block of threads: (%d, %d, %d) \n", 
        properties.maxThreadsDim[0],
        properties.maxThreadsDim[1],
        properties.maxThreadsDim[2]
    );
    fprintf(
        stderr, 
        "Max size of grid of blocks: (%d, %d, %d) \n",
        properties.maxGridSize[0],
        properties.maxGridSize[1],
        properties.maxGridSize[2]
    );
    fprintf(stderr, "---------------------------------------------------\n\n");
}

__host__ double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9);
}

// TODO Vedere se riesco a fare in modo che inizializzi tutto su device
// magari parallelizzando
__host__ void init_world(t_sphere world[]) {
    fprintf(stderr, "Initializing spheres...");

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
            my_decimal choose_mat = h_random_my_decimal();
            t_point3 center = point3_new(a + 0.9 * h_random_my_decimal(), 0.2, b + 0.9 * h_random_my_decimal());
            t_material sphere_material;

            if (choose_mat < 0.8) {
                // Lambertian (diffuse)
                sphere_material = new_lambertian(
                    color_new(h_random_my_decimal()*h_random_my_decimal(), 
                              h_random_my_decimal()*h_random_my_decimal(), 
                              h_random_my_decimal()*h_random_my_decimal()
                            ));
                world[index++] = sphere_new(center, 0.2, sphere_material);
            } 
            else if (choose_mat < 0.95) {
                // Metal
                t_color color = color_new(h_random_my_decimal_in(0.5, 1), h_random_my_decimal_in(0.5, 1), h_random_my_decimal_in(0.5, 1));
                sphere_material = new_metal(color, h_random_my_decimal_in(0, 0.5));
                world[index++] = sphere_new(center, 0.2, sphere_material);
            } 
            else {
                // Dielectric (glass)
                world[index++] = sphere_new(center, 0.2, new_dielectric(1.5));
            }
            
        }
    }

    fprintf(stderr, "\r                            \r");
    fprintf(stderr, "Spheres initilized\n");
}

__host__ void write_PPM_img_to_stdout(unsigned char *img, int width, int height) {
    // PPM header
    printf("P3\n%d %d\n255\n", width, height);
    for (int pixel = 0; pixel < width*height*3; pixel+=3) {
        printf("%d %d %d\n", img[pixel], img[pixel+1], img[pixel+2]);
    }
}

__host__ int main() {

    print_device_info(0);

    srand((unsigned int) RNG_SEED);

    // Initialize spheres on host
    int world_size = NUMBER_OF_SPHERES*sizeof(t_sphere);
    t_sphere *h_world = (t_sphere*) malloc(world_size); 
    init_world(h_world);

    // Copy spheres host -> device
    t_sphere *d_world;
    CHECK(cudaMalloc((void**)&d_world, world_size));
    CHECK(cudaMemcpy(d_world, h_world, world_size, cudaMemcpyHostToDevice));

    // TODO Vedere se possibile creare oggetto direttamente su device per
    // ottimizzare
    // Create camera object and move it on device
    t_camera cam = camera_new(ASPECT_RATIO, VIEWPORT_WIDTH, 
                              VERTICAL_FOV_DEGREES, (t_point3) LOOK_FROM, 
                              (t_point3) LOOK_AT, DEFOCUS_ANGLE, 
                              FOCUS_DISTANCE);
    t_camera *h_cam = &cam;
    t_camera *d_cam;
    CHECK(cudaMalloc((void**)&d_cam, sizeof(cam)));
    CHECK(cudaMemcpy(d_cam, h_cam, sizeof(cam), cudaMemcpyHostToDevice));

    // Allocate on device one RNG state for each pixel
    int number_of_pixels = cam.image_width*cam.image_height;
    curandState *d_random_states;
    CHECK(cudaMalloc(
        (void**) &d_random_states, 
        number_of_pixels*sizeof(curandState)
    ));

    // Allocate space for the image on host and device
    long img_size = cam.image_height*cam.image_width*sizeof(unsigned char)*3;
    unsigned char *h_result_img = (unsigned char*) malloc(img_size);
    unsigned char *d_result_img;
    CHECK(cudaMalloc((void**)&d_result_img, img_size));
    CHECK(cudaMemcpy(d_result_img, h_result_img, img_size, cudaMemcpyHostToDevice));

    fprintf(
        stderr,
        "Launching render kernel with 2D grid shape (%u, %u)\n", 
        (cam.image_width + block.x - 1) / block.x, 
        (cam.image_height + block.y - 1) / block.y
    );
        
    double start = cpuSecond();

    dim3 grid(
        (cam.image_width + block.x - 1) / block.x, 
        (cam.image_height + block.y - 1) / block.y
    );
    camera_render<<<grid, block>>>(
        d_cam, 
        d_world, 
        NUMBER_OF_SPHERES, 
        d_result_img, 
        d_random_states
    );
    cudaDeviceSynchronize();

    double end = cpuSecond();
    fprintf(stderr, "Computation time: %.6fs\n", end - start);

    // Copy the result image device -> host -> .ppm file 
    fprintf(
        stderr, 
        "Image size: %dx%d, %d channels, %ld bytes\n", 
        cam.image_height, 
        cam.image_width,
        3,
        img_size
    );
    CHECK(cudaMemcpy(h_result_img, d_result_img, img_size, cudaMemcpyDeviceToHost));
    write_PPM_img_to_stdout(h_result_img, cam.image_width, cam.image_height);

    CHECK(cudaFree(d_world));
    CHECK(cudaFree(d_cam));
    CHECK(cudaFree(d_result_img));
    free(h_world);
    free(h_result_img);
}
