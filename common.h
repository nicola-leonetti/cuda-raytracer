#ifndef COMMON_H
#define COMMON_H

#include "parameters.h"

#include <stdbool.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

// Returns a my_decimal in [0, 1)
my_decimal random_my_decimal() {
    return ((my_decimal) rand()) / ((my_decimal) RAND_MAX + 1);
}

// Returns a my_decimal in [min, max)
my_decimal random_my_decimal_in(my_decimal min, my_decimal max) {;
    return min + (max-min)*random_my_decimal();
}

my_decimal degrees_to_radians(my_decimal deg) {
    return (deg*PI) / 180.0;
}

#endif