#pragma once

#include "image.hh"
#include "pipeline.hh"
#include "deviceArray.cuh"

#include "build_predicate.cuh"
#include "reduce.cuh"
#include "scan.cuh"
#include "shift.cuh"
#include "histogram.cuh"
#include "scatter_adresses.cuh"
#include "apply_map.cuh"
#include "find_first_non_zero.cuh"
#include "apply_map_transformation.cuh"

void fix_image_gpu(Image& to_fix, const int image_size, const int buffer_size);

int main_gpu([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline);