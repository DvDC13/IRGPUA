#pragma once

#include "image.hh"
#include "pipeline.hh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

void fix_image_thrust(thrust::device_vector<int>& d_image, const int image_size);

int main_thrust([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline);