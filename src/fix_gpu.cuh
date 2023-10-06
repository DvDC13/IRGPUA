#pragma once

#include "image.hh"
#include "pipeline.hh"
#include "deviceArray.cuh"
#include "kernels.cuh"

void fix_image_gpu(Image& to_fix);

int main_gpu([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline);