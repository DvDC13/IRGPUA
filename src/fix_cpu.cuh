#pragma once

#include "image.hh"
#include "pipeline.hh"

void fix_image_cpu(Image& to_fix);

int main_cpu([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline);