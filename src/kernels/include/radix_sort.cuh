#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>

#include "deviceArray.cuh"
#include "scan.cuh"
#include "shift.cuh"

void radix_sort_gpu(int* arr, int size);