#include "image.hh"
#include "pipeline.hh"
#include "parse.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Parse arguments
    ParsingArgs args = parse_args(argc, argv);

    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // -- Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // -- Check if there there is need to benchmark
    if (args.benchmark == false)
    {
        // -- Init pipeline object
        Pipeline pipeline(filepaths);
        
        // -- Pipeline execution
        if (args.version == ParsingArgs::Version::CPU)
            main_cpu(argc, argv, pipeline);
        else if (args.version == ParsingArgs::Version::GPU)
            main_gpu(argc, argv, pipeline);
        else
            return EXIT_FAILURE;
    }
    else
    {
        // -- Benchmarking

        std::cout << "Benchmarking..." << std::endl;

        // -- Init pipeline object

        Pipeline cpu_pipeline(filepaths);
        Pipeline gpu_pipeline(filepaths);

        // -- Launch cpu
        auto start_time_cpu = std::chrono::high_resolution_clock::now();
        main_cpu(argc, argv, cpu_pipeline);
        auto end_time_cpu = std::chrono::high_resolution_clock::now();

        // -- Launch gpu
        auto start_time_gpu = std::chrono::high_resolution_clock::now();
        main_gpu(argc, argv, gpu_pipeline);
        auto end_time_gpu = std::chrono::high_resolution_clock::now();

        // -- Compute duration
        auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_cpu - start_time_cpu);
        auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_gpu - start_time_gpu);

        // -- Print results
        std::cout << "CPU time: " << duration_cpu.count() << " ms" << std::endl;
        std::cout << "GPU time: " << duration_gpu.count() << " ms" << std::endl;
    }

    return EXIT_SUCCESS;
}
