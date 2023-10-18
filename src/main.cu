#include "image.hh"
#include "pipeline.hh"
#include "parse.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "fix_thrust.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

void compare_versions(Pipeline& pipeline_cpu, Pipeline& pipeline_gpu)
{
    // std::cout << std::endl;
    // for (int i = 0; i < pipeline_cpu.images.size(); ++i)
    // {
    //     std::cout << "Index: " << i << std::endl;
    //     std::cout << "CPU Buffer size: " << pipeline_cpu.images[i].size() << " GPU Buffer size: " << pipeline_gpu.images[i].size() << std::endl;
    //     std::cout << "CPU Total: " << pipeline_cpu.images[i].to_sort.total << " GPU Total: " << pipeline_gpu.images[i].to_sort.total << std::endl;
    //     std::cout << std::endl;
    // }

    // -- Compare results
    for (long unsigned int i = 0; i < pipeline_cpu.images.size(); ++i)
    {
        auto& cpu_images = pipeline_cpu.images;
        auto& gpu_images = pipeline_gpu.images;

        if (cpu_images[i].size() != gpu_images[i].size())
        {
            std::cout << "Index: " << i << std::endl;
            std::cout << "Buffer size CPU: " << cpu_images[i].size() << std::endl;
            std::cout << "Buffer size GPU: " << gpu_images[i].size() << std::endl;
            std::cout << "Error: images are not equal" << std::endl;
            return;
        }

        if (cpu_images[i].to_sort.total != gpu_images[i].to_sort.total)
        {
            std::cout << "Index: " << i << std::endl;
            std::cout << "Total CPU: " << cpu_images[i].to_sort.total << std::endl;
            std::cout << "Total GPU: " << gpu_images[i].to_sort.total << std::endl;
            std::cout << "Error: images are not equal" << std::endl;
            return;
        }
    }

    std::cout << "Success: images are equal" << std::endl;
}

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
        else if (args.version == ParsingArgs::Version::THRUST)
            main_thrust(argc, argv, pipeline);
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
        Pipeline thrust_pipeline(filepaths);

        // -- Launch cpu
        auto start_time_cpu = std::chrono::high_resolution_clock::now();
        main_cpu(argc, argv, cpu_pipeline);
        auto end_time_cpu = std::chrono::high_resolution_clock::now();

        // -- Launch gpu
        auto start_time_gpu = std::chrono::high_resolution_clock::now();
        main_gpu(argc, argv, gpu_pipeline);
        auto end_time_gpu = std::chrono::high_resolution_clock::now();

        // -- Launch thrust
        auto start_time_thrust = std::chrono::high_resolution_clock::now();
        main_thrust(argc, argv, thrust_pipeline);
        auto end_time_thrust = std::chrono::high_resolution_clock::now();

        // -- Compute duration
        auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_cpu - start_time_cpu);
        auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_gpu - start_time_gpu);
        auto duration_thrust = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_thrust - start_time_thrust);

        // -- Print results
        std::cout << "CPU time: " << duration_cpu.count() << " ms" << std::endl;
        std::cout << "GPU time: " << duration_gpu.count() << " ms" << std::endl;
        std::cout << "Thrust time: " << duration_thrust.count() << " ms" << std::endl;

        // -- Compare results CPU & GPU
        std::cout << "Comparing results CPU & GPU" << std::endl;
        compare_versions(cpu_pipeline, gpu_pipeline);

        // -- Compare results CPU & Thrust
        std::cout << "Comparing results CPU & Thrust" << std::endl;
        compare_versions(cpu_pipeline, thrust_pipeline);
    }

    return EXIT_SUCCESS;
}
