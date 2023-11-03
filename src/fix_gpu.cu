#include "fix_gpu.cuh"

void fix_image_gpu(DeviceArray& d_image, const int image_size, const int buffer_size)
{    
    int block_size = 1024;
    int grid_size = (buffer_size + block_size - 1) / (block_size * 4);

    // #1 Compact
    // Build predicate vector
    DeviceArray d_predicate(buffer_size, 0);

    build_predicate3<<<grid_size, block_size>>>(d_image.data_, d_predicate.data_, buffer_size);
    cudaXDeviceSynchronize();

    block_size = 1024;
    grid_size = (buffer_size + block_size - 1) / block_size;

    // Compute the exclusive sum of the predicate
    cuda::std::atomic<char>* d_blockStates = nullptr;
    cudaXMalloc((void**)&d_blockStates, grid_size * sizeof(cuda::std::atomic<char>));
    cudaXMemset(d_blockStates, 'X', grid_size * sizeof(cuda::std::atomic<char>));

    DeviceArray d_globalCounter(1, 0);
    DeviceArray d_blocksAggregate(grid_size, 0);

    decoupled_look_back<<<grid_size, block_size, sizeof(int)>>>(d_predicate.data_, d_blocksAggregate.data_, d_globalCounter.data_, d_blockStates, buffer_size);
    cudaXDeviceSynchronize();
    cudaCheckError();

    DeviceArray predicate_shifted(buffer_size, 0);

    shift_buffer<<<grid_size, block_size>>>(d_predicate.data_, predicate_shifted.data_, buffer_size);
    cudaXDeviceSynchronize();
    cudaCheckError();

    // Scatter to the corresponding addresses
    scatter_adresses<<<grid_size, block_size>>>(d_image.data_, predicate_shifted.data_, buffer_size);
    cudaXDeviceSynchronize();

    // #2 Apply map to fix pixels
    apply_map4<<<grid_size, block_size>>>(d_image.data_, image_size);
    cudaXDeviceSynchronize();

    // #3 Histogram equalization
    // Histogram
    DeviceArray d_histo(256, 0);

    compute_histogram2<<<grid_size, block_size>>>(d_image.data_, d_histo.data_, image_size);
    cudaXDeviceSynchronize();

    // Compute the inclusive sum scan of the histogram

    cudaXMemset(d_blockStates, 'X', grid_size * sizeof(cuda::std::atomic<char>));
    d_globalCounter.setTo(1, 0);
    d_blocksAggregate.setTo(grid_size, 0);

    decoupled_look_back<<<1, 256, sizeof(int)>>>(d_histo.data_, d_blocksAggregate.data_, d_globalCounter.data_, d_blockStates, 256);
    cudaXDeviceSynchronize();

    // Find the first non-zero value in the cumulative histogram
    DeviceArray d_predicate_zeros(256, 0);

    build_predicate_zeros1<<<1, 256>>>(d_histo.data_, d_predicate_zeros.data_, 256);
    cudaXDeviceSynchronize();

    cudaXMemset(d_blockStates, 'X', grid_size * sizeof(cuda::std::atomic<char>));
    d_globalCounter.setTo(1, 0);
    d_blocksAggregate.setTo(grid_size, 0);

    decoupled_look_back<<<1, 256, sizeof(int)>>>(d_predicate_zeros.data_, d_blocksAggregate.data_, d_globalCounter.data_, d_blockStates, 256);
    cudaXDeviceSynchronize();

    DeviceArray d_firstNonZero(1, 0);

    find_first_non_zero<<<1, 256>>>(d_histo.data_, d_predicate_zeros.data_, d_firstNonZero.data_, 256);
    cudaXDeviceSynchronize();

    // Apply the map transformation of the histogram equalization
    apply_map_transformation1<<<grid_size, block_size>>>(d_image.data_, d_histo.data_, d_firstNonZero.data_, image_size);
    cudaXDeviceSynchronize();
}

uint64_t compute_reduce_gpu(DeviceArray &d_buffer, int image_size)
{
    int block_size = 1024;
    int num_blocks = (image_size + block_size - 1) / block_size;

    DeviceArray d_total(1, 0);

    reduce3<<<num_blocks, block_size, block_size * sizeof(int)>>>(d_buffer.data_, d_total.data_, image_size);
    cudaXDeviceSynchronize();

    int total = 0;
    d_total.copyToHost(&total, 1);

    return total;
}

int main_gpu([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline)
{
    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        const int image_size = (int)images[i].width * (int)images[i].height;
        const int buffer_size = images[i].size();

        DeviceArray d_image(buffer_size, 0);
        d_image.copyFromHost(images[i].buffer, buffer_size);

        fix_image_gpu(d_image, image_size, buffer_size);

        d_image.copyToHost(images[i].buffer, buffer_size);

        // -- All images are now fixed : compute stats (total then sort)

        // - First compute the total of each image

        // TODO : make it GPU compatible (aka faster)
        // You can use multiple CPU threads for your GPU version using openmp or not
        // Up to you :)

        images[i].to_sort.total = compute_reduce_gpu(d_image, image_size);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<int> to_sort(nb_images);

    for (int i = 0; i < nb_images; ++i)
        to_sort[i] = images[i].to_sort.total;

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    radix_sort_gpu(to_sort.data(), nb_images);

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    pipeline.set_images(images);

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        cudaXFreeHost(images[i].buffer);

    to_sort.clear();

    return EXIT_SUCCESS;
}