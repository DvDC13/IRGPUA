#include "fix_gpu.cuh"

void fix_image_gpu(DeviceArray<int>& d_image, const int image_size, const int buffer_size)
{    
    int block_size = 1024;
    int grid_size = (buffer_size + block_size - 1) / block_size;

    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    // #1 Compact
    // Build predicate vector
    DeviceArray<int> d_predicate(buffer_size, 0);

    build_predicate<<<dimGrid, dimBlock>>>(d_image.data_, d_predicate.data_, buffer_size);
    cudaXDeviceSynchronize();

    // Compute the exclusive sum of the predicate
    DeviceArray<int> d_blockStates(grid_size, 0);
    DeviceArray<int> d_blocksP(grid_size, 0);
    DeviceArray<int> d_blocksA(grid_size, 0);
    DeviceArray<int> d_globalCounter(1, 0);

    decoupled_lookback_scan<<<dimGrid, dimBlock, sizeof(int)>>>(d_predicate.data_, d_globalCounter.data_, d_blocksA.data_, d_blocksP.data_, d_blockStates.data_, buffer_size);
    cudaXDeviceSynchronize();
    cudaCheckError();

    DeviceArray<int> predicate_shifted(buffer_size, 0);

    shift_buffer<<<dimGrid, dimBlock>>>(d_predicate.data_, predicate_shifted.data_, buffer_size);
    cudaXDeviceSynchronize();
    cudaCheckError();

    // Scatter to the corresponding addresses
    scatter_adresses<<<dimGrid, dimBlock>>>(d_image.data_, predicate_shifted.data_, buffer_size);
    cudaXDeviceSynchronize();

    // #2 Apply map to fix pixels
    apply_map<<<dimGrid, dimBlock>>>(d_image.data_, image_size);
    cudaXDeviceSynchronize();

    // #3 Histogram equalization
    // Histogram
    DeviceArray<int> d_histo(256, 0);

    compute_histogram<<<dimGrid, dimBlock>>>(d_image.data_, d_histo.data_, image_size);
    cudaXDeviceSynchronize();

    // Compute the inclusive sum scan of the histogram
    d_blockStates.setTo(grid_size, 0);
    d_blocksP.setTo(grid_size, 0);
    d_blocksA.setTo(grid_size, 0);
    d_globalCounter.setTo(1, 0);

    decoupled_lookback_scan<<<1, 256, sizeof(int)>>>(d_histo.data_, d_globalCounter.data_, d_blocksA.data_, d_blocksP.data_, d_blockStates.data_, 256);
    cudaXDeviceSynchronize();

    // Find the first non-zero value in the cumulative histogram
    DeviceArray<int> d_predicate_zeros(256, 0);

    create_predicate_zeros<<<1, 256>>>(d_histo.data_, d_predicate_zeros.data_, 256);
    cudaXDeviceSynchronize();

    d_blockStates.setTo(grid_size, 0);
    d_blocksP.setTo(grid_size, 0);
    d_blocksA.setTo(grid_size, 0);
    d_globalCounter.setTo(1, 0);

    decoupled_lookback_scan<<<1, 256, sizeof(int)>>>(d_predicate_zeros.data_, d_globalCounter.data_, d_blocksA.data_, d_blocksP.data_, d_blockStates.data_, 256);
    cudaXDeviceSynchronize();

    DeviceArray<int> d_firstNonZero(1, 0);

    find_first_non_zero<<<1, 256>>>(d_histo.data_, d_predicate_zeros.data_, d_firstNonZero.data_, 256);
    cudaXDeviceSynchronize();

    // Apply the map transformation of the histogram equalization
    apply_map_transformation<<<dimGrid, dimBlock>>>(d_image.data_, d_histo.data_, d_firstNonZero.data_, image_size);
    cudaXDeviceSynchronize();
}

uint64_t compute_reduce(DeviceArray<int> &d_buffer, int image_size)
{
    int block_size = 1024;
    int num_blocks = (image_size + block_size - 1) / block_size;

    DeviceArray<int> d_total(1, 0);

    kernel_reduce<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_buffer.data_, d_total.data_, image_size);
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

    #pragma omp parallel for
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

        DeviceArray<int> d_image(buffer_size, 0);
        d_image.copyFromHost(images[i].buffer, buffer_size);

        fix_image_gpu(d_image, image_size, buffer_size);

        d_image.copyToHost(images[i].buffer, buffer_size);

        // -- All images are now fixed : compute stats (total then sort)

        // - First compute the total of each image

        // TODO : make it GPU compatible (aka faster)
        // You can use multiple CPU threads for your GPU version using openmp or not
        // Up to you :)
        
        d_image.setTo(image_size, buffer_size - image_size, 0);

        images[i].to_sort.total = compute_reduce(d_image, image_size);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

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
        free(images[i].buffer);

    return 0;
}