#include "fix_thrust.cuh"

// TO REMOVE
#include "fix_gpu.cuh"
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

struct Predicate
{
    __host__ __device__
    bool operator()(const int& x)
    {
        return x != -27;
    }
};

__device__ __constant__ int map_thrust[4] = {1, -5, 3, -8};

struct ApplyMapFunctor
{

    __host__ __device__
    void operator()(int &x) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        x += map_thrust[gid & 3];
    }
};

struct NonZeroPredicate
{
    __host__ __device__ bool operator()(int x) {
        return x != 0;
    }
};

struct ApplyMapTransformation
{
    thrust::device_ptr<int> histogram;
    int first_non_zero;
    int size;

    ApplyMapTransformation(thrust::device_ptr<int> _histogram, int _first_non_zero, int _size)
        : histogram(_histogram), first_non_zero(_first_non_zero), size(_size) {}

    __host__ __device__ int operator()(int x) const
    {
        float result = ((histogram[x] - first_non_zero) / (float)(size - first_non_zero)) * 255.0f;
        return std::roundf(result);
    }
};

struct CompareToSort
{
    __host__ __device__ bool operator()(const Image::ToSort& a, const Image::ToSort& b)
    {
        return a.total < b.total;
    }
};

void fix_image_thrust(thrust::device_vector<int>& d_image, const int image_size, const int buffer_size)
{    
    // #1 Compact

    thrust::copy_if(d_image.begin(), d_image.end(), d_image.begin(), Predicate());

    // #2 Apply map to fix pixels

    thrust::for_each_n(d_image.begin(), image_size, ApplyMapFunctor());

    // #3 Histogram equalization

    // Histogram

    thrust::device_vector<int> d_histo(256, 0);
    
    thrust::device_vector<int> d_image_copy(d_image.begin(), d_image.begin() + image_size);
    thrust::sort(d_image_copy.begin(), d_image_copy.end());
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(d_image_copy.begin(), d_image_copy.end(), search_begin, search_begin + 256, d_histo.begin());
    thrust::adjacent_difference(d_histo.begin(), d_histo.end(), d_histo.begin());

    // Compute the inclusive sum scan of the histogram

    thrust::inclusive_scan(d_histo.begin(), d_histo.end(), d_histo.begin());

    // Find the first non-zero value in the cumulative histogram

    thrust::device_vector<int>::iterator first_non_zero = thrust::find_if(d_histo.begin(), d_histo.end(), NonZeroPredicate());
    int index = first_non_zero - d_histo.begin();
    int cdf_min = d_histo[index];

    // Apply the map transformation of the histogram equalization

    thrust::transform(d_image.begin(), d_image.end(), d_image.begin(), ApplyMapTransformation(d_histo.data(), cdf_min, image_size));

    // Free the thrust vectors
    d_histo.clear();
    d_image_copy.clear();
}

int main_thrust([[maybe_unused]] int argc, [[maybe_unused]] char** argv, Pipeline& pipeline)
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

        DeviceArray d_image(buffer_size, 0);

        // use thrust to copy the image
        thrust::device_vector<int> d_image_thrust(buffer_size, 0);
        thrust::copy(images[i].buffer, images[i].buffer + buffer_size, d_image_thrust.begin());

        fix_image_thrust(d_image_thrust, image_size, buffer_size);

        thrust::copy(d_image_thrust.begin(), d_image_thrust.end(), images[i].buffer);

        // -- All images are now fixed : compute stats (total then sort)

        // - First compute the total of each image

        // TODO : make it GPU compatible (aka faster)
        // You can use multiple CPU threads for your GPU version using openmp or not
        // Up to you :)

        d_image_thrust.resize(image_size);

        images[i].to_sort.total = thrust::reduce(d_image_thrust.begin(), d_image_thrust.end(), 0);

        // Free the thrust image
        d_image_thrust.clear();
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    thrust::host_vector<ToSort> to_sort(nb_images);
    int n = 0;
    thrust::generate(to_sort.begin(), to_sort.end(), [&n, &images]() {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    thrust::sort(to_sort.begin(), to_sort.end(), CompareToSort());

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