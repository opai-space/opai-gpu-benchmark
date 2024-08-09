#include <cuda_runtime.h>  
#include <cuda_fp16.h>

#include "common.h"

typedef float BTYPE;

#define BLOCK_SIZE 1024
#define USEMEM 0.8 // Try to allocate 80% of memory


#define MAIN_CYCLE_TIMES    10       //Execute 10 tasks overall, Each task will perform a buffer shift
#define CALC_CYCLE_TIMES    1000     //In each task, "calcKernel" 1000 times in a loop


__global__ void calcKernel(int round, BTYPE* c, BTYPE* a, BTYPE* b, const ssize_t N)
{
    const ssize_t i = (ssize_t)blockIdx.x * (ssize_t)blockDim.x + (ssize_t)threadIdx.x;
    if (i < N)
    {
        for (int t = 0; t < round; t++) {
            //c[i] = __hadd(__hmul(__half(a[i]), __half(b[i])), __half(0.00f));
            c[i] = __hmul(__half(a[i]), __half(b[i]));
            b[i] = __hsub(__half(1.0), __half(c[i]));
        }
    }
}

__global__ void sumKernel(unsigned int* output, const BTYPE* input, const ssize_t sumBlockLength, const ssize_t N)
{
    const ssize_t i = (ssize_t)blockIdx.x * (ssize_t)blockDim.x + (ssize_t)threadIdx.x;
    if (i < N)
    {
        unsigned int value = (unsigned int)(input[i] * 1812433253);
        atomicAdd( &output[ i / sumBlockLength ], value);
    }
}

size_t totalMemory() {
    size_t freeMem, totalMem;
    checkError(cudaMemGetInfo(&freeMem, &totalMem));
    return totalMem;
}

size_t availMemory() {
    size_t freeMem, totalMem;
    checkError(cudaMemGetInfo(&freeMem, &totalMem));
    return freeMem;
}

int main(int argc, char** argv) {
    printf("-- OPAI GPU Benchmark -- \n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount <= 0) {
        throw std::string("No CUDA devices");
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("GPU Device %d:%s\n", i, devProp.name);
        printf("Total GlobalMem: %zu MB\n", devProp.totalGlobalMem / 1024 / 1024);
        printf("SM Count: %d\n", devProp.multiProcessorCount);
        printf("Shared Mem / Block: %zu KB\n", devProp.sharedMemPerBlock / 1024);
        printf("Max Threads / Block: %d\n", devProp.maxThreadsPerBlock);
        printf("Regs / Block: %d\n", devProp.regsPerBlock);
        printf("Max Threads / SM: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("======================================================\n");
    }

    //////////////////////////////////////////////////////
    // According to the GPU model, set the required video memory size
    
    const int MAX_DEVICES = 8;

    if (deviceCount<=0) {
        throw std::string("No GPU detected.");
    }
    if (deviceCount > MAX_DEVICES ) {
        throw std::string("Only supports up to 8 GPUs.");
    }

    int h_vmem_level = 0;
    ssize_t last_total_memory_gb = 0;
    std::string h_level_name;

    for (int i = 0; i < deviceCount; i++) {
        checkError(cudaSetDevice(i));

        ssize_t avail_memory = availMemory();
        ssize_t total_memory = totalMemory();
        if ( (double)avail_memory / (double)total_memory < USEMEM) {
            throw std::string("Insufficient GPU memory.");
        }
        ssize_t avail_memory_gb = avail_memory / (1024ul * 1024ul * 1024ul);
        ssize_t total_memory_gb = total_memory / (1024ul * 1024ul * 1024ul);

        if (last_total_memory_gb !=0 && last_total_memory_gb != total_memory_gb) {
            throw std::string("Requires the same GPU memory size to benchmark.");
        }

        last_total_memory_gb = total_memory_gb;

        if (avail_memory_gb >= 60) {        //80G level
            h_vmem_level = 60;
            h_level_name = "80GB";
        }
        else if (avail_memory_gb >= 30) {   //40/48G level
            h_vmem_level = 30;
            h_level_name = "40GB";
        }
        else if (avail_memory_gb >= 18) {   //24G level
            h_vmem_level = 18;
            h_level_name = "24GB";
        }
        else if (avail_memory_gb >= 13) {   //16G level
            h_vmem_level = 12;
            h_level_name = "16GB";
        }
        else if (avail_memory_gb >= 6) {    //8G level
            h_vmem_level = 6;
            h_level_name = "8GB";
        }
        else {
            throw std::string("GPU memory is too low to benchmark.");
        }
    }

    printf("> Switch to GPU level: %s\n", h_level_name.c_str());
        
    /////////////////////////////////////////////////////

    double tm = getTime();
    
    //Data buffer size allocated to each GPU (*3)
    const ssize_t s_gpuDataBytes = (ssize_t)h_vmem_level * 1024ul * 1024ul * 1024ul;
    const ssize_t s_gpuDataPieceSize = s_gpuDataBytes / 3;
    const ssize_t s_gpuStructSize = s_gpuDataPieceSize / sizeof(BTYPE);

    //Each computing unit block is divided into 256MB
    const ssize_t s_blockBytesSize = 256ul * 1024ul * 1024ul;
    const ssize_t s_blockStructSize = s_blockBytesSize / sizeof(BTYPE);

    //Each unit block will eventually get an unsiged integer result
    const ssize_t s_gpuBlockCount = s_gpuStructSize / s_blockStructSize;
    const ssize_t s_totalBlockResult = deviceCount * s_gpuBlockCount ;

    BTYPE* d_Cdata[MAX_DEVICES];
    BTYPE* d_Adata[MAX_DEVICES];
    BTYPE* d_Bdata[MAX_DEVICES];

    unsigned int* d_sum_data[MAX_DEVICES];

    printf("> Create Host Buffer: %zu MB... \n", 2 * s_blockBytesSize / 1024ul / 1024ul);
    BTYPE* A = (BTYPE*)malloc(s_blockBytesSize);
    BTYPE* B = (BTYPE*)malloc(s_blockBytesSize);


    //Set initialization seed, 64 bits
    __srand(10);
    //__srand((uint64_t)getTime());

    //Based on the initialization seed, expand the initialization data tables A and B
    for (int i = 0; i < deviceCount; i++) {
        printf("> Create Device[%d] Buffer: %zu GB... \n", i, s_gpuDataBytes / 1024ul / 1024ul / 1024ul);

        checkError(cudaSetDevice(i));

        checkError(cudaMalloc((void**)&d_Adata[i], s_gpuDataPieceSize), "A alloc");
        checkError(cudaMalloc((void**)&d_Bdata[i], s_gpuDataPieceSize), "B alloc");
        checkError(cudaMalloc((void**)&d_Cdata[i], s_gpuDataPieceSize), "C alloc");

        ssize_t seek_pos = 0;
        while (seek_pos < s_gpuDataPieceSize) {
            for (size_t t = 0; t < s_blockStructSize; t++) {
                A[t] = (BTYPE)(__rand() / 4294967295.0);
                B[t] = (BTYPE)(__rand() / 4294967295.0);
            }
            checkError(cudaMemcpy((byte*)d_Adata[i] + seek_pos, A, s_blockBytesSize, cudaMemcpyHostToDevice), "A Init");
            checkError(cudaMemcpy((byte*)d_Bdata[i] + seek_pos, B, s_blockBytesSize, cudaMemcpyHostToDevice), "B Init");

            seek_pos += s_blockBytesSize;
        }

        checkError(cudaMalloc((void**)&d_sum_data[i], s_gpuBlockCount * sizeof(unsigned int)), "sum alloc");
        checkError(cudaMemset(d_sum_data[i], 0 , s_gpuBlockCount * sizeof(unsigned int)), "sum memset");
    }

    free(A);
    free(B);

    /////////////////////////////////////////////////////////

    cudaError_t cudaStatus;

    printf("> Start compute... \n");
    
    for (int i = 0; i < deviceCount; i++) {
        printf("> Execute the task of Device[%d]... \n", i);

        checkError(cudaSetDevice(i));

        int block_size = BLOCK_SIZE;
        int grid_size = (int)(s_gpuStructSize / block_size);

        //printf("grid_size: %d  block_size:%d  gpuStructSize:%zu  s_gpuBlockCount:%zu\n", grid_size, block_size, s_gpuStructSize, s_gpuBlockCount);

        for (int iters = 0; iters < MAIN_CYCLE_TIMES; iters++) {
            calcKernel <<< grid_size, block_size >>> (CALC_CYCLE_TIMES, d_Cdata[i], d_Adata[i], d_Bdata[i], s_gpuStructSize);
           
            //Shift d_Adata
            for (int block = 0; block < s_gpuBlockCount - 1; block++) {
                cudaMemcpyAsync(
                    ((uint8_t*)d_Adata[i]) + (block+1) * s_blockBytesSize,
                    ((uint8_t*)d_Adata[i]) + (block) * s_blockBytesSize, 
                    s_blockBytesSize,
                    cudaMemcpyDeviceToDevice
                );
            }
        }

        sumKernel <<< grid_size, block_size >> > (d_sum_data[i], d_Cdata[i], s_blockStructSize, s_gpuStructSize);
    }

    // Check for any errors launching the kernel 
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iKernel!\n", cudaStatus);
        return 0;
    }

    /////////////////////////////////////////////////////////

    printf("> Get calculation outputs from GPU.\n");

    unsigned int* output = (unsigned int*)malloc(s_totalBlockResult * sizeof(unsigned int));

    ssize_t block_pos = 0;
    for (int i = 0; i < deviceCount; i++) {
        checkError(cudaSetDevice(i));
        checkError(cudaMemcpy((byte*)output + block_pos, d_sum_data[i], s_gpuBlockCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        block_pos += s_gpuBlockCount * sizeof(unsigned int);

        cudaFree(d_sum_data[i]);
        cudaFree(d_Adata[i]);
        cudaFree(d_Bdata[i]);
        cudaFree(d_Cdata[i]);
    }

    for (int t = 0; t < s_totalBlockResult; t++) {
        printf("[%2d] 0x%08X\n", t, output[t]);
    }

    free(output);

    tm = getTime() - tm;

    printf("\n\n");
    printf("Total blocks:            %d\n", (int)s_totalBlockResult);
    printf("Using total time:        %.3f seconds\n", (float)tm);
    printf("Using time per block:    %.3f seconds\n", (float)tm/s_totalBlockResult);

    return 0;
}
