#pragma once

#include <stdio.h>
#include <stdlib.h>   
#include <stdint.h>   
#include <math.h>

#include <string.h>
#include <string>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

static uint64_t next_random = 1;

inline uint32_t __rand(void)
{
    next_random = next_random * 6364136223846793005ul + 1442695040888963407ul;
    return (uint32_t)(next_random >> 32) % 0xffffffff;
}

inline void __srand(uint64_t seed)
{
    next_random = seed;
}

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char* err = cudaGetErrorString((cudaError_t)rCode);

        std::string errstr =
            (desc == "" ? std::string("Error (")
                : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err;

        printf("# %s\n", errstr.c_str());

        throw std::runtime_error(errstr);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}


/*
void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
        const char* err = "";// cublasGetStatusString(rCode);
#else
        const char* err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}
*/

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)


#ifdef WIN32
#include <windows.h>
typedef __int64 ssize_t;

double getTime() {

    return (double)GetTickCount() / 1000.0;
}

#else
typedef unsigned char byte;

#include <sys/time.h>
#include <sys/types.h>

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

#endif // WIN32

