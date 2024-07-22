# opai-gpu-benchmark

## GPU benchmark based on FP16 and video memory
We benchmarked a method similar to training LLM models on real FP16 achievable on various GPUs (including single GPU and multi-GPU). It can help you estimate the actual performance of GPUs in AI applications. During the test, as much video memory as possible was used to simulate the transfer between layers.

Generates a random data set using a 64-bit seed value and fills all video memory. Each memory block (set to 256MB) will be calculated with a 32-bit HASH value. The indicator is the average calculation time for each block.

Actual performance depends on many factors, including your hardware, CUDA version, etc. We list the numbers obtained on PCs and enterprise-level GPUs:

## Benchmarking Summary

|Model|VRam Size|PCIE|Number of blocks|Total time(secs)|Average block time|
|---|---|---|---|---|---|
|RTX3070Laptop *1|8G|x8|8|43.784|5.473|
|A10 *2|24G*2|x16|48|116.496|2.427|
|A10 *8|24G*8|x16|192|238.092|1.240|

## Principles
Complete the computational challenge based on uint64 seeds. The same seed should get the same block result array.

According to the video memory of the graphics card, it will be divided into 4 levels:
8GB / 16GB / 24GB / 40GB / 80GB

Computational tasks:
1. The task starts with a uint64 (64bits) seed.
2. Use the Linear congruential generator to generate continuous float numbers and fill them into the video memory cache DataA and DataB, and divide them into 256MB per block.
3. Use FP16 to calculate DataC using DataA and DataB. The intermediate process of DataC is inverted and overwritten into DataB
4. Every 1000 times of step 3, the block of DataA is translated.
5. Execute step 4 10 times in total.
6. Quantize all floats of DataC in each block into a uint32, which is the result of the block.

Verification tasks:
1. Check whether the average block time is within the specified range - determine whether the FP16 performance and video memory bandwidth meet the standards
2. Check whether a sufficient number of block results can be completed and each result is correct.

## Optimization solution
* Use multithreading to generate initial data (soon)
