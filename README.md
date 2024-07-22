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

## Optimization solution
* Use multithreading to generate initial data (soon)
