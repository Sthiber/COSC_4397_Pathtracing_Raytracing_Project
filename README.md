CUDA Pathtracing and Raytracing Project
================

**University of Houston, COSC 4397: Parallel Computations of GPU, Panruo Wu, Spring 2025, Final Project**  

**Group 3: Leo Nguyen and Sthiber Guevara**

## GPU and Rendering Metrics Explanation
| **Metric** | **What It Measures** | **Why It Matters** |
|------------|-----------------------|---------------------|
| **Total render time** | Time to render the full image across all iterations. | Helps compare full performance between different optimization versions. |
| **Average iteration time** | Average time (in ms) per rendering iteration. | Lets you estimate how scalable or efficient each iteration is. |
| **Samples per second** | Number of rays traced per second. | Key throughput metric — higher = faster rendering. |
| **GPU memory used** | Total GPU memory allocated (in MB). | Important for checking memory efficiency and avoiding overuse. |
| **PSNR (Peak Signal-to-Noise Ratio)** | Measures image quality by comparing current frame to the reference. | Higher PSNR = less noise. Good for confirming visual quality improvements. |
| **Iterations to clean image** | Number of iterations needed to reach high image quality (based on PSNR). | Useful to evaluate how quickly noise is reduced. |
| **Total kernel time** | Combined time (in ms) spent running all GPU kernels. | Helps identify overall GPU computation cost. |
| **Ray generation time** | Time spent creating rays from the camera. | Optimizing this helps early-stage performance. |
| **Intersection time** | Time spent checking ray-object intersections. | Most affected by acceleration structures like BVH. |
| **Shading time** | Time used for lighting/material calculations. | Useful for evaluating BRDF or lighting changes. |
| **Final gather time** | Time used to accumulate color results into the image buffer. | Helps confirm that pixel accumulation is efficient. |


## Naive Approach
The baseline for our project is the original implementation from the repository we forked. This represents the unoptimized, naive version of the path tracer. After running the Cornell Box scene using this baseline, we obtained the following performance metrics:






