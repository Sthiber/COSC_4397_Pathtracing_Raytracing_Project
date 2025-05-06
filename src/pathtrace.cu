#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <fstream>           // Group3 Mod - For metrics logging
#include <chrono>            // Group3 Mod - For timing
#include <vector>            // Group3 Mod - For storing frames

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// ─────────── Timing with CUDA events ───────────
cudaEvent_t startKernel, stopKernel;
float totalKernelTime = 0.0f;

#define ERRORCHECK 1
#define MAX_MATERIALS 64


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

// Group3 Mod - PSNR reference frame storage
static bool firstFrame = true;
static std::vector<glm::vec3> referenceFrame;

// Group3 Mod - Performance metrics structure
struct PerformanceMetrics {
    float totalRenderTime = 0.0f;          // Total time over all iterations
    float avgIterationTime = 0.0f;         // Running average time per iteration (ms)
    float samplesPerSecond = 0.0f;         // Rays per second
    size_t gpuMemoryUsed = 0;              // Current GPU memory usage
    float lastPSNR = 0.0f;                 // PSNR of last iteration
    int iterationsToClean = -1;            // First iteration with infinite PSNR

    std::chrono::high_resolution_clock::time_point startTime;
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    void end(int iter, int pixelcount) {
        auto endTime = std::chrono::high_resolution_clock::now();
        float secs = std::chrono::duration<float>(endTime - startTime).count();
        totalRenderTime += secs;  // accumulate instead of overwrite
        avgIterationTime = (totalRenderTime / iter) * 1000.0f; // ms
        samplesPerSecond = (pixelcount * iter) / totalRenderTime;
    }
    
};
static PerformanceMetrics metrics;

// Group3 Mod - Query GPU memory usage
void updateGpuMemory() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    metrics.gpuMemoryUsed = totalMem - freeMem;
}

// Group3 Mod - Compute PSNR against first frame
float computePSNR(const std::vector<glm::vec3>& currentRaw, int iter) {
    std::vector<glm::vec3> current = currentRaw;
    for (auto& c : current) {
        c /= static_cast<float>(iter);  // average pixel values
    }

    if (firstFrame && iter == 10) {
        referenceFrame = current;
        firstFrame = false;
        return FLT_MAX;
    }

    if (firstFrame) {
        return FLT_MAX;  // skip until reference is set
    }

    double mse = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
        glm::vec3 diff = current[i] - referenceFrame[i];
        mse += glm::dot(diff, diff);
    }
    mse /= (current.size() * 3.0);

    if (mse <= 1e-12) return FLT_MAX;
    return 10.0f * log10f(1.0f / static_cast<float>(mse));
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / float(iter);
        // Group3 Mod - apply gamma correction
        pix = glm::pow(pix, glm::vec3(1.0f / 2.2f));

        glm::ivec3 color;
        color.x = glm::clamp(int(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp(int(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp(int(pix.z * 255.0f), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for stream-compaction, material-sorting, BVH, etc

void InitDataContainer(GuiDataContainer* imGuiData) {
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    // TODO: initialize stream-compaction buffers, material-sorting arrays, BVH structures...

    updateGpuMemory();  // Group3 Mod - record initial GPU memory usage
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: free any additional device memory (compaction, sorting, BVH)

    checkCUDAError("pathtraceFree");
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - float(cam.resolution.x) * 0.5f)
            - cam.up    * cam.pixelLength.y * ((float)y - float(cam.resolution.y) * 0.5f));

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections
) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths) {
        PathSegment pathSegment = pathSegments[path_index];
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        glm::vec3 tmp_intersect, tmp_normal;
        bool outside = true;
        for (int i = 0; i < geoms_size; i++) {
            Geom& geom = geoms[i];
            float t;
            if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            if (t > 0.0f && t < t_min) {
                t_min = t;
                hit_geom_index = i;
                intersections[path_index].point = tmp_intersect;
                intersections[path_index].surfaceNormal = tmp_normal;
            }
        }
        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
        } else {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        }
    }
}

__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int materialCount)
{
    extern __shared__ Material sharedMaterials[]; // Shared memory block

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 of each block loads materials into shared memory
    for (int i = threadIdx.x; i < MAX_MATERIALS && i < materialCount; i += blockDim.x) {
        sharedMaterials[i] = materials[i];
    }
    
    __syncthreads(); // Ensure all threads wait until materials are loaded

    if (idx < num_paths) {
        const ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& segment = pathSegments[idx];

        if (intersection.t > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = sharedMaterials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
                segment.color *= materialColor * material.emittance;
            } else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                segment.color *= (materialColor * lightTerm) * 0.3f 
                               + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                segment.color *= u01(rng);
            }
        } else {
            segment.color = glm::vec3(0.0f);
        }
    }
}


__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths) {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

void pathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y
    );
    const int blockSize1d = 128;

    // ────────────── Timing Variables ──────────────
    float rayGenTime = 0.0f;
    float intersectTime = 0.0f;
    float shadeTime = 0.0f;
    float gatherTime = 0.0f;
    float totalKernelTime = 0.0f;

    metrics.start();  // Start iteration timer

    // ────────────── Ray Generation ──────────────
    cudaEventRecord(startKernel);
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&rayGenTime, startKernel, stopKernel);
    totalKernelTime += rayGenTime;
    checkCUDAError("generate camera ray");

    // ────────────── Intersection + Shading ──────────────
    int depth = 0;
    int num_paths = pixelcount;
    bool iterationComplete = false;
    while (!iterationComplete) {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing((num_paths + blockSize1d - 1) / blockSize1d);

        // Intersection
        cudaEventRecord(startKernel);
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections
        );
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);
        cudaEventElapsedTime(&intersectTime, startKernel, stopKernel);
        totalKernelTime += intersectTime;
        checkCUDAError("trace one bounce");
        depth++;

        // Shading
        cudaEventRecord(startKernel);
        int materialCount = static_cast<int>(hst_scene->materials.size());
        size_t sharedMemBytes = materialCount * sizeof(Material);
        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d, sharedMemBytes>>>(
            iter, num_paths, dev_intersections, dev_paths, dev_materials, materialCount);
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);
        cudaEventElapsedTime(&shadeTime, startKernel, stopKernel);
        totalKernelTime += shadeTime;
        checkCUDAError("shade");

        iterationComplete = true;  // TODO: replace with stream compaction condition
        if (guiData != NULL) guiData->TracedDepth = depth;
    }

    // ────────────── Final Gather ──────────────
    dim3 numBlocksPixels((pixelcount + blockSize1d - 1) / blockSize1d);
    cudaEventRecord(startKernel);
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&gatherTime, startKernel, stopKernel);
    totalKernelTime += gatherTime;
    checkCUDAError("finalGather");

    // ────────────── Display ──────────────
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("sendImageToPBO");

    // ────────────── PSNR Calculation ──────────────
    std::vector<glm::vec3> current(pixelcount);
    cudaMemcpy(current.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    metrics.end(iter, pixelcount);           // Stop iteration timer
    updateGpuMemory();                       // Update GPU mem usage
    float psnr = computePSNR(current, iter); // Compute PSNR
    metrics.lastPSNR = psnr;
    if (psnr > 35.0f && metrics.iterationsToClean < 0) {
        metrics.iterationsToClean = iter;
    }

    // ────────────── Summary Printout ──────────────
    printf("\n====== PERFORMANCE METRICS SUMMARY ======\n");
    printf("Total render time: %.2f seconds\n", metrics.totalRenderTime);
    printf("Avg iteration time: %.2f ms\n", metrics.avgIterationTime);
    printf("Samples per second: %.2f million rays/s\n", metrics.samplesPerSecond / 1e6f);
    printf("GPU memory used: %.2f MB\n", metrics.gpuMemoryUsed / float(1 << 20));
    if (psnr == FLT_MAX) printf("PSNR: Inf dB\n");
    else printf("PSNR: %.2f dB\n", psnr);
    if (metrics.iterationsToClean > 0)
        printf("Iterations to clean: %d\n", metrics.iterationsToClean);
    printf("Total kernel time: %.2f ms\n", totalKernelTime);
    printf("  - Ray generation:   %.2f ms\n", rayGenTime);
    printf("  - Intersection:     %.2f ms\n", intersectTime);
    printf("  - Shading:          %.2f ms\n", shadeTime);
    printf("  - Final gather:     %.2f ms\n", gatherTime);
    printf("=========================================\n");

    // Store final image
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}

