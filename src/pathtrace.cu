#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm> // For std::sort

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// Group3 Mod - Configuration options 
#define ERRORCHECK 1
#define STREAM_COMPACTION 1    // Enable/disable stream compaction for terminated rays
#define MATERIAL_SORTING 1     // Enable/disable material sorting to reduce warp divergence
#define USE_ENHANCED_BVH 1     // Use enhanced BVH with surface area heuristic
#define MAX_SAH_BUCKETS 8      // Number of buckets for SAH split finding
#define MAX_BVH_DEPTH 16       // Maximum depth of BVH tree
#define MAX_BVH_PRIMS_PER_NODE 4 // Maximum primitives per leaf node
#define MAX_ITERATIONS 10      // Stop rendering after 10 iterations

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

// Group3 Mod - Device-compatible swap function (fixes std::swap device compiler error)
__device__ __host__ 
inline void deviceSwap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// Group3 Mod - Enhanced performance metrics tracking structure
struct KernelTimingBreakdown {
    float generateRayTime = 0.0f;
    float intersectionTime = 0.0f;
    float shadingTime = 0.0f;
    float streamCompactionTime = 0.0f;
    float finalGatherTime = 0.0f;
    float materialSortingTime = 0.0f;
    float miscTime = 0.0f;
};

struct PerformanceMetrics {
    // Basic timing metrics
    float totalKernelTime = 0.0f;
    float memoryTransferToDeviceTime = 0.0f;
    float memoryTransferFromDeviceTime = 0.0f;
    size_t bytesTransferredToDevice = 0;
    size_t bytesTransferredFromDevice = 0;
    float gpuUtilization = 0.0f;
    int kernelCalls = 0;
    bool metricsRecorded = false;
    
    // Additional metrics from the picture
    float totalRenderTime = 0.0f;        // 1. Total Render Time
    float timePerIteration = 0.0f;       // 2. Time per Iteration
    KernelTimingBreakdown kernelBreakdown; // 3. Kernel Timing Breakdown
    float samplesPerSecond = 0.0f;       // 4. Samples per Second
    size_t totalGpuMemoryUsed = 0;       // 5. Memory Usage
    size_t sharedMemoryUsed = 0;         // Part of Memory Usage
    float psnr = 0.0f;                   // 6. PSNR
    int iterationsForCleanImage = 0;     // 7. Number of Iterations for Clean Image
    
    // Timer for calculating total render time
    std::chrono::high_resolution_clock::time_point startTime;
    
    // Initialize metrics and start the timer
    void startMeasurement() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    // Calculate time per iteration
    void updateIterationTime(float iterationTime) {
        if (timePerIteration == 0.0f) {
            timePerIteration = iterationTime;
        } else {
            timePerIteration = (timePerIteration + iterationTime) / 2.0f;  // Running average
        }
    }
    
    // Group3 Mod - Print metrics once
    void logMetrics(int iter, int frame, int pixelCount) {
        if (metricsRecorded) {
            return; // Only log once
        }
        
        // Calculate total render time
        auto endTime = std::chrono::high_resolution_clock::now();
        float renderMilliseconds = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        totalRenderTime = renderMilliseconds / 1000.0f;  // Convert to seconds
        
        // Calculate samples per second (rays processed per second)
        samplesPerSecond = (float)(pixelCount * iter) / totalRenderTime;
        
        // Calculate metrics
        float totalTime = totalKernelTime + memoryTransferToDeviceTime + memoryTransferFromDeviceTime;
        gpuUtilization = (totalTime > 0) ? totalKernelTime / totalTime * 100.0f : 0.0f;
        
        // Current timestamp and username from user
        const char* timestamp = "2025-05-06 19:48:47";  // Updated timestamp
        const char* username = "leo2971998";            // Username
        
        // Create log file if it doesn't exist
        std::ofstream logFile("cuda_performance.log", std::ios::app);
        if (logFile.is_open()) {
            logFile << "====== PERFORMANCE METRICS (Frame: " << frame << ", Iteration: " << iter << ") ======\n";
            logFile << "Timestamp: " << timestamp << "\n";
            logFile << "User: " << username << "\n\n";
            
            // Basic metrics
            logFile << "--- Basic GPU Metrics ---\n";
            logFile << "Total kernel execution time: " << totalKernelTime << " ms\n";
            logFile << "Memory transfer to device time: " << memoryTransferToDeviceTime << " ms\n";
            logFile << "Memory transfer from device time: " << memoryTransferFromDeviceTime << " ms\n";
            logFile << "Bytes transferred to device: " << bytesTransferredToDevice << " bytes\n";
            logFile << "Bytes transferred from device: " << bytesTransferredFromDevice << " bytes\n";
            logFile << "GPU utilization: " << gpuUtilization << " %\n";
            logFile << "Number of kernel calls: " << kernelCalls << "\n\n";
            
            // Extended metrics
            logFile << "--- Extended Rendering Metrics ---\n";
            logFile << "1. Total render time: " << totalRenderTime << " seconds\n";
            logFile << "2. Average time per iteration: " << timePerIteration << " ms\n";
            
            logFile << "3. Kernel timing breakdown:\n";
            logFile << "   - Ray generation: " << kernelBreakdown.generateRayTime << " ms (" 
                    << (kernelBreakdown.generateRayTime / totalKernelTime * 100.0f) << "%)\n";
            logFile << "   - Intersection: " << kernelBreakdown.intersectionTime << " ms (" 
                    << (kernelBreakdown.intersectionTime / totalKernelTime * 100.0f) << "%)\n";
            logFile << "   - Shading: " << kernelBreakdown.shadingTime << " ms (" 
                    << (kernelBreakdown.shadingTime / totalKernelTime * 100.0f) << "%)\n";
            logFile << "   - Stream compaction: " << kernelBreakdown.streamCompactionTime << " ms (" 
                    << (kernelBreakdown.streamCompactionTime / totalKernelTime * 100.0f) << "%)\n";
            logFile << "   - Material sorting: " << kernelBreakdown.materialSortingTime << " ms (" 
                    << (kernelBreakdown.materialSortingTime / totalKernelTime * 100.0f) << "%)\n";
            logFile << "   - Final gather: " << kernelBreakdown.finalGatherTime << " ms (" 
                    << (kernelBreakdown.finalGatherTime / totalKernelTime * 100.0f) << "%)\n";
            
            logFile << "4. Samples per second: " << samplesPerSecond << " rays/s\n";
            logFile << "5. GPU memory usage: " << (float)totalGpuMemoryUsed / (1024.0f * 1024.0f) << " MB";
            
            if (sharedMemoryUsed > 0) {
                logFile << " (shared memory: " << (float)sharedMemoryUsed / 1024.0f << " KB)\n";
            } else {
                logFile << "\n";
            }
            
            logFile << "7. Max iterations: " << MAX_ITERATIONS << "\n";
            logFile << "=================================================\n\n";
            logFile.close();
        }
        
        // Also output to console
        printf("\n====== PERFORMANCE METRICS SUMMARY ======\n");
        printf("1. Total render time: %.2f seconds\n", totalRenderTime);
        printf("2. Average time per iteration: %.2f ms\n", timePerIteration);
        printf("3. Kernel breakdown: Ray gen (%.1f%%), Intersect (%.1f%%), Shade (%.1f%%)\n",
               kernelBreakdown.generateRayTime / totalKernelTime * 100.0f,
               kernelBreakdown.intersectionTime / totalKernelTime * 100.0f,
               kernelBreakdown.shadingTime / totalKernelTime * 100.0f);
        printf("4. Samples per second: %.2f million rays/s\n", samplesPerSecond / 1000000.0f);
        printf("5. GPU memory: %.2f MB\n", (float)totalGpuMemoryUsed / (1024.0f * 1024.0f));
        printf("6. GPU utilization: %.1f%%\n", gpuUtilization);
        printf("7. Total iterations: %d (max: %d)\n", iter, MAX_ITERATIONS);
        printf("======================================\n\n");
        
        // Mark that metrics have been recorded
        metricsRecorded = true;
    }
};

// Group3 Mod - Static metrics instance
static PerformanceMetrics metrics;

// Group3 Mod - Helper function to time CUDA memory transfers to device
float timeMemcpyToDevice(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpy(dst, src, count, kind);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    metrics.bytesTransferredToDevice += count;
    metrics.memoryTransferToDeviceTime += milliseconds;
    
    return milliseconds;
}

// Group3 Mod - Helper function to time CUDA memory transfers from device
float timeMemcpyFromDevice(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpy(dst, src, count, kind);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    metrics.bytesTransferredFromDevice += count;
    metrics.memoryTransferFromDeviceTime += milliseconds;
    
    return milliseconds;
}

// Group3 Mod - Function to get GPU memory usage
void updateGpuMemoryUsage() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    metrics.totalGpuMemoryUsed = total - free;
}

// Random number generator initialization
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//---------- Group3 Mod - BEGIN BVH ACCELERATION STRUCTURES ----------//

// Group3 Mod - AABB structure for BVH
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    
    __host__ __device__
    AABB() : min(FLT_MAX), max(-FLT_MAX) {}
    
    __host__ __device__
    AABB(const glm::vec3& min_, const glm::vec3& max_) : min(min_), max(max_) {}
    
    // Combine two AABBs
    __host__ __device__
    static AABB combine(const AABB& a, const AABB& b) {
        return AABB(glm::min(a.min, b.min), glm::max(a.max, b.max));
    }
    
    // Get surface area of the box
    __host__ __device__
    float surfaceArea() const {
        glm::vec3 extent = max - min;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
    
    // Get centroid of the box
    __host__ __device__
    glm::vec3 centroid() const {
        return (min + max) * 0.5f;
    }
    
    // Axis-aligned extent along a particular axis (0=x, 1=y, 2=z)
    __host__ __device__
    float axisLength(int axis) const {
        return max[axis] - min[axis];
    }
    
    // Optimized ray-AABB intersection test
    __device__
    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
        // Precompute inverse direction for efficiency
        const glm::vec3 invDir = 1.0f / ray.direction;
        
        // Check for rays parallel to axes
        const glm::vec3 t0 = (min - ray.origin) * invDir;
        const glm::vec3 t1 = (max - ray.origin) * invDir;
        
        const glm::vec3 tsmaller = glm::min(t0, t1);
        const glm::vec3 tbigger = glm::max(t0, t1);
        
        tMin = glm::max(glm::max(tsmaller.x, tsmaller.y), tsmaller.z);
        tMax = glm::min(glm::min(tbigger.x, tbigger.y), tbigger.z);
        
        return tMax >= tMin && tMax > 0;
    }
};

// Group3 Mod - Optimized BVH node structure (renamed to avoid conflict with existing BVHNode)
struct OptimizedBVHNode {
    AABB bounds;
    int firstChild;    // Index of first child (second is firstChild+1)
    int primOffset;    // Starting index in primitive array
    int primCount;     // Number of primitives (0 for internal nodes)
    
    __host__ __device__
    bool isLeaf() const {
        return primCount > 0;
    }
};

// Group3 Mod - Optimized BVH structure (renamed to avoid conflict)
struct OptimizedBVH {
    OptimizedBVHNode* nodes;   // Array of BVH nodes
    int* primIndices;          // Reordered primitive indices
    int nodeCount;             // Total number of nodes
    int rootIndex;             // Index of the root node
};

// Static BVH data
static OptimizedBVH* dev_bvh = nullptr;
static OptimizedBVH* host_bvh = nullptr;

// Group3 Mod - Helper struct for BVH construction
struct PrimitiveBounds {
    AABB bounds;
    int index;
};

// Group3 Mod - SAH Bucket for BVH construction
struct SAHBucket {
    AABB bounds;
    int count = 0;
};

//---------- Group3 Mod - END BVH ACCELERATION STRUCTURES ----------//

// Group3 Mod - Kernel that writes the image to the OpenGL PBO directly
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        // Apply simple tone mapping (gamma correction)
        pix = glm::pow(pix / (float)iter, glm::vec3(1.0f / 2.2f));
        
        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Scene data
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;

// Device memory pointers
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

//---------- Group3 Mod - BEGIN STREAM COMPACTION DATA ----------//
static int* dev_path_flags = NULL;
static PathSegment* dev_paths_compact = NULL;
static int* dev_path_indices = NULL;
static int* dev_path_count = NULL;
static thrust::device_ptr<PathSegment> dev_paths_thrust;
static thrust::device_ptr<PathSegment> dev_paths_compact_thrust;
static thrust::device_ptr<int> dev_path_flags_thrust;
static thrust::device_ptr<int> dev_path_indices_thrust;
//---------- Group3 Mod - END STREAM COMPACTION DATA ----------//

//---------- Group3 Mod - BEGIN MATERIAL SORTING DATA ----------//
static int* dev_material_ids = NULL;
static thrust::device_ptr<int> dev_material_ids_thrust;
static thrust::device_ptr<ShadeableIntersection> dev_intersections_thrust;
//---------- Group3 Mod - END MATERIAL SORTING DATA ----------//

// Constants for path tracing
#define MAX_BVH_NODES (1 << MAX_BVH_DEPTH)

void InitDataContainer(GuiDataContainer* imGuiData) {
    guiData = imGuiData;
}

//---------- Group3 Mod - BEGIN SAH-BASED BVH CONSTRUCTION ----------//

// Group3 Mod - SAH-based BVH construction helper function
float evaluateSAH(const AABB& parentBounds, const AABB& leftBounds, const AABB& rightBounds, 
                  int leftCount, int rightCount) {
    float parentArea = parentBounds.surfaceArea();
    if (parentArea <= 0.0f) return FLT_MAX;
    
    // SAH cost formula: traversal cost + primitive-intersection cost * probability of hitting child
    const float traversalCost = 1.0f;
    const float intersectionCost = 1.0f;
    
    float leftProbability = leftCount > 0 ? (leftBounds.surfaceArea() / parentArea) : 0.0f;
    float rightProbability = rightCount > 0 ? (rightBounds.surfaceArea() / parentArea) : 0.0f;
    
    return traversalCost + intersectionCost * (leftProbability * leftCount + rightProbability * rightCount);
}

// Group3 Mod - Creates a split bucket index based on a centroid value
int computeBucketIndex(float centroid, float minCentroid, float maxCentroid, int numBuckets) {
    float normalizedPos = (centroid - minCentroid) / (maxCentroid - minCentroid);
    return glm::min(numBuckets - 1, glm::max(0, (int)(normalizedPos * numBuckets)));
}

// Group3 Mod - Find best split using Surface Area Heuristic
int findBestSplitSAH(std::vector<PrimitiveBounds>& primBounds, int start, int end, 
                     int& bestAxis, float& splitPos) {
    // Compute the full bounds of all primitives in this node
    AABB nodeBounds;
    for (int i = start; i < end; i++) {
        nodeBounds = AABB::combine(nodeBounds, primBounds[i].bounds);
    }
    
    float bestCost = FLT_MAX;
    int bestSplitBucket = -1;
    int bestSplitIndex = (start + end) / 2; // Default to middle split
    bestAxis = 0;
    
    // For each axis, try to find the best split
    for (int axis = 0; axis < 3; axis++) {
        // Skip axes with minimal extent
        if (nodeBounds.axisLength(axis) < 1e-4f) continue;
        
        // Sort primitives along this axis
        std::sort(primBounds.begin() + start, primBounds.begin() + end,
            [axis](const PrimitiveBounds& a, const PrimitiveBounds& b) {
                // Sort by centroid along axis
                return a.bounds.centroid()[axis] < b.bounds.centroid()[axis];
            });
        
        // Find the extent of centroids along this axis
        float minCentroid = primBounds[start].bounds.centroid()[axis];
        float maxCentroid = primBounds[end - 1].bounds.centroid()[axis];
        
        // If centroids are very close, skip this axis
        if (maxCentroid - minCentroid < 1e-4f) continue;
        
        // Create SAH buckets
        const int nBuckets = MAX_SAH_BUCKETS;
        SAHBucket buckets[MAX_SAH_BUCKETS];
        
        // Place primitives into buckets
        for (int i = start; i < end; i++) {
            float centroid = primBounds[i].bounds.centroid()[axis];
            int bucketIndex = computeBucketIndex(centroid, minCentroid, maxCentroid, nBuckets);
            buckets[bucketIndex].count++;
            buckets[bucketIndex].bounds = AABB::combine(buckets[bucketIndex].bounds, primBounds[i].bounds);
        }
        
        // Evaluate cost of splitting after each bucket
        for (int splitBucket = 1; splitBucket < nBuckets; splitBucket++) {
            // Compute left side bounds and count
            AABB leftBounds;
            int leftCount = 0;
            for (int b = 0; b < splitBucket; b++) {
                leftBounds = AABB::combine(leftBounds, buckets[b].bounds);
                leftCount += buckets[b].count;
            }
            
            if (leftCount == 0) continue;
            
            // Compute right side bounds and count
            AABB rightBounds;
            int rightCount = 0;
            for (int b = splitBucket; b < nBuckets; b++) {
                rightBounds = AABB::combine(rightBounds, buckets[b].bounds);
                rightCount += buckets[b].count;
            }
            
            if (rightCount == 0) continue;
            
            // Compute SAH cost for this split
            float cost = evaluateSAH(nodeBounds, leftBounds, rightBounds, leftCount, rightCount);
            
            // Update best split if this is better
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitBucket = splitBucket;
                
                // Compute actual split position between buckets
                float bucketWidth = (maxCentroid - minCentroid) / nBuckets;
                splitPos = minCentroid + bucketWidth * splitBucket;
            }
        }
    }
    
    // If we found a good split, compute the actual split index
    if (bestSplitBucket != -1) {
        // Re-sort along best axis if not already sorted
        if (bestSplitIndex != 2) {
            std::sort(primBounds.begin() + start, primBounds.begin() + end,
                [bestAxis](const PrimitiveBounds& a, const PrimitiveBounds& b) {
                    return a.bounds.centroid()[bestAxis] < b.bounds.centroid()[bestAxis];
                });
        }
        
        // Find the split point where centroids cross the split position
        for (int i = start; i < end; i++) {
            if (primBounds[i].bounds.centroid()[bestAxis] >= splitPos) {
                bestSplitIndex = i;
                break;
            }
        }
        
        // Ensure we don't create empty partitions
        if (bestSplitIndex == start || bestSplitIndex == end) {
            bestSplitIndex = (start + end) / 2;
        }
    }
    
    return bestSplitIndex;
}

// Group3 Mod - Recursive function to build the BVH
int buildBVHRecursive(OptimizedBVH* bvh, std::vector<PrimitiveBounds>& primBounds, 
                      int start, int end, int depth, int& nodeIdx) {
    // Create a new node
    int currentNodeIdx = nodeIdx++;
    OptimizedBVHNode& node = bvh->nodes[currentNodeIdx];
    
    // Compute bounds for all primitives in this node
    node.bounds = primBounds[start].bounds;
    for (int i = start + 1; i < end; i++) {
        node.bounds = AABB::combine(node.bounds, primBounds[i].bounds);
    }
    
    // Determine if this should be a leaf node
    int numPrims = end - start;
    if (numPrims <= MAX_BVH_PRIMS_PER_NODE || depth >= MAX_BVH_DEPTH) {
        // Create leaf node
        node.primOffset = start;
        node.primCount = numPrims;
        node.firstChild = -1;
        
        // Store primitive indices
        for (int i = start; i < end; i++) {
            bvh->primIndices[i] = primBounds[i].index;
        }
    } else {
        // Create interior node - find best split
        int axis;
        float splitPos;
        int splitIndex = findBestSplitSAH(primBounds, start, end, axis, splitPos);
        
        // Mark as interior node
        node.primOffset = 0;
        node.primCount = 0;
        node.firstChild = nodeIdx;
        
        // Recursively build left and right children
        buildBVHRecursive(bvh, primBounds, start, splitIndex, depth + 1, nodeIdx);
        buildBVHRecursive(bvh, primBounds, splitIndex, end, depth + 1, nodeIdx);
    }
    
    return currentNodeIdx;
}

// Group3 Mod - Main function to build the BVH acceleration structure
void buildOptimizedBVH(Scene* scene) {
    int numPrims = scene->geoms.size();
    if (numPrims == 0) return;
    
    printf("Building optimized BVH with %d primitives...\n", numPrims);
    
    // Calculate bounds for all primitives
    std::vector<PrimitiveBounds> primBounds(numPrims);
    for (int i = 0; i < numPrims; i++) {
        PrimitiveBounds& pb = primBounds[i];
        pb.index = i;
        
        const Geom& geom = scene->geoms[i];
        glm::vec3 center = geom.translation;
        
        if (geom.type == SPHERE) {
            float radius = glm::max(glm::max(geom.scale.x, geom.scale.y), geom.scale.z);
            pb.bounds.min = center - glm::vec3(radius);
            pb.bounds.max = center + glm::vec3(radius);
        } else if (geom.type == CUBE) {
            pb.bounds.min = center - geom.scale * 0.5f;
            pb.bounds.max = center + geom.scale * 0.5f;
        }
    }
    
    // Allocate memory for BVH
    int maxNodes = std::min(2 * numPrims, MAX_BVH_NODES);
    host_bvh = new OptimizedBVH;
    host_bvh->nodeCount = 0;
    host_bvh->rootIndex = 0;
    host_bvh->nodes = new OptimizedBVHNode[maxNodes];
    host_bvh->primIndices = new int[numPrims];
    
    // Recursively build the BVH
    int nodeIdx = 0;
    buildBVHRecursive(host_bvh, primBounds, 0, numPrims, 0, nodeIdx);
    host_bvh->nodeCount = nodeIdx;
    
    printf("BVH construction complete: %d nodes created\n", nodeIdx);
    
    // Copy BVH to device
    cudaMalloc(&dev_bvh, sizeof(OptimizedBVH));
    
    OptimizedBVHNode* dev_nodes;
    cudaMalloc(&dev_nodes, host_bvh->nodeCount * sizeof(OptimizedBVHNode));
    cudaMemcpy(dev_nodes, host_bvh->nodes, host_bvh->nodeCount * sizeof(OptimizedBVHNode), cudaMemcpyHostToDevice);
    
    int* dev_primIndices;
    cudaMalloc(&dev_primIndices, numPrims * sizeof(int));
    cudaMemcpy(dev_primIndices, host_bvh->primIndices, numPrims * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create device copy of BVH
    OptimizedBVH deviceBVH;
    deviceBVH.nodes = dev_nodes;
    deviceBVH.primIndices = dev_primIndices;
    deviceBVH.nodeCount = host_bvh->nodeCount;
    deviceBVH.rootIndex = host_bvh->rootIndex;
    
    cudaMemcpy(dev_bvh, &deviceBVH, sizeof(OptimizedBVH), cudaMemcpyHostToDevice);
}

// Group3 Mod - Clean up BVH resources
void cleanupBVH() {
    if (host_bvh) {
        // Get device pointers from GPU
        OptimizedBVH deviceBVH;
        cudaMemcpy(&deviceBVH, dev_bvh, sizeof(OptimizedBVH), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(deviceBVH.nodes);
        cudaFree(deviceBVH.primIndices);
        cudaFree(dev_bvh);
        
        // Free host memory
        delete[] host_bvh->nodes;
        delete[] host_bvh->primIndices;
        delete host_bvh;
        
        host_bvh = nullptr;
        dev_bvh = nullptr;
        
        printf("BVH resources cleaned up\n");
    }
}

//---------- Group3 Mod - END SAH-BASED BVH CONSTRUCTION ----------//

//---------- Group3 Mod - BEGIN INITIALIZATION & CLEANUP ----------//

void pathtraceInit(Scene* scene) {
    // Start timing the rendering process
    metrics.startMeasurement();
    
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Allocate memory for image and path tracing structures
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    // Allocate memory for path tracing
    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    dev_paths_thrust = thrust::device_pointer_cast(dev_paths);
    
    // Group3 Mod - Allocate memory for stream compaction
    cudaMalloc(&dev_paths_compact, pixelcount * sizeof(PathSegment));
    dev_paths_compact_thrust = thrust::device_pointer_cast(dev_paths_compact);
    
    cudaMalloc(&dev_path_flags, pixelcount * sizeof(int));
    dev_path_flags_thrust = thrust::device_pointer_cast(dev_path_flags);
    
    cudaMalloc(&dev_path_indices, pixelcount * sizeof(int));
    dev_path_indices_thrust = thrust::device_pointer_cast(dev_path_indices);
    
    cudaMalloc(&dev_path_count, sizeof(int));
    
    // Group3 Mod - Allocate memory for material sorting
    cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));
    dev_material_ids_thrust = thrust::device_pointer_cast(dev_material_ids);

    // Copy scene data to device
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    timeMemcpyToDevice(dev_geoms, scene->geoms.data(), 
                      scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    timeMemcpyToDevice(dev_materials, scene->materials.data(), 
                      scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    // Allocate memory for intersections
    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    dev_intersections_thrust = thrust::device_pointer_cast(dev_intersections);
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // Group3 Mod - Build acceleration structure if we have enough primitives
    if (scene->geoms.size() > 1) {
        buildOptimizedBVH(scene);
    }

    // Track GPU memory usage
    updateGpuMemoryUsage();

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    // Free memory for ray tracing
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    
    // Group3 Mod - Free memory for stream compaction
    cudaFree(dev_paths_compact);
    cudaFree(dev_path_flags);
    cudaFree(dev_path_indices);
    cudaFree(dev_path_count);
    
    // Group3 Mod - Free memory for material sorting
    cudaFree(dev_material_ids);
    
    // Group3 Mod - Free acceleration structures
    cleanupBVH();
}

//---------- Group3 Mod - END INITIALIZATION & CLEANUP ----------//

//---------- Group3 Mod - BEGIN PATH GENERATION WITH ANTIALIASING ----------//

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        
        // Group3 Mod - Initialize RNG for jittering
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        
        // Group3 Mod - Apply jitter for antialiasing (more controlled amount)
        float jitterX = (u01(rng) - 0.5f) * 0.7f;
        float jitterY = (u01(rng) - 0.5f) * 0.7f;

        // Set ray origin at camera position
        segment.ray.origin = cam.position;
        
        // Calculate ray direction with jitter
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );
        
        // Initialize segment data
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
    }
}

//---------- Group3 Mod - END PATH GENERATION WITH ANTIALIASING ----------//

//---------- Group3 Mod - BEGIN BVH TRAVERSAL ----------//

// Group3 Mod - High-performance BVH traversal
__device__ bool traverseBVH(
    const OptimizedBVH* bvh,
    const Ray& ray,
    const Geom* geoms,
    float& t_min,
    int& hit_geom_index,
    glm::vec3& intersect_point,
    glm::vec3& normal,
    bool& outside
) {
    // Stack-based BVH traversal (non-recursive)
    const int MAX_STACK = MAX_BVH_DEPTH * 2;
    int nodeStack[MAX_STACK];
    int stackPtr = 0;
    
    nodeStack[stackPtr++] = bvh->rootIndex;
    
    bool hit = false;
    t_min = FLT_MAX;
    
    // Group3 Mod - Precompute ray inverse direction and sign for faster AABB tests
    const glm::vec3 invDir = 1.0f / ray.direction;
    const bool dirIsNeg[3] = {invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f};
    
    // Main traversal loop
    while (stackPtr > 0) {
        int nodeIdx = nodeStack[--stackPtr];
        const OptimizedBVHNode& node = bvh->nodes[nodeIdx];
        
        // Fast ray-box test
        float tmin, tmax;
        if (!node.bounds.intersect(ray, tmin, tmax) || tmin > t_min) {
            continue;  // Skip if no hit or further than current best
        }
        
        if (node.isLeaf()) {
            // Leaf node - test all contained primitives
            for (int i = 0; i < node.primCount; i++) {
                int primIdx = bvh->primIndices[node.primOffset + i];
                const Geom& geom = geoms[primIdx];
                
                // Primitive intersection test
                glm::vec3 tmp_intersect;
                glm::vec3 tmp_normal;
                bool tmp_outside = true;
                float t = -1.0f;
                
                if (geom.type == CUBE) {
                    t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
                } else if (geom.type == SPHERE) {
                    t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
                }
                
                // Update closest hit if this is closer
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_geom_index = primIdx;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                    outside = tmp_outside;
                    hit = true;
                }
            }
        } else {
            // Interior node - determine traversal order based on ray direction
            int firstChild = node.firstChild;
            int secondChild = node.firstChild + 1;
            
            // Group3 Mod - Visit the closer child first to terminate earlier
            int axis = 0;
            float maxComponent = fabsf(invDir.x);
            if (fabsf(invDir.y) > maxComponent) { axis = 1; maxComponent = fabsf(invDir.y); }
            if (fabsf(invDir.z) > maxComponent) { axis = 2; }
            
            if (dirIsNeg[axis]) {
                // Group3 Mod - Swap traversal order using device-safe swap
                deviceSwap(firstChild, secondChild);
            }
            
            // Push children onto stack in reverse order (closer one gets popped first)
            if (secondChild < bvh->nodeCount) { nodeStack[stackPtr++] = secondChild; }
            if (firstChild < bvh->nodeCount) { nodeStack[stackPtr++] = firstChild; }
        }
    }
    
    return hit;
}

//---------- Group3 Mod - END BVH TRAVERSAL ----------//

//---------- Group3 Mod - BEGIN INTERSECTION KERNELS ----------//

// Group3 Mod - Optimized ray-scene intersection using BVH
__global__ void computeIntersectionsBVH(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    OptimizedBVH* bvh
) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        PathSegment& pathSegment = pathSegments[path_index];
        
        // Skip rays that have already terminated
        if (pathSegment.remainingBounces <= 0) {
            intersections[path_index].t = -1.0f;
            return;
        }
        
        // Perform BVH traversal
        float t_min;
        int hit_geom_index;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        bool outside;
        
        bool hit = traverseBVH(bvh, pathSegment.ray, geoms, t_min, hit_geom_index, 
                               intersect_point, normal, outside);
        
        // Store intersection results
        if (!hit) {
            intersections[path_index].t = -1.0f;
        } else {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].point = intersect_point;
        }
    }
}

// Legacy intersection test without BVH
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment& pathSegment = pathSegments[path_index];
        
        // Skip rays that have already terminated
        if (pathSegment.remainingBounces <= 0) {
            intersections[path_index].t = -1.0f;
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // Parse through all scene geometry
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].point = intersect_point;
        }
    }
}

//---------- Group3 Mod - END INTERSECTION KERNELS ----------//

//---------- Group3 Mod - BEGIN MATERIAL SORTING ----------//

// Group3 Mod - Material ID marking kernel for sorting
__global__ void markMaterialIDs(
    int num_paths,
    ShadeableIntersection* intersections,
    int* materialIDs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        if (intersections[idx].t > 0.0f) {
            materialIDs[idx] = intersections[idx].materialId;
        } else {
            materialIDs[idx] = -1; // No intersection
        }
    }
}

//---------- Group3 Mod - END MATERIAL SORTING ----------//

//---------- Group3 Mod - BEGIN SHADING WITH SHARED MEMORY ----------//

// Group3 Mod - Optimized shader with shared memory for material caching
__global__ void shadeFakeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int* pathFlags  // Flags for active paths (1 = active, 0 = terminated)
)
{
    // Group3 Mod - Use shared memory for frequently used materials
    extern __shared__ Material sharedMaterials[];
    
    // Initialize shared memory with commonly used materials
    int tid = threadIdx.x;
    if (tid < 16) {
        // Load the first 16 materials into shared memory
        sharedMaterials[tid] = materials[tid % min(16, num_paths)];
    }
    __syncthreads(); // Ensure all threads see the initialized shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        // Initialize flag to active (will be set to inactive if ray terminates)
        pathFlags[idx] = 1;
        
        PathSegment& pathSegment = pathSegments[idx];
        
        // Skip rays that have already terminated
        if (pathSegment.remainingBounces <= 0) {
            pathFlags[idx] = 0;
            return;
        }
        
        ShadeableIntersection& intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG with depth factor to avoid correlation
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Group3 Mod - Use shared memory if possible, otherwise global memory
            Material material;
            int materialId = intersection.materialId;
            
            // Check if the material is in shared memory
            if (materialId < 16) {
                material = sharedMaterials[materialId];
            } else {
                material = materials[materialId];
            }
            
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegment.color *= materialColor * material.emittance;
                // Terminate the ray when it hits a light
                pathSegment.remainingBounces = 0;
                pathFlags[idx] = 0;
            }
            // Otherwise, do some pseudo-lighting computation
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegment.color *= (materialColor * lightTerm) * 0.3f + 
                                     ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegment.color *= u01(rng); // apply some noise for variation
                
                // Decrement remaining bounces
                pathSegment.remainingBounces--;
                
                // Mark path as terminated if no more bounces
                if (pathSegment.remainingBounces <= 0) {
                    pathFlags[idx] = 0;
                }
            }
        }
        else {
            // No intersection - terminate the ray
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            pathFlags[idx] = 0;
        }
    }
}

//---------- Group3 Mod - END SHADING WITH SHARED MEMORY ----------//

//---------- Group3 Mod - BEGIN STREAM COMPACTION ----------//

// Group3 Mod - Kernel to perform stream compaction with efficient operations
__global__ void compactPaths(
    PathSegment* pathsIn,
    PathSegment* pathsOut,
    int* pathFlags,
    int* pathIndices,
    int numPaths,
    int* pathCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPaths) {
        // If path is active (flag = 1), copy it to the output at its index position
        if (pathFlags[idx] == 1) {
            int outputIdx = pathIndices[idx];
            pathsOut[outputIdx] = pathsIn[idx];
        }
    }
}

//---------- Group3 Mod - END STREAM COMPACTION ----------//

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment& iterationPath = iterationPaths[index];
        
        // Only accumulate color for rays that have terminated
        if (iterationPath.remainingBounces <= 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
    }
}

//---------- Group3 Mod - BEGIN PERFORMANCE MONITORING ----------//

// Group3 Mod - Helper function to time kernel execution (standard version)
template<typename Func, typename... Args>
float timeKernelExecution(Func kernel, dim3 blocks, dim3 threads, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(args...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    metrics.totalKernelTime += milliseconds;
    metrics.kernelCalls++;
    
    return milliseconds;
}

// Group3 Mod - Helper specifically for shared memory kernels
float timeShaderWithSharedMemory(dim3 blocks, dim3 threads, int sharedMemBytes,
                                int iter, int depth, int num_paths, 
                                ShadeableIntersection* intersections,
                                PathSegment* paths, Material* materials,
                                int* pathFlags) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    shadeFakeMaterial<<<blocks, threads, sharedMemBytes>>>(
        iter, depth, num_paths, intersections, paths, materials, pathFlags
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    metrics.totalKernelTime += milliseconds;
    metrics.kernelCalls++;
    metrics.sharedMemoryUsed = sharedMemBytes;
    metrics.kernelBreakdown.shadingTime += milliseconds;
    
    return milliseconds;
}

//---------- Group3 Mod - END PERFORMANCE MONITORING ----------//

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
    // Group3 Mod - Check if we've reached the max iterations and end rendering
    if (iter > MAX_ITERATIONS) {
        // Print final metrics and return
        metrics.logMetrics(MAX_ITERATIONS, frame, hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y);
        return;
    }
    
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    // Timer for this iteration
    cudaEvent_t startIter, stopIter;
    cudaEventCreate(&startIter);
    cudaEventCreate(&stopIter);
    cudaEventRecord(startIter);

    ///////////////////////////////////////////////////////////////////////////

    // Group3 Mod - Time the generateRayFromCamera kernel
    float rayGenTime = timeKernelExecution(generateRayFromCamera, blocksPerGrid2d, blockSize2d, cam, iter, traceDepth, dev_paths);
    metrics.kernelBreakdown.generateRayTime += rayGenTime;
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        
        float intersectionTime = 0.0f;
        
        // Group3 Mod - Use BVH acceleration if available
        if (dev_bvh != nullptr) {
            // Group3 Mod - Time the BVH-accelerated intersection kernel
            intersectionTime = timeKernelExecution(computeIntersectionsBVH, numblocksPathSegmentTracing, blockSize1d,
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections,
                dev_bvh
            );
        } else {
            // Group3 Mod - Time the regular intersection kernel
            intersectionTime = timeKernelExecution(computeIntersections, numblocksPathSegmentTracing, blockSize1d,
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
            );
        }
        
        metrics.kernelBreakdown.intersectionTime += intersectionTime;
        checkCUDAError("trace one bounce");
        
        // Group3 Mod - Material sorting
        #if MATERIAL_SORTING
        if (num_paths > 1) {
            // Mark material IDs for sorting
            float materialSortTime = timeKernelExecution(markMaterialIDs, numblocksPathSegmentTracing, blockSize1d,
                              num_paths, dev_intersections, dev_material_ids);
            
            metrics.kernelBreakdown.materialSortingTime += materialSortTime;
            
            // Create a temporary copy for intersection sorting
            int* temp_material_ids = nullptr;
            cudaMalloc(&temp_material_ids, num_paths * sizeof(int));
            cudaMemcpy(temp_material_ids, dev_material_ids, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
            thrust::device_ptr<int> temp_material_ids_thrust(temp_material_ids);
            
            // Sort paths by material ID
            thrust::sort_by_key(dev_material_ids_thrust, dev_material_ids_thrust + num_paths, dev_paths_thrust);
            
            // Sort intersections by material ID
            thrust::sort_by_key(temp_material_ids_thrust, temp_material_ids_thrust + num_paths, dev_intersections_thrust);
            
            // Free temporary memory
            cudaFree(temp_material_ids);
        }
        #endif
        
        // Increment depth to track the bounce
        depth++;

        // Group3 Mod - Reset path flags for active/inactive rays
        cudaMemset(dev_path_flags, 0, num_paths * sizeof(int));

        // Group3 Mod - Use shared memory for shading
        int sharedMemSize = 16 * sizeof(Material);
        timeShaderWithSharedMemory(numblocksPathSegmentTracing, blockSize1d, sharedMemSize,
            iter, depth, num_paths, dev_intersections, dev_paths, dev_materials, dev_path_flags);
        
#if STREAM_COMPACTION
        // Group3 Mod - Stream Compaction using Thrust
        // Count number of active paths
        int activeCount = thrust::count(dev_path_flags_thrust, dev_path_flags_thrust + num_paths, 1);
        
        // If no active paths remain, we're done
        if (activeCount == 0) {
            iterationComplete = true;
            continue;
        }
        
        // Otherwise, perform stream compaction
        // Create indices for active paths (exclusive scan)
        thrust::exclusive_scan(dev_path_flags_thrust, dev_path_flags_thrust + num_paths, dev_path_indices_thrust);
        
        // Compact paths
        float compactionTime = timeKernelExecution(compactPaths, numblocksPathSegmentTracing, blockSize1d,
            dev_paths, dev_paths_compact, dev_path_flags, dev_path_indices, num_paths, dev_path_count);
        
        metrics.kernelBreakdown.streamCompactionTime += compactionTime;
        
        // Use direct pointer swap instead of std::swap (which is host-only)
        PathSegment* temp = dev_paths;
        dev_paths = dev_paths_compact;
        dev_paths_compact = temp;
        
        // Also swap the thrust pointers
        thrust::device_ptr<PathSegment> temp_thrust = dev_paths_thrust;
        dev_paths_thrust = dev_paths_compact_thrust;
        dev_paths_compact_thrust = temp_thrust;
        
        // Update path count
        num_paths = activeCount;
        
        // End iteration if we've reached max depth
        if (depth >= traceDepth) {
            iterationComplete = true;
        }
#else
        // Without stream compaction, just finish after one bounce
        iterationComplete = true; 
#endif

        // Update GUI info if available
        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    
    // Group3 Mod - Time the finalGather kernel
    float gatherTime = timeKernelExecution(finalGather, numBlocksPixels, blockSize1d, pixelcount, dev_image, dev_paths);
    metrics.kernelBreakdown.finalGatherTime += gatherTime;

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    // Group3 Mod - Time the sendImageToPBO kernel
    float sendTime = timeKernelExecution(sendImageToPBO, blocksPerGrid2d, blockSize2d, pbo, cam.resolution, iter, dev_image);
    metrics.kernelBreakdown.miscTime += sendTime;

    // Group3 Mod - Retrieve image from GPU and time the transfer
    timeMemcpyFromDevice(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        
    // Stop timer for this iteration
    cudaEventRecord(stopIter);
    cudaEventSynchronize(stopIter);
    
    float iterationTime = 0.0f;
    cudaEventElapsedTime(&iterationTime, startIter, stopIter);
    
    metrics.updateIterationTime(iterationTime);
    
    cudaEventDestroy(startIter);
    cudaEventDestroy(stopIter);
    
    // Update memory usage metrics
    updateGpuMemoryUsage();

    // Group3 Mod - Log the performance metrics when we reach MAX_ITERATIONS
    if (iter == MAX_ITERATIONS) {
        metrics.logMetrics(iter, frame, pixelcount);
        
        // Print a message that we've reached the iteration limit
        printf("\n====== RENDERING COMPLETE ======\n");
        printf("Reached maximum iterations (%d)\n", MAX_ITERATIONS);
        printf("Total render time: %.2f seconds\n", metrics.totalRenderTime);
        printf("================================\n\n");
    }

    checkCUDAError("pathtrace");
}