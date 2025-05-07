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
#include <algorithm>         // Group3 Mod - For std::sort
#include <stack>             // Group3 Mod - For BVH construction
#include <functional>        // Group3 Mod - For std::function

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// Maximum BVH depth for traversal
#define MAX_BVH_DEPTH 24

// ─────────── Timing with CUDA events ───────────
cudaEvent_t startKernel, stopKernel;
cudaEvent_t bvhStartKernel, bvhStopKernel;
float totalKernelTime = 0.0f;
float bvhBuildTime = 0.0f;

#define ERRORCHECK 1

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

// Group3 Mod - BVH Data Structures
struct BVHNode {
    glm::vec3 min;             // Minimum bounds
    glm::vec3 max;             // Maximum bounds
    int leftChildIndex;        // Index of the left child node
    int rightChildIndex;       // Index of the right child node
    int firstPrimitiveIndex;   // Index of first primitive in this node
    int primitivesCount;       // Number of primitives in this node
};

struct BVHBuildNode {
    int firstPrimitiveIndex;
    int primitivesCount;
    glm::vec3 min;
    glm::vec3 max;
    BVHBuildNode* left;
    BVHBuildNode* right;
};

struct PrimitiveRef {
    int index;                // Index of the primitive in the original array
    glm::vec3 centroid;       // Centroid of the primitive for sorting
    glm::vec3 min;            // Minimum bounds
    glm::vec3 max;            // Maximum bounds
};

// Group3 Mod - BVH device memory
static BVHNode* dev_bvhNodes = NULL;
static int* dev_primitiveIndices = NULL;
static int totalBVHNodes = 0;
static bool useBVH = false;

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
    float naiveIntersectTime = 0.0f;       // Time for naive intersection (ms)
    float bvhIntersectTime = 0.0f;         // Time for BVH intersection (ms)

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

// Group3 Mod - Get bounding box for a primitive
void getGeomBounds(const Geom& geom, glm::vec3& min, glm::vec3& max) {
    // For accessing Geom fields correctly
    if (geom.type == CUBE) {
        // For a cube, extract bounds from transform
        glm::vec3 halfExtent = 0.5f * glm::vec3(1.0f);  // Default 1x1x1 cube
        
        // Assuming transform matrix contains the position/scale information
        glm::vec3 center = glm::vec3(geom.transform[3]); // Position is in the 4th column
        glm::vec3 scale = glm::vec3(
            glm::length(glm::vec3(geom.transform[0])),  // Scale x from first column
            glm::length(glm::vec3(geom.transform[1])),  // Scale y from second column
            glm::length(glm::vec3(geom.transform[2]))   // Scale z from third column
        );
        
        // Calculate bounds using extracted information
        min = center - scale * halfExtent;
        max = center + scale * halfExtent;
    } 
    else if (geom.type == SPHERE) {
        // For a sphere, extract position and radius
        glm::vec3 center = glm::vec3(geom.transform[3]); // Position
        float radius = glm::length(glm::vec3(geom.transform[0])) * 0.5f; // Use scale from x column
        
        // Calculate bounds
        min = center - glm::vec3(radius);
        max = center + glm::vec3(radius);
    }
    // Add other primitive types as needed
}

// Group3 Mod - Build BVH tree recursively
BVHBuildNode* buildBVHRecursive(std::vector<PrimitiveRef>& primitiveRefs, int start, int end, int depth) {
    BVHBuildNode* node = new BVHBuildNode();
    
    // Calculate bounds for this node
    glm::vec3 boundMin(FLT_MAX);
    glm::vec3 boundMax(-FLT_MAX);
    
    for (int i = start; i < end; i++) {
        boundMin = glm::min(boundMin, primitiveRefs[i].min);
        boundMax = glm::max(boundMax, primitiveRefs[i].max);
    }
    
    node->min = boundMin;
    node->max = boundMax;
    
    int primitiveCount = end - start;
    
    // If this is a leaf node or we're too deep
    if (primitiveCount <= 4 || depth >= MAX_BVH_DEPTH) {
        node->firstPrimitiveIndex = start;
        node->primitivesCount = primitiveCount;
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    
    // Find the longest axis to split along
    glm::vec3 extent = boundMax - boundMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    
    float splitPos = 0.5f * (boundMin[axis] + boundMax[axis]);
    
    // Partition primitives
    int mid = start;
    for (int i = start; i < end; i++) {
        if (primitiveRefs[i].centroid[axis] < splitPos) {
            std::swap(primitiveRefs[i], primitiveRefs[mid]);
            mid++;
        }
    }
    
    // If we couldn't split, create a leaf
    if (mid == start || mid == end) {
        mid = start + primitiveCount / 2;
    }
    
    // Recursively build children
    node->left = buildBVHRecursive(primitiveRefs, start, mid, depth + 1);
    node->right = buildBVHRecursive(primitiveRefs, mid, end, depth + 1);
    
    return node;
}

// Group3 Mod - Flatten BVH tree into array for GPU
int flattenBVHTree(BVHBuildNode* node, std::vector<BVHNode>& bvhNodes, std::vector<int>& primitiveIndices, int& nextNodeIndex) {
    int currentNodeIndex = nextNodeIndex++;
    
    BVHNode& flatNode = bvhNodes[currentNodeIndex];
    flatNode.min = node->min;
    flatNode.max = node->max;
    
    // Handle leaf node
    if (!node->left && !node->right) {
        flatNode.leftChildIndex = -1;  // Mark as leaf
        flatNode.firstPrimitiveIndex = primitiveIndices.size();
        flatNode.primitivesCount = node->primitivesCount;
        
        // Copy primitive indices to sequential memory
        for (int i = 0; i < node->primitivesCount; i++) {
            primitiveIndices.push_back(node->firstPrimitiveIndex + i);
        }
    } else {
        // Interior node - recursively flatten children
        flatNode.primitivesCount = 0;
        
        // Left child first
        int leftChildIndex = flattenBVHTree(node->left, bvhNodes, primitiveIndices, nextNodeIndex);
        flatNode.leftChildIndex = leftChildIndex;
        
        // Right child
        int rightChildIndex = flattenBVHTree(node->right, bvhNodes, primitiveIndices, nextNodeIndex);
        flatNode.rightChildIndex = rightChildIndex;
    }
    
    // Clean up
    delete node;
    return currentNodeIndex;
}

// Group3 Mod - Build BVH from scene geometry
void buildBVH(const std::vector<Geom>& geoms) {
    // Start timing BVH build
    cudaEventRecord(bvhStartKernel);
    
    // Create primitive references from scene geometry
    std::vector<PrimitiveRef> primitiveRefs;
    primitiveRefs.resize(geoms.size());
    
    for (int i = 0; i < geoms.size(); i++) {
        primitiveRefs[i].index = i;
        getGeomBounds(geoms[i], primitiveRefs[i].min, primitiveRefs[i].max);
        primitiveRefs[i].centroid = 0.5f * (primitiveRefs[i].min + primitiveRefs[i].max);
    }
    
    // Build BVH recursively
    BVHBuildNode* root = buildBVHRecursive(primitiveRefs, 0, geoms.size(), 0);
    
    // Count nodes to allocate memory
    int nodeCount = 0;
    std::function<void(BVHBuildNode*)> countNodes = [&](BVHBuildNode* node) {
        if (!node) return;
        nodeCount++;
        if (node->left) countNodes(node->left);
        if (node->right) countNodes(node->right);
    };
    countNodes(root);
    
    // Allocate memory for flattened tree
    std::vector<BVHNode> bvhNodes(nodeCount);
    std::vector<int> primitiveIndices;
    primitiveIndices.reserve(geoms.size());
    
    // Flatten the BVH tree for GPU
    int nextNodeIndex = 0;
    flattenBVHTree(root, bvhNodes, primitiveIndices, nextNodeIndex);
    
    // Copy BVH to GPU
    totalBVHNodes = bvhNodes.size();
    cudaMalloc(&dev_bvhNodes, totalBVHNodes * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, bvhNodes.data(), totalBVHNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
    
    cudaMalloc(&dev_primitiveIndices, primitiveIndices.size() * sizeof(int));
    cudaMemcpy(dev_primitiveIndices, primitiveIndices.data(), primitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // End timing BVH build
    cudaEventRecord(bvhStopKernel);
    cudaEventSynchronize(bvhStopKernel);
    cudaEventElapsedTime(&bvhBuildTime, bvhStartKernel, bvhStopKernel);
    
    printf("BVH built with %d nodes for %zu primitives in %.2f ms\n", 
           totalBVHNodes, geoms.size(), bvhBuildTime);
    
    useBVH = true;
}

// Group3 Mod - Check if a ray intersects with an AABB (Axis-Aligned Bounding Box)
__device__ bool rayAABBIntersection(const Ray& ray, const glm::vec3& min, const glm::vec3& max, float& t_near, float& t_far) {
    // Calculate inverse direction to avoid divisions
    glm::vec3 invDir = 1.0f / ray.direction;
    
    // Calculate intersections with x, y, z slabs
    glm::vec3 tmin = (min - ray.origin) * invDir;
    glm::vec3 tmax = (max - ray.origin) * invDir;
    
    // Handle negative directions where min and max are swapped
    glm::vec3 real_tmin = glm::min(tmin, tmax);
    glm::vec3 real_tmax = glm::max(tmin, tmax);
    
    // Find largest tmin and smallest tmax
    t_near = glm::max(glm::max(real_tmin.x, real_tmin.y), real_tmin.z);
    t_far = glm::min(glm::min(real_tmax.x, real_tmax.y), real_tmax.z);
    
    // Check if there's an intersection
    return t_far >= t_near && t_far > 0;
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
    cudaEventCreate(&bvhStartKernel);
    cudaEventCreate(&bvhStopKernel);

    // Build BVH acceleration structure
    buildBVH(scene->geoms);

    updateGpuMemory();  // Group3 Mod - record initial GPU memory usage
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    // Make sure all CUDA operations have completed before cleanup
    cudaDeviceSynchronize();
    
    // Clean up BVH memory with null checks
    if (dev_bvhNodes) {
        cudaFree(dev_bvhNodes);
        dev_bvhNodes = nullptr;
    }
    
    if (dev_primitiveIndices) {
        cudaFree(dev_primitiveIndices);
        dev_primitiveIndices = nullptr;
    }
    
    // Clean up CUDA event handles with error checking
    if (startKernel) {
        cudaEventSynchronize(startKernel);  // Ensure event has completed
        cudaEventDestroy(startKernel);
    }
    
    if (stopKernel) {
        cudaEventSynchronize(stopKernel);
        cudaEventDestroy(stopKernel);
    }
    
    if (bvhStartKernel) {
        cudaEventSynchronize(bvhStartKernel);
        cudaEventDestroy(bvhStartKernel);
    }
    
    if (bvhStopKernel) {
        cudaEventSynchronize(bvhStopKernel);
        cudaEventDestroy(bvhStopKernel);
    }
    
    // Free device memory with null checks
    if (dev_image) {
        cudaFree(dev_image);
        dev_image = nullptr;
    }
    
    if (dev_paths) {
        cudaFree(dev_paths);
        dev_paths = nullptr;
    }
    
    if (dev_geoms) {
        cudaFree(dev_geoms);
        dev_geoms = nullptr;
    }
    
    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }
    
    if (dev_intersections) {
        cudaFree(dev_intersections);
        dev_intersections = nullptr;
    }

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

// Group3 Mod - BVH Intersection test
__global__ void computeBVHIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    BVHNode* bvhNodes,
    int* primitiveIndices,
    ShadeableIntersection* intersections
) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths) {
        PathSegment pathSegment = pathSegments[path_index];
        Ray ray = pathSegment.ray;
        
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        glm::vec3 hit_point, hit_normal;
        bool hit_outside = true;
        
        // Create stack for traversal (max depth nodes)
        int nodeStack[MAX_BVH_DEPTH];
        int stackIndex = 0;
        nodeStack[stackIndex++] = 0; // Start with root node
        
        // Traverse BVH
        while (stackIndex > 0) {
            int nodeIdx = nodeStack[--stackIndex];
            BVHNode node = bvhNodes[nodeIdx];
            
            // Check ray-node intersection
            float t_near, t_far;
            if (!rayAABBIntersection(ray, node.min, node.max, t_near, t_far) || t_near > t_min) {
                continue; // Skip this node if no intersection or already found closer hit
            }
            
            // Leaf node - test primitives
            if (node.leftChildIndex < 0) {
                for (int i = 0; i < node.primitivesCount; i++) {
                    int primIdx = primitiveIndices[node.firstPrimitiveIndex + i];
                    Geom& geom = geoms[primIdx];
                    
                    float t;
                    glm::vec3 tmp_intersect, tmp_normal;
                    bool outside = true;
                    
                    if (geom.type == CUBE) {
                        t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
                    } else if (geom.type == SPHERE) {
                        t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
                    }
                    
                    if (t > 0.0f && t < t_min) {
                        t_min = t;
                        hit_geom_index = primIdx;
                        hit_point = tmp_intersect;
                        hit_normal = tmp_normal;
                        hit_outside = outside;
                    }
                }
            } else {
                // Interior node - add children to stack
                // Add closer child first (front to back traversal)
                nodeStack[stackIndex++] = node.leftChildIndex;
                nodeStack[stackIndex++] = node.rightChildIndex;
            }
        }
        
        // Store intersection results
        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
        } else {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].point = hit_point;
            intersections[path_index].surfaceNormal = hit_normal;
        }
    }
}

__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            } else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f 
                                         + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng);
            }
        } else {
            pathSegments[idx].color = glm::vec3(0.0f);
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

        // Intersection - Alternate between naive and BVH methods by frame
        cudaEventRecord(startKernel);
        
        // Use naive method on even frames, BVH method on odd frames for comparison
        if (frame % 2 == 0 || !useBVH) {
            // Naive intersection
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections
            );
            cudaEventRecord(stopKernel);
            cudaEventSynchronize(stopKernel);
            cudaEventElapsedTime(&intersectTime, startKernel, stopKernel);
            metrics.naiveIntersectTime = intersectTime;
        } else {
            // BVH intersection
            computeBVHIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth, num_paths, dev_paths, dev_geoms, dev_bvhNodes, dev_primitiveIndices, dev_intersections
            );
            cudaEventRecord(stopKernel);
            cudaEventSynchronize(stopKernel);
            cudaEventElapsedTime(&intersectTime, startKernel, stopKernel);
            metrics.bvhIntersectTime = intersectTime;
        }
        
        totalKernelTime += intersectTime;
        checkCUDAError("trace one bounce");
        depth++;

        // Shading
        cudaEventRecord(startKernel);
        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter, num_paths, dev_intersections, dev_paths, dev_materials
        );
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

    // ────────────── Enhanced Summary Printout with BVH Metrics ──────────────
    printf("\n====== PERFORMANCE METRICS SUMMARY (Iteration %d) ======\n", iter);
    printf("Total render time: %.2f seconds\n", metrics.totalRenderTime);
    printf("Avg iteration time: %.2f ms\n", metrics.avgIterationTime);
    printf("Samples per second: %.2f million rays/s\n", metrics.samplesPerSecond / 1e6f);
    printf("GPU memory used: %.2f MB\n", metrics.gpuMemoryUsed / float(1 << 20));
    
    if (psnr == FLT_MAX) printf("PSNR: Inf dB\n");
    else printf("PSNR: %.2f dB\n", psnr);
    
    if (metrics.iterationsToClean > 0)
        printf("Iterations to clean: %d\n", metrics.iterationsToClean);
    
    printf("\nKERNEL BREAKDOWN:\n");
    printf("Total kernel time: %.2f ms\n", totalKernelTime);
    printf("  - Ray generation:   %.2f ms (%.1f%%)\n", rayGenTime, 
           (rayGenTime/totalKernelTime)*100.0f);
    printf("  - Intersection:     %.2f ms (%.1f%%)\n", intersectTime, 
           (intersectTime/totalKernelTime)*100.0f);
    printf("  - Shading:          %.2f ms (%.1f%%)\n", shadeTime, 
           (shadeTime/totalKernelTime)*100.0f);
    printf("  - Final gather:     %.2f ms (%.1f%%)\n", gatherTime, 
           (gatherTime/totalKernelTime)*100.0f);
    
    // BVH Performance comparison  
    if (frame >= 3 && metrics.naiveIntersectTime > 0 && metrics.bvhIntersectTime > 0) {
        float speedup = metrics.naiveIntersectTime / metrics.bvhIntersectTime;
        printf("\nBVH ACCELERATION:\n");
        printf("  - Naive intersection: %.2f ms\n", metrics.naiveIntersectTime);
        printf("  - BVH intersection:   %.2f ms\n", metrics.bvhIntersectTime);
        printf("  - Speedup factor:     %.2fx\n", speedup);
        printf("  - BVH build time:     %.2f ms\n", bvhBuildTime);
    }
    
    printf("================================================\n");

    // Store final image
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}