#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <limits>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <fstream>
#include <chrono>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// ─────────── BVH DATA STRUCTURES ───────────
struct AABB {
    glm::vec3 min, max;
};

struct BVHNodeGPU {
    AABB bounds;
    int left, right;    // child indices, -1 for leaf
    int geomIndex;      // leaf: index into geoms array
};

AABB computeBounds(const Geom& g) {
    glm::vec3 corners[8] = {
        {-0.5f,-0.5f,-0.5f},{+0.5f,-0.5f,-0.5f},
        {-0.5f,+0.5f,-0.5f},{+0.5f,+0.5f,-0.5f},
        {-0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,+0.5f},
        {-0.5f,+0.5f,+0.5f},{+0.5f,+0.5f,+0.5f}
    };
    AABB box;
    box.min = glm::vec3(std::numeric_limits<float>::max());
    box.max = glm::vec3(-std::numeric_limits<float>::max());
    for (int i = 0; i < 8; ++i) {
        glm::vec4 w = g.transform * glm::vec4(corners[i], 1.0f);
        box.min = glm::min(box.min, glm::vec3(w));
        box.max = glm::max(box.max, glm::vec3(w));
    }
    return box;
}

int buildBVHRecursive(
    const std::vector<AABB>& bboxes,
    std::vector<int>& indices,
    int start, int end,
    std::vector<BVHNodeGPU>& nodes)
{
    int nodeIdx = (int)nodes.size();
    nodes.push_back({});
    int count = end - start;
    if (count == 1) {
        nodes[nodeIdx].bounds    = bboxes[indices[start]];
        nodes[nodeIdx].left      = -1;
        nodes[nodeIdx].right     = -1;
        nodes[nodeIdx].geomIndex = indices[start];
        return nodeIdx;
    }
    // centroid bbox
    AABB cbox;
    cbox.min = glm::vec3(std::numeric_limits<float>::max());
    cbox.max = glm::vec3(-std::numeric_limits<float>::max());
    for (int i = start; i < end; ++i) {
        const AABB &b = bboxes[indices[i]];
        glm::vec3 cent = (b.min + b.max) * 0.5f;
        cbox.min = glm::min(cbox.min, cent);
        cbox.max = glm::max(cbox.max, cent);
    }
    glm::vec3 extent = cbox.max - cbox.min;
    int axis = (extent.x > extent.y && extent.x > extent.z) ? 0
             : (extent.y > extent.z) ? 1 : 2;
    std::sort(indices.begin() + start, indices.begin() + end,
        [&](int a, int b) {
            const AABB &ba = bboxes[a], &bb = bboxes[b];
            float ca = (ba.min[axis] + ba.max[axis]) * 0.5f;
            float cb = (bb.min[axis] + bb.max[axis]) * 0.5f;
            return ca < cb;
        });
    int mid = start + count/2;
    int leftChild  = buildBVHRecursive(bboxes, indices, start, mid, nodes);
    int rightChild = buildBVHRecursive(bboxes, indices, mid, end, nodes);
    nodes[nodeIdx].left      = leftChild;
    nodes[nodeIdx].right     = rightChild;
    nodes[nodeIdx].geomIndex = -1;
    // union bounds
    const AABB &bl = nodes[leftChild].bounds;
    const AABB &br = nodes[rightChild].bounds;
    nodes[nodeIdx].bounds.min = glm::min(bl.min, br.min);
    nodes[nodeIdx].bounds.max = glm::max(bl.max, br.max);
    return nodeIdx;
}

void buildBVH(const std::vector<Geom>& geoms, std::vector<BVHNodeGPU>& nodes) {
    int n = (int)geoms.size();
    std::vector<AABB> bboxes(n);
    for (int i = 0; i < n; ++i)
        bboxes[i] = computeBounds(geoms[i]);
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    nodes.clear();
    buildBVHRecursive(bboxes, indices, 0, n, nodes);
}

__device__ bool intersectAABB(const AABB &box, const Ray &r) {
    float tmin = 0.0f, tmax = FLT_MAX;
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / r.direction[i];
        float t0   = (box.min[i] - r.origin[i]) * invD;
        float t1   = (box.max[i] - r.origin[i]) * invD;
        if (invD < 0.0f) {
            // manual swap (std::swap not allowed in device code)
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin) return false;
    }
    return true;
}

// ─────────── GLOBAL BVH STORAGE ───────────
static BVHNodeGPU* dev_bvhNodes = nullptr;
static int          h_bvhNodeCount = 0;

// ─────────── Timing & Error Checking ───────────
cudaEvent_t startKernel, stopKernel;
float totalKernelTime = 0.0f;
#define ERRORCHECK 1
#define MAX_MATERIALS 64
#define FILENAME (strrchr(__FILE__,'/')?strrchr(__FILE__,'/')+1:__FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) return;
    fprintf(stderr,"CUDA error (%s:%d): %s: %s\n",
            file, line, msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
#endif
}

// Group3 Mod - PSNR reference frame storage
static bool firstFrame = true;
static std::vector<glm::vec3> referenceFrame;

// Group3 Mod - Performance metrics
struct PerformanceMetrics {
    float totalRenderTime   = 0.0f;
    float avgIterationTime  = 0.0f;
    float samplesPerSecond  = 0.0f;
    size_t gpuMemoryUsed    = 0;
    float lastPSNR          = 0.0f;
    int   iterationsToClean = -1;
    std::chrono::high_resolution_clock::time_point startTime;
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    void end(int iter, int pixelcount) {
        auto endTime = std::chrono::high_resolution_clock::now();
        float secs = std::chrono::duration<float>(endTime - startTime).count();
        totalRenderTime  += secs;
        avgIterationTime  = (totalRenderTime / iter) * 1000.0f;
        samplesPerSecond  = (pixelcount * float(iter)) / totalRenderTime;
    }
};
static PerformanceMetrics metrics;

void updateGpuMemory() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    metrics.gpuMemoryUsed = totalMem - freeMem;
}

float computePSNR(const std::vector<glm::vec3>& currentRaw, int iter) {
    std::vector<glm::vec3> current = currentRaw;
    for (auto& c : current) c /= float(iter);
    if (firstFrame && iter == 10) {
        referenceFrame = current;
        firstFrame = false;
        return FLT_MAX;
    }
    if (firstFrame) return FLT_MAX;
    double mse = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
        glm::vec3 d = current[i] - referenceFrame[i];
        mse += glm::dot(d, d);
    }
    mse /= (current.size() * 3.0);
    if (mse <= 1e-12) return FLT_MAX;
    return 10.0f * log10f(1.0f / float(mse));
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1<<31) | (depth<<22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Group3 Mod - Helper functions for improved lighting and reflections
__device__ glm::vec3 sampleHemisphere(float u1, float u2) {
    float r = sqrt(1.0f - u1 * u1);
    float phi = 2.0f * M_PI * u2;
    return glm::vec3(r * cos(phi), u1, r * sin(phi));
}

__device__ void createLocalCoordinateSystem(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent) {
    if (fabs(normal.x) > fabs(normal.y)) {
        tangent = glm::normalize(glm::vec3(normal.z, 0, -normal.x));
    } else {
        tangent = glm::normalize(glm::vec3(0, -normal.z, normal.y));
    }
    bitangent = glm::cross(normal, tangent);
}

__device__ glm::vec3 sampleCosineWeightedHemisphere(float u1, float u2, const glm::vec3& normal) {
    glm::vec3 tangent, bitangent;
    createLocalCoordinateSystem(normal, tangent, bitangent);
    
    // Cosine-weighted hemisphere sampling
    float theta = acos(sqrt(1.0f - u1));
    float phi = 2.0f * M_PI * u2;
    
    float x = sin(theta) * cos(phi);
    float y = cos(theta);
    float z = sin(theta) * sin(phi);
    
    return glm::normalize(tangent * x + normal * y + bitangent * z);
}

__device__ glm::vec3 reflect(const glm::vec3& incident, const glm::vec3& normal) {
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

__device__ float schlickFresnel(float cosTheta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
                               int iter, glm::vec3* image)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        int idx = x + y*resolution.x;
        glm::vec3 pix = image[idx] / float(iter);
        pix = glm::pow(pix, glm::vec3(1.0f/2.2f));
        glm::ivec3 col;
        col.x = glm::clamp(int(pix.x*255.0f), 0, 255);
        col.y = glm::clamp(int(pix.y*255.0f), 0, 255);
        col.z = glm::clamp(int(pix.z*255.0f), 0, 255);
        pbo[idx].w = 0;
        pbo[idx].x = col.x;
        pbo[idx].y = col.y;
        pbo[idx].z = col.z;
    }
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int idx = x + y*cam.resolution.x;
        PathSegment &seg = pathSegments[idx];
        seg.ray.origin = cam.position;
        seg.color = glm::vec3(1.0f);
        seg.ray.direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * (float(x) - cam.resolution.x*0.5f)
            - cam.up    * cam.pixelLength.y * (float(y) - cam.resolution.y*0.5f)
        );
        seg.pixelIndex = idx;
        seg.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth, int num_paths,
    PathSegment* pathSegments,
    Geom* geoms, int geoms_size,
    ShadeableIntersection* intersections,
    BVHNodeGPU* bvhNodes)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    Ray ray = pathSegments[idx].ray;
    float t_min = FLT_MAX;
    int   hitG  = -1;

    int stack[64], sp = 0;
    stack[sp++] = 0; // root node

    while (sp > 0) {
        BVHNodeGPU node = bvhNodes[stack[--sp]];
        if (!intersectAABB(node.bounds, ray)) continue;
        if (node.left < 0) {
            int g = node.geomIndex;
            glm::vec3 pt, nrm; bool out;
            float t = (geoms[g].type == CUBE)
                ? boxIntersectionTest(geoms[g], ray, pt, nrm, out)
                : sphereIntersectionTest(geoms[g], ray, pt, nrm, out);
            if (t > 0 && t < t_min) {
                t_min = t; hitG = g;
                intersections[idx].point         = pt;
                intersections[idx].surfaceNormal = nrm;
                intersections[idx].outsideObject = out;  // Group3 Mod - Track whether ray hit from outside
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }

    if (hitG < 0) {
        intersections[idx].t = -1.0f;
    } else {
        intersections[idx].t          = t_min;
        intersections[idx].materialId = geoms[hitG].materialid;
        intersections[idx].geomIndex  = hitG;  // Group3 Mod - Store geometry index
    }
}

// Group3 Mod - Improved physically-based shading kernel
__global__ void shadeAndExtendRays(
    int iter, int depth, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials, int materialCount,
    glm::vec3* lightPositions, int numLights)
{
    extern __shared__ Material sharedMat[];
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    // load materials into shared memory
    for (int i = threadIdx.x; i < materialCount && i < MAX_MATERIALS; i += blockDim.x)
        sharedMat[i] = materials[i];
    __syncthreads();

    const ShadeableIntersection hit = shadeableIntersections[idx];
    PathSegment &segment = pathSegments[idx];
    
    // For missed rays or depleted bounce count
    if (hit.t < 0.0f || segment.remainingBounces <= 0) {
        // Group3 Mod - Environment lighting
        if (hit.t < 0.0f) {
            // Simple sky/environment light contribution
            float t = 0.5f * (segment.ray.direction.y + 1.0f);
            glm::vec3 skyColor = (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
            segment.color *= skyColor * 0.5f;  // Dimmer sky for better contrast
        }
        segment.remainingBounces = 0;
        return;
    }
    
    auto rng = makeSeededRandomEngine(iter, idx, depth);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    
    Material material = sharedMat[hit.materialId];
    
    // Emissive surfaces (lights)
    if (material.emittance > 0.0f) {
        segment.color *= material.color * material.emittance;
        segment.remainingBounces = 0;  // terminate path at light sources
        return;
    }

    // Group3 Mod - Russian roulette path termination
    if (depth > 3) {  // Start Russian Roulette after a few bounces
        float continueProbability = fmaxf(material.color.x, fmaxf(material.color.y, material.color.z));
        if (u01(rng) > continueProbability) {
            segment.remainingBounces = 0;
            return;
        }
        segment.color /= continueProbability;  // Compensate for termination probability
    }

    glm::vec3 hitPoint = hit.point;
    glm::vec3 normal = hit.surfaceNormal;
    glm::vec3 viewDir = -segment.ray.direction;  // Direction toward camera
    
    // Group3 Mod - Material-based shading
    segment.remainingBounces--;
    
    // Choose between specular reflection and diffuse based on material properties
    // Using hasReflective as reflectivity strength and hasRefractive (inverted) as roughness
    float reflectivity = material.hasReflective;
    float roughness = 1.0f - material.hasRefractive;
    
    if (reflectivity > 0.0f && u01(rng) < reflectivity) {
        // Specular reflection (mirror-like)
        glm::vec3 reflectDir = reflect(segment.ray.direction, normal);
        
        // Add some roughness/perturbation if needed
        if (roughness > 0.0f) {
            glm::vec3 tangent, bitangent;
            createLocalCoordinateSystem(reflectDir, tangent, bitangent);
            float angle = roughness * u01(rng) * M_PI * 0.5f;
            float x = sin(angle) * cos(2.0f * M_PI * u01(rng));
            float y = cos(angle);
            float z = sin(angle) * sin(2.0f * M_PI * u01(rng));
            reflectDir = glm::normalize(tangent * x + reflectDir * y + bitangent * z);
        }
        
        // Set up next ray
        segment.ray.origin = hitPoint + normal * 0.001f;  // Offset to avoid self-intersection
        segment.ray.direction = reflectDir;
        
        // For pure reflection, maintain color but apply material color tint
        segment.color *= material.specular.color;  // Use specular color for reflections
    } else {
        // Diffuse reflection (Lambertian)
        float u1 = u01(rng);
        float u2 = u01(rng);
        
        glm::vec3 diffuseDir = sampleCosineWeightedHemisphere(u1, u2, normal);
        
        // Set up next ray
        segment.ray.origin = hitPoint + normal * 0.001f;
        segment.ray.direction = diffuseDir;
        
        // Apply material color for diffuse reflection
        segment.color *= material.color;
    }
}

__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* paths) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < nPaths) {
        image[paths[idx].pixelIndex] += paths[idx].color;
    }
}

static Scene*                    hst_scene      = nullptr;
static GuiDataContainer*         guiData        = nullptr;
static glm::vec3*                dev_image      = nullptr;
static Geom*                     dev_geoms      = nullptr;
static Material*                 dev_materials  = nullptr;
static PathSegment*              dev_paths      = nullptr;
static ShadeableIntersection*    dev_intersections = nullptr;

// Group3 Mod - Light positions for direct lighting
static glm::vec3*                dev_lightPositions = nullptr;
static int                       h_numLights = 0;

void InitDataContainer(GuiDataContainer* imGuiData) {
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera &cam = scene->state.camera;
    int pixelcount   = cam.resolution.x * cam.resolution.y;

    // image & paths
    cudaMalloc(&dev_image,  pixelcount*sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount*sizeof(glm::vec3));
    cudaMalloc(&dev_paths,  pixelcount*sizeof(PathSegment));

    // geoms
    int G = (int)scene->geoms.size();
    cudaMalloc(&dev_geoms,    G*sizeof(Geom));
    cudaMemcpy(dev_geoms,     scene->geoms.data(), G*sizeof(Geom), cudaMemcpyHostToDevice);

    // materials
    int M = (int)scene->materials.size();
    cudaMalloc(&dev_materials, M*sizeof(Material));
    cudaMemcpy(dev_materials,  scene->materials.data(), M*sizeof(Material), cudaMemcpyHostToDevice);

    // build & upload BVH
    {
        std::vector<BVHNodeGPU> h_bvh;
        buildBVH(scene->geoms, h_bvh);
        h_bvhNodeCount = (int)h_bvh.size();
        cudaMalloc(&dev_bvhNodes, h_bvhNodeCount*sizeof(BVHNodeGPU));
        cudaMemcpy(dev_bvhNodes, h_bvh.data(), h_bvhNodeCount*sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
    }

    // Group3 Mod - Find light sources for direct lighting
    std::vector<glm::vec3> lightPositions;
    for (size_t i = 0; i < scene->geoms.size(); ++i) {
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f) {
            // Extract center of geometry as light position
            glm::vec4 center = scene->geoms[i].transform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            lightPositions.push_back(glm::vec3(center));
        }
    }
    h_numLights = (int)lightPositions.size();
    if (h_numLights > 0) {
        cudaMalloc(&dev_lightPositions, h_numLights * sizeof(glm::vec3));
        cudaMemcpy(dev_lightPositions, lightPositions.data(), h_numLights * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }

    // intersections buffer
    cudaMalloc(&dev_intersections, pixelcount*sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount*sizeof(ShadeableIntersection));

    // timing events
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    updateGpuMemory();
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_bvhNodes);
    if (dev_lightPositions) cudaFree(dev_lightPositions);  // Group3 Mod
    checkCUDAError("pathtraceFree");
}

void pathtrace(uchar4* pbo, int frame, int iter) {
    const Camera &cam = hst_scene->state.camera;
    int pixelcount   = cam.resolution.x * cam.resolution.y;

    dim3 blockSize2d(8,8),
         blocks2d((cam.resolution.x+7)/8, (cam.resolution.y+7)/8);
    int blockSize1d = 128;

    metrics.start();

    float rayGenTime=0, intersectTime=0, shadeTime=0, gatherTime=0;
    float totalK = 0.0f;

    // Ray generation
    cudaEventRecord(startKernel);
    generateRayFromCamera<<<blocks2d,blockSize2d>>>(cam, iter, hst_scene->state.traceDepth, dev_paths);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&rayGenTime, startKernel, stopKernel);
    totalK += rayGenTime;
    checkCUDAError("generate camera ray");

    // Intersection & shading loop
    int depth = 0;
    int num_paths = pixelcount;
    bool iterationComplete = false;

    // Group3 Mod - Limit the number of path segments by active rays only
    int* dev_numActiveRays;
    cudaMalloc(&dev_numActiveRays, sizeof(int));
    cudaMemcpy(dev_numActiveRays, &pixelcount, sizeof(int), cudaMemcpyHostToDevice);

    while (!iterationComplete && depth < hst_scene->state.traceDepth) {
        cudaMemset(dev_intersections, 0, pixelcount*sizeof(ShadeableIntersection));
        int numBlocks1d = (num_paths + blockSize1d - 1) / blockSize1d;

        // BVH-based intersection
        cudaEventRecord(startKernel);
        computeIntersections<<<numBlocks1d,blockSize1d>>>(
            depth, num_paths,
            dev_paths,
            dev_geoms, (int)hst_scene->geoms.size(),
            dev_intersections,
            dev_bvhNodes
        );
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);
        cudaEventElapsedTime(&intersectTime, startKernel, stopKernel);
        totalK += intersectTime;
        checkCUDAError("trace one bounce");

        // Group3 Mod - Physically based shading
        cudaEventRecord(startKernel);
        int matCount = (int)hst_scene->materials.size();
        size_t shMemBytes = matCount * sizeof(Material);
        shadeAndExtendRays<<<numBlocks1d,blockSize1d,shMemBytes>>>(
            iter, depth, num_paths,
            dev_intersections,
            dev_paths,
            dev_materials, matCount,
            dev_lightPositions, h_numLights
        );
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);
        cudaEventElapsedTime(&shadeTime, startKernel, stopKernel);
        totalK += shadeTime;
        checkCUDAError("shade");

        depth++;
        
        // Group3 Mod - Exit condition based on maximum depth reached
        if (depth >= hst_scene->state.traceDepth) {
            iterationComplete = true;
        }
    }
    
    cudaFree(dev_numActiveRays);  // Group3 Mod - Cleanup

    // Final gather
    int numBlocksPix = (pixelcount + blockSize1d - 1) / blockSize1d;
    cudaEventRecord(startKernel);
    finalGather<<<numBlocksPix,blockSize1d>>>(pixelcount, dev_image, dev_paths);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&gatherTime, startKernel, stopKernel);
    totalK += gatherTime;
    checkCUDAError("finalGather");

    // Display
    sendImageToPBO<<<blocks2d,blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("sendImageToPBO");

    // PSNR & metrics
    std::vector<glm::vec3> current(pixelcount);
    cudaMemcpy(current.data(), dev_image, pixelcount*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    metrics.end(iter, pixelcount);
    updateGpuMemory();
    float psnr = computePSNR(current, iter);
    metrics.lastPSNR = psnr;
    if (psnr > 35.0f && metrics.iterationsToClean < 0)
        metrics.iterationsToClean = iter;

    printf("\n====== PERFORMANCE METRICS SUMMARY ======\n");
    printf("Total render time: %.2f seconds\n", metrics.totalRenderTime);
    printf("Avg iteration time: %.2f ms\n",    metrics.avgIterationTime);
    printf("Samples per second: %.2f million rays/s\n", metrics.samplesPerSecond/1e6f);
    printf("GPU memory used: %.2f MB\n", metrics.gpuMemoryUsed / float(1<<20));
    if (psnr == FLT_MAX)    printf("PSNR: Inf dB\n");
    else                    printf("PSNR: %.2f dB\n", psnr);
    if (metrics.iterationsToClean > 0)
        printf("Iterations to clean: %d\n", metrics.iterationsToClean);
    printf("Total kernel time: %.2f ms\n", totalK);
    printf("  - Ray generation:   %.2f ms\n", rayGenTime);
    printf("  - Intersection:     %.2f ms\n", intersectTime);
    printf("  - Shading:          %.2f ms\n", shadeTime);
    printf("  - Final gather:     %.2f ms\n", gatherTime);
    printf("=========================================\n");

    cudaMemcpy(hst_scene->state.image.data(),
               dev_image,
               pixelcount*sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}
