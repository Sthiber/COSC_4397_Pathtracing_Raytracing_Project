#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Group3 Mod - Added timing variable declarations for performance measurement
extern cudaEvent_t startPathTrace, stopPathTrace;
extern cudaEvent_t startIntersection, stopIntersection;
extern cudaEvent_t startShading, stopShading;
extern cudaEvent_t startStreamCompaction, stopStreamCompaction;
extern float totalTime, intersectionTime, shadingTime, streamCompactionTime;