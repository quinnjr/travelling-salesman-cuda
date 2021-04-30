#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>

#include "cuda_runtime.h"

enum TSPError {
  E_MEMORY_ALLOCATION,
  E_BAD_DEVICE
};

#define checkCudaError(input) { gpuAssert((input), __FILE__, __LINE__); }
__host__ inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if(code != cudaSuccess) {
    const char* errorMessage = cudaGetErrorString(code);
    fprintf(stderr, "CUDA Error: %s %s %s\n", errorMessage, file, line);
    exit(code);
  }
}

#define checkCudaErrorDevice(input) { gpuDeviceAssert((input), __FILE__, __LINE__); }
__device__ void gpuDeviceAssert(cudaError_t code, const char* file, int line) {
  if(code != cudaSuccess) {
    const char* errorMessage = cudaGetErrorString(code);
    printf("GPU kernel assert: %s %s %d\n", errorMessage, file, line);
    assert(0);
  }
}

struct Lock {
  int* dState;

  Lock(void) {
    int hState = 0;
    checkCudaError(cudaMalloc(&dState, sizeof(int)));
    checkCudaError(cudaMemcpy(dState, &hState, sizeof(int), cudaMemcpyHostToDevice));
  }

  __host__ __device__ ~Lock(void) {
#ifndef __CUDACC__
      checkCudaError(cudaFree(dState));
#endif
  }

  __device__ void lock(void) { while(atomicCAS(dState, 0, 1) != 0); }

  __device__ void unlock(void) { atomicExch(dState, 0); }
};

class Vertex {
public:
  int x, y;
  Vertex(int cx, int cy);
  friend std::ostream &operator<<(std::ostream&, const Vertex&);
};

Vertex::Vertex(int cx, int cy) : x(cx), y(cy) {}

std::ostream &operator<<(std::ostream &stream, const Vertex& v) {
  return stream << "(" << v.x << ", " << v.y << ")";
}

class PermutationNeighborhood {
private:
  size_t size;
  int* p;
  int loc1;
  int loc2;
public:
  __device__ PermutationNeighborhood(int*& input) {
    this->size = (sizeof(input) / sizeof(*input));

    this->p = (int*) malloc(this->size * sizeof(int));

    this->loc1 = 0;
    this->loc2 = 1;
  }

  __device__ ~PermutationNeighborhood() {
    cudaFree(this->p);
  }

  __device__ bool has_next();
  __device__ int* next();
};

__device__ bool PermutationNeighborhood::has_next() {
  return (this->loc1 != (this->size - 1));
}

__device__ int* PermutationNeighborhood::next() {
  if(this->has_next()) {
    int* a = (int*) malloc(this->size);
    memcpy(a, this->p, this->size * sizeof(int));

    a[this->loc1] = this->p[this->loc2];
    a[this->loc2] = this->p[this->loc2];

    if(this->loc2 == this->size - 1) {
      this->loc1 = this->loc1 + 1;
      this->loc2 = this->loc1 + 1;
    } else {
      this->loc2 = this->loc2 + 1;
    }

    return a;
  } else {
    return NULL;
  }
}

class Graph {
  size_t vertices;
  int* matrix;
public:
  __host__ Graph(size_t sz) {
    this->vertices = sz;
    int* m;

    checkCudaError(cudaMallocManaged(&m, sz * sz * sizeof(int)));
    this->matrix = m;
  }

  __host__ ~Graph(void) {
    checkCudaError(cudaFree(this->matrix));
  }

  __host__ void addEdge(const int, const int, const int);
  __host__ __device__ int calculateDistance(int*&);
  __device__ int localSearch(int*&);
};

__host__ void Graph::addEdge(const int u, const int v, const int w) {
  this->matrix[u * this->vertices + v] = w;
  this->matrix[v * this->vertices + u] = w;
}

__host__ __device__ int Graph::calculateDistance(int*& input) {
  int totalWeight = 0;

  for(auto i = 0; i < this->vertices; i++) {
    int weight = this->matrix[input[i] * this->vertices + input[(i + 1) % this->vertices]];
    totalWeight += weight;
  }

  return totalWeight;
}

__device__ int Graph::localSearch(int*& shortestRoute) {
  int bestDistance;
  bool betterSolutionFound;

  bestDistance = this->calculateDistance(shortestRoute);

  do {
    betterSolutionFound = false;

    PermutationNeighborhood* pn = new PermutationNeighborhood(shortestRoute);

    while(pn->has_next()) {
      int* a = pn->next();
      int currentDistance = this->calculateDistance(a);
      if(currentDistance < bestDistance) {
        shortestRoute = a;
        bestDistance = currentDistance;
        betterSolutionFound = true;
      }
    }
  } while(betterSolutionFound);

  return bestDistance;
}



template<typename T>
__host__ T* constructCandidatePath(T start, size_t nVertices) {
  T* path;
  cudaMallocManaged(&path, nVertices * sizeof(T));
  path[0] = start;

  for(uint_fast8_t i = 1; i < nVertices; i++) {
    if(i != start) {
      path[i] = i;
    } else {
      path[i] = 0;
    }
  }

  return path;
}

/**
 *
 */
__global__ void minimumDistance(Lock lock, Graph* graph, int** paths, const int nVertices) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pathIndex = index % nVertices;

  int* candidatePath = paths[pathIndex];

  int shortestLocalDistance = graph->localSearch(candidatePath);

  lock.lock();
  if(shortestLocalDistance < graph->calculateDistance(paths[pathIndex])) {
    paths[pathIndex] = candidatePath;
  }
  lock.unlock();
}

/**
 *
 */
int main(int argc, char** argv) {
  std::ifstream inputFile("input.csv"); // Input graph file. Currently hard-coded.
  std::ofstream outputFile("output.csv");
  size_t nVertices = 0;
  cudaDeviceProp prop;
  Lock lock;
  int deviceId;
  int** paths;
  std::vector<Vertex*> vertices;
  std::string line, sx, sy;
  int cx, cy;

  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&prop, deviceId);

  if(!prop.unifiedAddressing) {
    fprintf(stderr, "GPU does not support Unified Addressing. Exiting...");
    exit(E_BAD_DEVICE);
  }

  // Read the rest of the graph file into the application.
  while(!inputFile.eof()) {

    std::getline(inputFile, line);

    sx = line.substr(0, line.find(","));
    sy = line.substr(line.find(",") + 1, line.length());

    cx = std::stoi(sx);
    cy = std::stoi(sy);

    Vertex* v = new Vertex(cx, cy);

    vertices.push_back(v);
    nVertices++;
  }

  inputFile.close();

  Graph* graph = new Graph(nVertices);

  // Construct the adjacency matrix of the Graph instance.
  for(auto i = 0; i < vertices.size() - 1; i++) {
    Vertex* v1 = vertices[i];

    for(auto j = 0; j < vertices.size() - 1; j++) {
      Vertex* v2 = vertices[j];

      int weight = (int) sqrt(pow(v2->x - v1->x, 2) + pow(v2->y - v1->y, 2));

      graph->addEdge(i, j, weight);
    }
  }

  size_t pathsSize = nVertices * sizeof(int*);
  checkCudaError(cudaMallocManaged(&paths, pathsSize));

  for(auto i = 0; i < nVertices; i++) {
    int* candidate = constructCandidatePath(i, nVertices);
    paths[i] = candidate;
  }

  checkCudaError(cudaMemPrefetchAsync(paths, pathsSize, deviceId));

  dim3 nBlocks(nVertices);
  dim3 nThreads(256);

  // minimumDistance<<<nBlocks, nThreads>>>(lock, graph, paths, nVertices);

  checkCudaError(cudaGetLastError());

  checkCudaError(cudaDeviceSynchronize());

  checkCudaError(cudaMemPrefetchAsync(paths, pathsSize, cudaCpuDeviceId));

  int* bestPath = paths[0];
  int bestDistance = graph->calculateDistance(bestPath);

  for(auto i = 0; i < pathsSize; i++) {
    int newDistance = graph->calculateDistance(paths[i]);
    if(newDistance < bestDistance) {
      bestPath = paths[i];
      bestDistance = newDistance;
    }
  }

  std::cout << "The best calculdated path is: ";

  for(auto i = 0; i < nVertices; i++) {
    std::cout << bestPath[i] + " ";
  }

  std::cout << "with a total distance traveled of " << bestDistance << std::endl;

  checkCudaError(cudaFree(paths));
  delete graph;

  outputFile.close();

  return EXIT_SUCCESS;
}
