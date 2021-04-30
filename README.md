# Travelling Salesman CUDA

`travelling-salesman-cuda` is a thought experiement in attempting to solve the [Travelling Salesman Probelm](https://en.wikipedia.org/wiki/Travelling_salesman_problem) using [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) to efficiently divide-and-conquer the problem for large numbers of vertices.

As per the Travelling Salesman Problem, an undirected graph is given where the question is asked:

> Given a list of [vertices] and the distances between each pair of [vertices], what is the shortest possible route that visits each [vertex] exactly once and returns to the origin [vertex]?

Knowing the number of vertices V in the graph G=<V,E>, the number of possible candidate routes in the graph is n! (n-factorial) where n is the number of vertices in V.

For example, if the number of vertices in the graph given for the Travelling Salesman Problem is 7, the total number of candidate solutions that must be examined in a brute-force examination of all candidate solutions is 360 permutations. Given 15 vertices, the number of candidate solutions is 1.307674368x10^12 (over 1 trillion candidate solutions).

To efficiently solve the Travelling Salesman Problem, a heuristic method of examining the candidate solutions must be used to avoid allocating resources for a factorial number of candidate solutions as the number of vertices increases in size.

CUDA offers a wonderful solution to parallelizing the calculations for the Travelling Salesman Problem. Instead of running all calculations on a single CPU or being limited to parallelization based on the number of threads available to a CPU (for example, an [AMD Threadripper](https://www.amd.com/en/products/ryzen-threadripper) has at least 128 threads to be used), GPU programming allows for using 1024 threads per block. This allows for significantly increased performance in tackling the Travelling Salesman Problem by dividing candidate solutions onto the GPU device and calculating minimal distance circuit(s) in the graph with an increased number of threads and GDDR memory.

`travelling-salesman-cuda` looks to solve the Traveling Salesman Problem for an input graph set of 13 vertices, or 6 billion 227 million 20 thousand and 800 candidate solutions in the smallest amount of time possible.

This implementation was devised in comparison to the TSP Solution I crafted in the Algorithm Techniques class at Florida International University during the Spring semester of 2021. My implementation of solving the Travelling Salesman Problem using Java in that course took over 23 minutes on a Core i9-7960X to execute a brute-force search to find the optimal solution to the question on the same size of input information. I was unable to devise a method to decrease the execution time of the brute-force search during the course of the class time.

## Optimizations
Based on the compiler used alongside NVCC, this solution opts to use the fastest data types available.
