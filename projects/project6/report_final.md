# CMSE/CSE 822 Final Project (Seyed Ali Eshtehardian)

## **Project Description
In this project, I tried to parallelize a subset of my own research, calculating the appropriate strength of the selection in the iterated Prisoners Dilemma with my preferred settings, and measure how much it can affect my runtime. In other words, my goal was to extend this parallelization to my whole research, in case that it is successful. So, I used this experiment as a test-bed to see how it works.

## **Explanation of my Research
In the evolution of the agents in the iterated Prisoners Dilemma, according to Queller's "inclusive fitness" theory, cooperation can evolve in some certain conditions. However, cause it uses genotypes and phenotypes of the player and its opponent, it cannot be generalized. Here, we come up with a new probabilistic idea that if it works, we are able to generalize a new law for the evolution of cooperation here. We are looking for showing that experimentally when $I(X;Y)/H(X)>c/b$, where I is the information that is exchanged between the player $X$ and its opponent $Y$, H is the Shannon entropy of the genes of the player $X$, and $c/b$ is the cost over benefit. 

Here, what I did was calculating the appropriate K (strength of the selection) in my evolution process for one of my experiments using both serial and parallel (CPU and GPU) methods, and calculate how much parallelization can affect my runtime. Cause I'm using a very similar code (with almost the same time complexity) for my other experiments, my assumption is that in case that parallelization can help me with this part, it will help me with other parts of my project as well.

Here, I'm explaining the parts of my code where I parallelized (loops), section by section, and I uploaded my code with comments and also I sent you my poster, where I explained more about its theoritical side. Finally, I will calculate an estimation of the run times for every run when I used serial, parallel(CPU), and parallel(GPU), and then decide how well is that.

## **Code Explantion**
My codes are uploaded and commented with IPDN3_serial.cpp and IPDN3_parallel.cpp. I compiled IPDN3_serial using the command below (on hpcc):

### Serial:
```bash
g++ IPDN3_serial.cpp -o IPDN3_serial
```
### Parallel (CPU only):
```bash
g++ -std=c++20 -fopenmp -O3 -o IPDN3_parallel IPDN3_parallel.cpp
```
### GPU with NVIDIA HPC SDK:
```bash
nvc++ -std=c++20 -mp=gpu -O3 -o IPDN3_parallel IPDN3_parallel.cpp
```


# Parallelization Report for `IPDN3.cpp`

## Overview
The file `IPDN3.cpp` implements a simulation of agent-based interactions in an Iterated Prisoner's Dilemma with noise and evolutionary dynamics. The main goal of the parallelization was to leverage multi-core CPUs and GPUs (where supported) using OpenMP to speed up long-running simulations.


## Parallelization Strategy
The simulation is compute-heavy with independent operations across agents and generations. This makes it highly amenable to parallelism.

### Key Parallel Regions

#### 1. Agent-Game Interactions
Each agent plays with its neighbors:
```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int n = 0; n < popSize; n++) {
    for (int k = 0; k < neighbours; k++) {
        population[n]->play(population[(n + k + 1) % popSize], nr[m]);
    }
}
```

#### 2. Score and Strategy Statistics
Reduction is used to safely compute aggregate statistics:
```cpp
#pragma omp parallel for reduction(+:meanPlay[:2])
for (int n = 0; n < popSize; n++) {
    for (int i = 0; i < 2; i++) {
        meanPlay[i] += population[n]->pi[i];
    }
}
```
Similar reductions are applied to variables `x1`, `x2`, `x3`, `x4`, `x5`.

#### 3. Offspring Creation (Selection Phase)
```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < popSize; i++) {
    // selection logic
    #pragma omp critical
    nextGen.push_back(make_shared<Agent>(population[who], my));
}
```
*Note*: `critical` ensures thread-safe access to the `nextGen` vector.

---

## GPU Offloading (Optional)
To support GPU parallelism using `nvc++`, the following changes are needed:
- Replace `shared_ptr<Agent>` with a flat data structure (e.g., `struct AgentData`)
- Avoid dynamic memory and STL containers inside `#pragma omp target` regions

Example:
```cpp
#pragma omp target teams distribute parallel for map(tofrom: agents[0:popSize])
for (int i = 0; i < popSize; ++i) {
    // compute fitness or simulate games
}
```

---

## Timing Code
To measure total runtime:
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// simulation loop

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
cout << "Total Execution Time: " << elapsed.count() << " seconds" << endl;
```

---

## Conclusion
The `IPDN3.cpp` simulation has been successfully parallelized for multi-core CPUs using OpenMP, with optional GPU support enabled via `nvc++`. These changes are expected to significantly reduce runtime for large simulations, especially with high population sizes and many generations.

Further performance improvements can be achieved by refactoring the data structures for full GPU compatibility.



### **Results**

Here, I added the average runtime for 5 runs for each part:

| Optimization Step             | Estimated average time takes (per run)  |
|-------------------------------|-----------------------------|
| Serial                        | 3309.5 seconds |
| Parallel(CPU)                 | 761.99 seconds |
| Parallel (GPU)                | 133.49 seconds |

Also, I uploaded a couple of diagrams, one of them is the output for serial (serial.jpg) and another one is for parallel with cpu (parallel.png) to show that there is not such difference between the outputs for both. 


