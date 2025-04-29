# CMSE/CSE 822 Project 4 final report: GPU Computing 

## **Serial Implementation**
### **Step 2,3,4,5, and 6: Declaring Constants and Vectors**
In these steps, I implemented the jacobi solver exactly as explained in the instructions. My code is in the file "jacobi_solver_serial.cpp". I also completed these two lines:

$xnew[i] += A[i * Ndim + j] * xold[j]$

and 

$xnew[i] = (b[i] - xnew[i]) / A[i * Ndim + i]$

### **the results**

For compilation, I used the command below:

g++ -std=c++20 -fopenmp jacobi_solver_serial.cpp -o jacobi_solver

I used the default dimension 1000 for grids:

This is my output:

Converged after 4587 iterations in 23.2966 seconds with final convergence = 0.00099889
Solution verification: Error = 0.000999865, Checksum = 123.595

So, it took 23.2966 seconds for the serial method to find the answer.


## **Parallelization with OpenMP**

Starting from your serial Jacobi solver, we will now proceed with implementing parallelization via OpenMP for both the CPU and GPU. 

### **Step 1: Parallelization with OpenMP(CPU)**

I followed the steps on the Readme.md, and added the required pragmas for OpenMP. My code is in "jacobi_solver_parallel_cpu.cpp". I compiled the code using the command below:

g++ -std=c++20 -fopenmp jacobi_solver_parallel_cpu.cpp -o jacobi_solver

With the dimension 1000, here is my output:

Converged after 4565 iterations in 14.9034 seconds with final convergence = 0.00099826
Solution verification: Error = 0.00099924, Checksum = 123.665

So, it took 14.9034 seconds, which is considerably lower than the serial method (23.2966).


### **Step 2: Parallelization with OpenMP(GPU)-First step**

I followed the steps on the Readme.md, and added the required pragmas for OpenMP. My code is in "jacobi_solver_parallel_gpu.cpp". I compiled the code using the commands below:

module purge
module load NVHPC/23.7-CUDA-12.1.1
nvc++ -std=c++20 -mp=gpu -gpu=managed jacobi_solver_parallel_gpu.cpp -o jacobi_solver

With the dimension 1000, here is my output:

Converged after 4473 iterations in 5.76771 seconds with final convergence = 0.000998292
Solution verification: Error = 0.000999293, Checksum = 123.593

So, it took 5.76771 seconds, which is considerably improvement comparing to the last step with CPU (it is around 1/3 of that). 

### **Step 3: Parallelization with OpenMP(GPU)-Second step-data regions (branchless)**
I followed the steps on the Readme.md, and added the required pragmas for OpenMP. My code is in "jacobi_solver_parallel_gpu_dataregions.cpp". I compiled the code using the commands below:

module purge
module load NVHPC/23.7-CUDA-12.1.1
nvc++ -std=c++20 -mp=gpu -gpu=managed jacobi_solver_parallel_gpu_dataregions.cpp -o jacobi_solver

With the dimension 1000, here is my output:

Converged after 4564 iterations in 5.09002 seconds with final convergence = 0.000998575
Solution verification: Error = 0.000999555, Checksum = 123.593

So, it took 5.09002 seconds, which is slightly improved comparing to the last step with GPU (5.76771). 




### **Step 4: Parallelization with OpenMP(GPU)-Third step-Coalesced memory accesses**

I followed the steps on the Readme.md, and added the required pragmas for OpenMP. My code is in "jacobi_solver_parallel_gpu_coalesced.cpp". I compiled the code using the commands below:

module purge
module load NVHPC/23.7-CUDA-12.1.1
nvc++ -std=c++20 -mp=gpu -gpu=managed jacobi_solver_parallel_gpu_coalesced.cpp -o jacobi_solver

I tried multiple times, but always the error exceeded the margins for this case. Btw, here is one of my best outputs here:

Converged after 4562 iterations in 4.00466 seconds with final convergence = 0.000999583
Solution verification: Error = 0.143195, Checksum = 123.631
WARNING: Solution error exceeds tolerance!

It looks like in case that it can find the answer successfully, there is a considerable decrease in the running time cause even with some error the time decreased considerably comparing to the previous methods. 

Summarize your optimizations and clearly state each step’s impact on runtime:

| Optimization Step             | Typical Performance Impact  |
|-------------------------------|-----------------------------|
| Initial parallelization       | Poor (expected) |
| Persistent data regions       | Large improvement (expected) |
| Branchless implementation     | Moderate improvement (expected) |
| Coalesced memory accesses     | Significant improvement (unknown due to the error) |

