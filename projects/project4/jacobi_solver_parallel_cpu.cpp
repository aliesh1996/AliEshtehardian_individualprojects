#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "mm_utils.hpp"

// Constants
constexpr double TOLERANCE = 0.001;
constexpr int DEF_SIZE = 1000;
constexpr int MAX_ITERS = 100000;
constexpr double LARGE = 1000000.0;

int main(int argc, char **argv) {
    int Ndim = (argc == 2) ? std::atoi(argv[1]) : DEF_SIZE;
    std::cout << "Matrix dimension (Ndim) = " << Ndim << std::endl;

    // Use std::vector to allocate storage dynamically.
    std::vector<TYPE> A(Ndim * Ndim);
    std::vector<TYPE> b(Ndim);
    std::vector<TYPE> xnew(Ndim, 0.0);
    std::vector<TYPE> xold(Ndim, 0.0);

    // Generate diagonally dominant matrix A
    initDiagDomNearIdentityMatrix(Ndim, A.data());

    // Initialize b with random values (between 0.0 and 0.5)
    for (int i = 0; i < Ndim; ++i) {
        b[i] = static_cast<TYPE>(std::rand() % 51) / 100.0;
    }

    // Create raw pointers for OpenMP target mapping
    TYPE* A_ptr = A.data();
    TYPE* b_ptr = b.data();
    TYPE* xnew_ptr = xnew.data();
    TYPE* xold_ptr = xold.data();

    double start_time = omp_get_wtime();
    TYPE conv = LARGE;
    int iters = 0;

    // Parallel implementation
    while ((conv > TOLERANCE) && (iters < MAX_ITERS)) {
        ++iters;

        // Compute new iteration
        #pragma omp target map(tofrom: xnew_ptr[0:Ndim], xold_ptr[0:Ndim]) \
                           map(to: A_ptr[0:Ndim*Ndim], b_ptr[0:Ndim])
        #pragma omp loop
        for (int i = 0; i < Ndim; ++i) {
            xnew_ptr[i] = 0.0;
            for (int j = 0; j < Ndim; ++j) {
                if (i != j)
                    xnew_ptr[i] += A_ptr[i * Ndim + j] * xold_ptr[j];
            }
            xnew_ptr[i] = (b_ptr[i] - xnew_ptr[i]) / A_ptr[i * Ndim + i];
        }

        // Compute convergence criterion (Euclidean norm of difference)
        conv = 0.0;
        #pragma omp target map(to: xnew_ptr[0:Ndim], xold_ptr[0:Ndim]) \
                           map(tofrom: conv)
        #pragma omp loop reduction(+:conv)
        for (int i = 0; i < Ndim; ++i) {
            TYPE tmp = xnew_ptr[i] - xold_ptr[i];
            conv += tmp * tmp;
        }
        conv = std::sqrt(conv);

        // Swap pointers for next iteration
        std::swap(xold_ptr, xnew_ptr);
    }

    double elapsed_time = omp_get_wtime() - start_time;
    std::cout << "Converged after " << iters << " iterations in "
              << elapsed_time << " seconds with final convergence = "
              << conv << std::endl;

    TYPE err = 0.0, chksum = 0.0;

    for (int i = 0; i < Ndim; ++i) {
        xold_ptr[i] = static_cast<TYPE>(0.0);
        for (int j = 0; j < Ndim; ++j)
            xold_ptr[i] += A_ptr[i * Ndim + j] * xnew_ptr[j];

        TYPE diff = xold_ptr[i] - b_ptr[i];
        chksum += xnew_ptr[i];
        err += diff * diff;
    }
    err = static_cast<TYPE>(std::sqrt(err));

    std::cout << "Solution verification: Error = " << err
              << ", Checksum = " << chksum << std::endl;

    if (err > TOLERANCE)
        std::cout << "WARNING: Solution error exceeds tolerance!" << std::endl;

    return 0;
}