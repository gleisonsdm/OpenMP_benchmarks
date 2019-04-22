/**
 * syr2k.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define N 2048
#define M 2048

#define GPU_DEVICE 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *C_GPU) {
  int i, j;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 4194304))
  || ((void*) (B + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C_GPU + 4194304))
  || ((void*) (C_GPU + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (B + 4194304)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C_GPU + 4194304))
  || ((void*) (C_GPU + 0) > (void*) (B + 4194304)));
  RST_AI1 |= !(((void*) (C + 0) > (void*) (C_GPU + 4194304))
  || ((void*) (C_GPU + 0) > (void*) (C + 4194304)));
  #pragma omp target data map(tofrom: A[0:4194304],B[0:4194304],C[0:4194304],C_GPU[0:4194304]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
      C_GPU[i * N + j] = C[i * N + j];
    }

    for (j = 0; j < M; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
      B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
    }
  }
}

void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 4194304))
  || ((void*) (B + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (B + 4194304)));
  #pragma omp target data map(tofrom: A[0:4194304],B[0:4194304],C[0:4194304]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (A + 0) > (void*) (B + 4194304))
  || ((void*) (B + 0) > (void*) (A + 4194304)));
  RST_AI2 |= !(((void*) (A + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (A + 4194304)));
  RST_AI2 |= !(((void*) (B + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (B + 4194304)));
  #pragma omp target data map(to: A[0:4194304],B[0:4194304]) map(tofrom: C[0:4194304]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void syr2k_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : A[ : N *M], B[ : N *M]) map(tofrom : C[ : N *N])
#pragma omp parallel for collapse(2)
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 4194304))
  || ((void*) (B + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (A + 4194304)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 4194304))
  || ((void*) (C + 0) > (void*) (B + 4194304)));
  #pragma omp target data map(to: A[0:4194304],B[0:4194304]) map(tofrom: C[0:4194304]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k] +
                        ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (C + 0) > (void*) (C_outputFromGpu + 4194304))
  || ((void*) (C_outputFromGpu + 0) > (void*) (C + 4194304)));
  #pragma omp target data map(to: C[0:4194304],C_outputFromGpu[0:4194304])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(C[i * N + j], C_outputFromGpu[i * N + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *C_outputFromGpu;

  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-2k operations >>\n");

  init_arrays(A, B, C, C_outputFromGpu);

  t_start = rtclock();
  syr2k_OMP(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  syr2k(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(C_outputFromGpu);

  return 0;
}

