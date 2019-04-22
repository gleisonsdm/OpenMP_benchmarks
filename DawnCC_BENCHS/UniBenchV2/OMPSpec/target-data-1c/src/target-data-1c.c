// ----------------------------------------------------------------------------------------
// Implementation of Example target.3c (Section 52.3, page 196) from Openmp
// 4.0.2 Examples
// on the document http://openmp.org/mp-documents/openmp-examples-4.0.2.pdf
//
//
//
//
// ----------------------------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define N 8192

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE *A, DATA_TYPE *B) {
  int i;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 8192))
  || ((void*) (B + 0) > (void*) (A + 8192)));
  #pragma omp target data map(tofrom: A[0:8192],B[0:8192]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    A[i] = i / 2.0;
    B[i] = ((N - 1) - i) / 3.0;
  }

  return;
}

void vec_mult(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 8192))
  || ((void*) (B + 0) > (void*) (A + 8192)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 8192))
  || ((void*) (C + 0) > (void*) (A + 8192)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 8192))
  || ((void*) (C + 0) > (void*) (B + 8192)));
  #pragma omp target data map(to: A[0:8192],B[0:8192]) map(tofrom: C[0:8192]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++)
    C[i] = A[i] * B[i];
}

void vec_mult_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i;

#pragma omp target data map(to : A[ : N], B[ : N]) map(from : C[ : N])         \
                                device(GPU_DEVICE)
  {
#pragma target
#pragma omp parallel for
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 8192))
    || ((void*) (B + 0) > (void*) (A + 8192)));
    RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 8192))
    || ((void*) (C + 0) > (void*) (A + 8192)));
    RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 8192))
    || ((void*) (C + 0) > (void*) (B + 8192)));
    #pragma omp target data map(to: A[0:8192],B[0:8192]) map(tofrom: C[0:8192]) if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < N; i++)
      C[i] = A[i] * B[i];
  }
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_GPU) {
  int i, fail;
  fail = 0;

  // Compare B and B_GPU
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (B + 0) > (void*) (B_GPU + 8192))
  || ((void*) (B_GPU + 0) > (void*) (B + 8192)));
  #pragma omp target data map(to: B[0:8192],B_GPU[0:8192])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < N; i++) {
    if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  double t_start, t_end, t_start_OMP, t_end_OMP;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *C_OMP;

  A = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  C_OMP = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two vector multiplication <<\n");

  // initialize the arrays
  init(A, B);

  t_start_OMP = rtclock();
  vec_mult_OMP(A, B, C_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP); //);

  t_start = rtclock();
  vec_mult(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); //);

  compareResults(C, C_OMP);

  free(A);
  free(B);
  free(C);
  free(C_OMP);

  return 0;
}

