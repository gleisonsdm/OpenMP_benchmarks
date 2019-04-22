/*
   This program performs matrix sum on the GPU with
   dynamically allocated matrices.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 2.0

    Run:
    ipmacc mat-sum_gpu.c -o mat
    ./mat matrix-size
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01

// Initialize matrices.
void init(float *a, float *b, float *c_cpu, float *c_gpu) {
  int i, j;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (a + 0) > (void*) (b + 1000000))
  || ((void*) (b + 0) > (void*) (a + 1000000)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (c_cpu + 1000000))
  || ((void*) (c_cpu + 0) > (void*) (a + 1000000)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (c_gpu + 1000000))
  || ((void*) (c_gpu + 0) > (void*) (a + 1000000)));
  RST_AI1 |= !(((void*) (b + 0) > (void*) (c_cpu + 1000000))
  || ((void*) (c_cpu + 0) > (void*) (b + 1000000)));
  RST_AI1 |= !(((void*) (b + 0) > (void*) (c_gpu + 1000000))
  || ((void*) (c_gpu + 0) > (void*) (b + 1000000)));
  RST_AI1 |= !(((void*) (c_cpu + 0) > (void*) (c_gpu + 1000000))
  || ((void*) (c_gpu + 0) > (void*) (c_cpu + 1000000)));
  #pragma omp target data map(tofrom: a[0:1000000],b[0:1000000],c_cpu[0:1000000],c_gpu[0:1000000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i * SIZE + j] = (float)i + j;
      b[i * SIZE + j] = (float)i + j;
      c_cpu[i * SIZE + j] = 0.0f;
      c_gpu[i * SIZE + j] = 0.0f;
    }
  }
}

/// matrix sum algorithm GPU
/// s = size of matrix
void sum_GPU(float *a, float *b, float *c) {
  int i, j;

#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : a[0 : SIZE *SIZE], b[0 : SIZE *SIZE])              \
                           map(tofrom : c[0 : SIZE *SIZE])
  {
#pragma omp parallel for collapse(2)
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (a + 0) > (void*) (b + 1000000))
    || ((void*) (b + 0) > (void*) (a + 1000000)));
    RST_AI1 |= !(((void*) (a + 0) > (void*) (c + 1000000))
    || ((void*) (c + 0) > (void*) (a + 1000000)));
    RST_AI1 |= !(((void*) (b + 0) > (void*) (c + 1000000))
    || ((void*) (c + 0) > (void*) (b + 1000000)));
    #pragma omp target data map(to: a[0:1000000],b[0:1000000]) map(tofrom: c[0:1000000]) if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < SIZE; ++i) {
      for (j = 0; j < SIZE; ++j) {
        c[i * SIZE + j] = a[i * SIZE + j] + b[i * SIZE + j];
      }
    }
  }
}

/// matrix sum algorithm CPU
/// s = size of matrix
void sum_CPU(float *a, float *b, float *c) {
  int i, j;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (a + 0) > (void*) (b + 1000000))
  || ((void*) (b + 0) > (void*) (a + 1000000)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (c + 1000000))
  || ((void*) (c + 0) > (void*) (a + 1000000)));
  RST_AI1 |= !(((void*) (b + 0) > (void*) (c + 1000000))
  || ((void*) (c + 0) > (void*) (b + 1000000)));
  #pragma omp target data map(to: a[0:1000000],b[0:1000000]) map(tofrom: c[0:1000000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      c[i * SIZE + j] = a[i * SIZE + j] + b[i * SIZE + j];
    }
  }
}

void compareResults(float *b_cpu, float *b_gpu) {
  int i, j, fail;
  fail = 0;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (b_cpu + 0) > (void*) (b_gpu + 1000000))
  || ((void*) (b_gpu + 0) > (void*) (b_cpu + 1000000)));
  #pragma omp target data map(to: b_cpu[0:1000000],b_gpu[0:1000000])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(b_cpu[i * SIZE + j], b_gpu[i * SIZE + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  double t_start, t_end;
  float *a, *b, *c_cpu, *c_gpu;

  a = (float *)malloc(sizeof(float) * SIZE * SIZE);
  b = (float *)malloc(sizeof(float) * SIZE * SIZE);
  c_cpu = (float *)malloc(sizeof(float) * SIZE * SIZE);
  c_gpu = (float *)malloc(sizeof(float) * SIZE * SIZE);

  fprintf(stdout, "<< Matrix Sum >>\n");

  init(a, b, c_cpu, c_gpu);

  t_start = rtclock();
  sum_GPU(a, b, c_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  sum_CPU(a, b, c_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(c_cpu, c_gpu);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);

  return 0;
}

