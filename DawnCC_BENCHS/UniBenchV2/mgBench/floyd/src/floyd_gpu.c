/*
    Analisar porque SIZE>=419 gera erro de falha de segmentação


    This program calculates the distance between the k neighbors in a Cartesian
   map.
    It generates a matrix with the distance between the neighbors.
    This program create a csv file with the time execution results for each
   function(CPU,GPU) in this format: size of matrix, cpu time, gpu time.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 1.0

    Run:
    folder_ipmacc/ipmacc folder_archive/k-nearest.c
    ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 400
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.05

/// initialize the cartesian map
void init(int *matrix, int *matrix_dist_cpu, int *matrix_dist_gpu) {
  int i, j, r, m;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (matrix + 0) > (void*) (matrix_dist_cpu + 160000))
  || ((void*) (matrix_dist_cpu + 0) > (void*) (matrix + 160000)));
  RST_AI1 |= !(((void*) (matrix + 0) > (void*) (matrix_dist_gpu + 160000))
  || ((void*) (matrix_dist_gpu + 0) > (void*) (matrix + 160000)));
  RST_AI1 |= !(((void*) (matrix_dist_cpu + 0) > (void*) (matrix_dist_gpu + 160000))
  || ((void*) (matrix_dist_gpu + 0) > (void*) (matrix_dist_cpu + 160000)));
  #pragma omp target data map(tofrom: matrix[0:160000],matrix_dist_cpu[0:160000],matrix_dist_gpu[0:160000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      matrix[i * SIZE + j] = 99999999;
      matrix_dist_cpu[i * SIZE + j] = 99999999;
      matrix_dist_gpu[i * SIZE + j] = 99999999;
    }
  }

  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (matrix + 0) > (void*) (matrix_dist_cpu + 160000))
  || ((void*) (matrix_dist_cpu + 0) > (void*) (matrix + 160000)));
  RST_AI2 |= !(((void*) (matrix + 0) > (void*) (matrix_dist_gpu + 160000))
  || ((void*) (matrix_dist_gpu + 0) > (void*) (matrix + 160000)));
  RST_AI2 |= !(((void*) (matrix_dist_cpu + 0) > (void*) (matrix_dist_gpu + 160000))
  || ((void*) (matrix_dist_gpu + 0) > (void*) (matrix_dist_cpu + 160000)));
  #pragma omp target data map(tofrom: matrix[0:160000],matrix_dist_cpu[0:160000],matrix_dist_gpu[0:160000]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < SIZE; i++) {
    r = (i * 97) % SIZE;
    for (j = 0; j < r; j++) {
      m = (((j * 1021) * 71 % (SIZE * SIZE)) + 1);
      matrix[i * SIZE + j] = m;
      if (i == j) {
        matrix[i * SIZE + j] = 0;
      }
    }
  }
}

/// Knearest algorithm GPU
/// s = size of cartesian map
void Knearest_GPU(int *matrix, int *matrix_dist) {
  int i, j, k;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (matrix + 0) > (void*) (matrix_dist + 160000))
  || ((void*) (matrix_dist + 0) > (void*) (matrix + 160000)));
  #pragma omp target data map(to: matrix[0:160000]) map(tofrom: matrix_dist[0:160000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (matrix[i * SIZE + j] != 99999999) {
        matrix_dist[i * SIZE + j] = matrix[i * SIZE + j];
      }
    }
    matrix_dist[i * SIZE + i] = 0;
  }

/// opportunity of parallelism here
#pragma omp target device(GPU_DEVICE)
#pragma omp target map(tofrom : matrix_dist[ : SIZE *SIZE])
  {
    char RST_AI2 = 0;
    RST_AI2 |= !(((void*) (matrix + 0) > (void*) (matrix_dist + 160000))
    || ((void*) (matrix_dist + 0) > (void*) (matrix + 160000)));
    #pragma omp target data map(tofrom: matrix[0:160000],matrix_dist[0:160000]) if(!RST_AI2)
    #pragma omp target if(!RST_AI2)
    for (i = 0; i < SIZE; i++) {
#pragma omp parallel for
      for (k = 0; k < SIZE; k++) {
        for (j = 0; j < SIZE; j++) {
          if (matrix_dist[k * SIZE + i] != 99999999 &&
              matrix_dist[i * SIZE + j] != 99999999 &&
              matrix_dist[k * SIZE + j] >
                  matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j]) {
            matrix_dist[k * SIZE + j] =
                matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j];
          }
        }
      }
    }
  }
}

void Knearest_CPU(int *matrix, int *matrix_dist) {
  int i, j, k;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (matrix + 0) > (void*) (matrix_dist + 160000))
  || ((void*) (matrix_dist + 0) > (void*) (matrix + 160000)));
  #pragma omp target data map(to: matrix[0:160000]) map(tofrom: matrix_dist[0:160000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (matrix[i * SIZE + j] != 99999999) {
        matrix_dist[i * SIZE + j] = matrix[i * SIZE + j];
      }
    }
    matrix_dist[i * SIZE + i] = 0;
  }

  /// opportunity of parallelism here
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (matrix + 0) > (void*) (matrix_dist + 160000))
  || ((void*) (matrix_dist + 0) > (void*) (matrix + 160000)));
  #pragma omp target data map(tofrom: matrix[0:160000],matrix_dist[0:160000]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < SIZE; i++) {
    for (k = 0; k < SIZE; k++) {
      for (j = 0; j < SIZE; j++) {
        if (matrix_dist[k * SIZE + i] != 99999999 &&
            matrix_dist[i * SIZE + j] != 99999999 &&
            matrix_dist[k * SIZE + j] >
                matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j]) {
          matrix_dist[k * SIZE + j] =
              matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j];
        }
      }
    }
  }
}

void compareResults(int *B, int *B_GPU) {
  int i, j, fail;
  fail = 0;

  // Compare B and B_GPU
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (B + 0) > (void*) (B_GPU + 160000))
  || ((void*) (B_GPU + 0) > (void*) (B + 160000)));
  #pragma omp target data map(to: B[0:160000],B_GPU[0:160000])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(B[i * SIZE + j], B_GPU[i * SIZE + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  int i;
  int points, var;
  double t_start, t_end;

  int *matrix;
  int *matrix_dist_cpu, *matrix_dist_gpu;

  fprintf(stdout, "<< K-nearest GPU >>\n");

  matrix = (int *)malloc(sizeof(int) * SIZE * SIZE);
  matrix_dist_cpu = (int *)malloc(sizeof(int) * SIZE * SIZE);
  matrix_dist_gpu = (int *)malloc(sizeof(int) * SIZE * SIZE);

  init(matrix, matrix_dist_cpu, matrix_dist_gpu);

  t_start = rtclock();
  Knearest_GPU(matrix, matrix_dist_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  Knearest_CPU(matrix, matrix_dist_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(matrix_dist_cpu, matrix_dist_gpu);

  free(matrix);
  free(matrix_dist_cpu);
  free(matrix_dist_gpu);

  return 0;
}

