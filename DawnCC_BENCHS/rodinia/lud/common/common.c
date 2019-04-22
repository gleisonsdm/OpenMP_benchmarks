#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "common.h"

void stopwatch_start(stopwatch *sw) {
  if (sw == NULL)
    return;

  bzero(&sw->begin, sizeof(struct timeval));
  bzero(&sw->end, sizeof(struct timeval));

  gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw) {
  if (sw == NULL)
    return;

  gettimeofday(&sw->end, NULL);
}

double get_interval_by_sec(stopwatch *sw) {
  if (sw == NULL)
    return 0;
  return ((double)(sw->end.tv_sec - sw->begin.tv_sec) +
          (double)(sw->end.tv_usec - sw->begin.tv_usec) / 1000000);
}

int get_interval_by_usec(stopwatch *sw) {
  if (sw == NULL)
    return 0;
  return ((sw->end.tv_sec - sw->begin.tv_sec) * 1000000 +
          (sw->end.tv_usec - sw->begin.tv_usec));
}

func_ret_t create_matrix_from_file(float **mp, const char *filename,
                                   int *size_p) {
  int i, j, size;
  float *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL) {
    return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (float *)malloc(sizeof(float) * size * size);
  if (m == NULL) {
    fclose(fp);
    return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      fscanf(fp, "%f ", m + i * size + j);
    }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}

func_ret_t create_matrix_from_random(float **mp, int size) {
  float *l, *u, *m;
  int i, j, k;

  srand(time(NULL));

  l = (float *)malloc(size * size * sizeof(float));
  if (l == NULL)
    return RET_FAILURE;

  u = (float *)malloc(size * size * sizeof(float));
  if (u == NULL) {
    free(l);
    return RET_FAILURE;
  }

  long long int AI1[9];
  AI1[0] = size + -1;
  AI1[1] = size * AI1[0];
  AI1[2] = AI1[1] + AI1[0];
  AI1[3] = 4 * AI1[2];
  AI1[4] = AI1[3] * 1;
  AI1[5] = AI1[4] + 1;
  AI1[6] = AI1[5] / 4;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  #pragma omp target data map(tofrom: l[0:AI1[8]])
  #pragma omp target
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (i > j) {
        l[i * size + j] = GET_RAND_FP;
      } else if (i == j) {
        l[i * size + j] = 1;
      } else {
        l[i * size + j] = 0;
      }
    }
  }

  long long int AI2[9];
  AI2[0] = size + -1;
  AI2[1] = size * AI2[0];
  AI2[2] = AI2[1] + AI2[0];
  AI2[3] = 4 * AI2[2];
  AI2[4] = AI2[3] * 1;
  AI2[5] = AI2[4] + 1;
  AI2[6] = AI2[5] / 4;
  AI2[7] = (AI2[6] > 0);
  AI2[8] = (AI2[7] ? AI2[6] : 0);
  #pragma omp target data map(tofrom: u[0:AI2[8]])
  #pragma omp target
  for (j = 0; j < size; j++) {
    for (i = 0; i < size; i++) {
      if (i > j) {
        u[j * size + i] = 0;
      } else {
        u[j * size + i] = GET_RAND_FP;
      }
    }
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k <= MIN(i, j); k++)
        m[i * size + j] = l[i * size + k] * u[j * size + k];
    }
  }

  free(l);
  free(u);

  *mp = m;

  return RET_SUCCESS;
}

void matrix_multiply(float *inputa, float *inputb, float *output, int size) {
  int i, j, k;

  long long int AI1[8];
  AI1[0] = size + -1;
  AI1[1] = size * AI1[0];
  AI1[2] = AI1[1] + AI1[0];
  AI1[3] = AI1[2] * 4;
  AI1[4] = AI1[3] + 4;
  AI1[5] = AI1[4] / 4;
  AI1[6] = (AI1[5] > 0);
  AI1[7] = (AI1[6] ? AI1[5] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (inputa + 0) > (void*) (inputb + AI1[7]))
  || ((void*) (inputb + 0) > (void*) (inputa + AI1[7])));
  RST_AI1 |= !(((void*) (inputa + 0) > (void*) (output + AI1[7]))
  || ((void*) (output + 0) > (void*) (inputa + AI1[7])));
  RST_AI1 |= !(((void*) (inputb + 0) > (void*) (output + AI1[7]))
  || ((void*) (output + 0) > (void*) (inputb + AI1[7])));
  #pragma omp target data map(to: inputa[0:AI1[7]],inputb[0:AI1[7]]) map(tofrom: output[0:AI1[7]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < size; i++)
    for (k = 0; k < size; k++)
      for (j = 0; j < size; j++)
        output[i * size + j] = inputa[i * size + k] * inputb[k * size + j];
}

func_ret_t lud_verify(float *m, float *lu, int matrix_dim) {
  int i, j, k;
  float *tmp = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

  for (i = 0; i < matrix_dim; i++)
    for (j = 0; j < matrix_dim; j++) {
      float sum = 0;
      float l, u;
      for (k = 0; k <= MIN(i, j); k++) {
        if (i == k)
          l = 1;
        else
          l = lu[i * matrix_dim + k];
        u = lu[k * matrix_dim + j];
        sum += l * u;
      }
      tmp[i * matrix_dim + j] = sum;
    }
  /* printf(">>>>>LU<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", lu[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>result<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", tmp[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>input<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", m[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  for (i = 0; i < matrix_dim; i++) {
    for (j = 0; j < matrix_dim; j++) {
      if (fabs(m[i * matrix_dim + j] - tmp[i * matrix_dim + j]) > 0.0001)
        printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j,
               m[i * matrix_dim + j], tmp[i * matrix_dim + j]);
    }
  }
  free(tmp);
}

void matrix_duplicate(float *src, float **dst, int matrix_dim) {
  int s = matrix_dim * matrix_dim * sizeof(float);
  float *p = (float *)malloc(s);
  memcpy(p, src, s);
  *dst = p;
}

void print_matrix(float *m, int matrix_dim) {
  int i, j;
  for (i = 0; i < matrix_dim; i++) {
    for (j = 0; j < matrix_dim; j++)
      printf("%f ", m[i * matrix_dim + j]);
    printf("\n");
  }
}

// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t create_matrix(float **mp, int size) {
  float *m;
  int i, j;
  float lamda = -0.001;
  float coe[2 * size - 1];
  float coe_i = 0.0;

  long long int AI1[2];
  AI1[0] = 2 * size;
  AI1[1] = AI1[0] - 1;
  #pragma omp target data map(tofrom: coe[0:AI1[1]])
  #pragma omp target
  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }

  m = (float *)malloc(sizeof(float) * size * size);
  if (m == NULL) {
    return RET_FAILURE;
  }

  long long int AI2[11];
  AI2[0] = 2 * size;
  AI2[1] = AI2[0] - 1;
  AI2[2] = size + -1;
  AI2[3] = size * AI2[2];
  AI2[4] = AI2[3] + AI2[2];
  AI2[5] = 4 * AI2[4];
  AI2[6] = AI2[5] * 1;
  AI2[7] = AI2[6] + 1;
  AI2[8] = AI2[7] / 4;
  AI2[9] = (AI2[8] > 0);
  AI2[10] = (AI2[9] ? AI2[8] : 0);
  char RST_AI2 = 0;
  #pragma omp target data map(to: coe[0:AI2[1]]) map(tofrom: m[0:AI2[10]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }

  *mp = m;

  return RET_SUCCESS;
}

