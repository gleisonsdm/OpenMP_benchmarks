//<libmptogpu> Error executing kernel. Global Work Size is NULL or exceeded
// valid range.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

typedef struct point {
  int x;
  int y;
} point;

typedef struct sel_points {
  int position;
  float value;
} sel_points;

#define SIZE 500
#define points 250
#define var SIZE / points
#define default_v 100000.00
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.01

void init(int s, point *vector, sel_points *selected) {
  int i, j;
  long long int AI1[9];
  AI1[0] = s + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = 4 + AI1[1];
  AI1[3] = AI1[2] > AI1[1];
  AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
  AI1[5] = AI1[4] + 8;
  AI1[6] = AI1[5] / 8;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  #pragma omp target data map(tofrom: vector[0:AI1[8]])
  #pragma omp target
  for (i = 0; i < s; i++) {
    vector[i].x = i;
    vector[i].y = i * 2;
  }
  long long int AI2[13];
  AI2[0] = s + -1;
  AI2[1] = s * AI2[0];
  AI2[2] = AI2[1] + AI2[0];
  AI2[3] = 2 * AI2[2];
  AI2[4] = AI2[3] * 4;
  AI2[5] = AI2[2] * 8;
  AI2[6] = AI2[4] > AI2[5];
  AI2[7] = (AI2[6] ? AI2[4] : AI2[5]);
  AI2[8] = (long long int) AI2[7];
  AI2[9] = AI2[8] + 8;
  AI2[10] = AI2[9] / 8;
  AI2[11] = (AI2[10] > 0);
  AI2[12] = (AI2[11] ? AI2[10] : 0);
  #pragma omp target data map(tofrom: selected[0:AI2[12]])
  #pragma omp target
  for (i = 0; i < s; i++) {
    for (j = 0; j < s; j++) {
      selected[i * s + j].position = 0;
      selected[i * s + j].value = default_v;
    }
  }
}

void k_nearest_gpu(int s, point *vector, sel_points *selected) {
  int i, j, m, q;
  q = s * s;

#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : vector[0 : s]) map(tofrom : selected[0 : q])
  {
#pragma omp parallel for collapse(2)
    long long int AI1[61];
    AI1[0] = s + -1;
    AI1[1] = 8 * AI1[0];
    AI1[2] = 12 + AI1[1];
    AI1[3] = s + -2;
    AI1[4] = -1 * AI1[0];
    AI1[5] = AI1[3] + AI1[4];
    AI1[6] = 8 * AI1[5];
    AI1[7] = AI1[2] + AI1[6];
    AI1[8] = 4 + AI1[1];
    AI1[9] = 8 + AI1[1];
    AI1[10] = AI1[9] + AI1[6];
    AI1[11] = AI1[10] > AI1[1];
    AI1[12] = (AI1[11] ? AI1[10] : AI1[1]);
    AI1[13] = AI1[8] > AI1[12];
    AI1[14] = (AI1[13] ? AI1[8] : AI1[12]);
    AI1[15] = AI1[7] > AI1[14];
    AI1[16] = (AI1[15] ? AI1[7] : AI1[14]);
    AI1[17] = AI1[16] + 8;
    AI1[18] = AI1[17] / 8;
    AI1[19] = (AI1[18] > 0);
    AI1[20] = (AI1[19] ? AI1[18] : 0);
    AI1[21] = s * 8;
    AI1[22] = 2 * s;
    AI1[23] = AI1[22] * 4;
    AI1[24] = AI1[23] < 8;
    AI1[25] = (AI1[24] ? AI1[23] : 8);
    AI1[26] = AI1[21] < AI1[25];
    AI1[27] = (AI1[26] ? AI1[21] : AI1[25]);
    AI1[28] = AI1[27] / 8;
    AI1[29] = (AI1[28] > 0);
    AI1[30] = (AI1[29] ? AI1[28] : 0);
    AI1[31] = s + 1;
    AI1[32] = AI1[31] * AI1[0];
    AI1[33] = s + AI1[32];
    AI1[34] = s * AI1[5];
    AI1[35] = AI1[33] + AI1[34];
    AI1[36] = AI1[35] * 8;
    AI1[37] = 2 * AI1[35];
    AI1[38] = AI1[37] * 4;
    AI1[39] = 1 + AI1[32];
    AI1[40] = AI1[39] + AI1[5];
    AI1[41] = AI1[40] * 8;
    AI1[42] = 2 * AI1[40];
    AI1[43] = AI1[42] * 4;
    AI1[44] = AI1[41] > AI1[43];
    AI1[45] = (AI1[44] ? AI1[41] : AI1[43]);
    AI1[46] = AI1[38] > AI1[45];
    AI1[47] = (AI1[46] ? AI1[38] : AI1[45]);
    AI1[48] = AI1[36] > AI1[47];
    AI1[49] = (AI1[48] ? AI1[36] : AI1[47]);
    AI1[50] = (long long int) AI1[49];
    AI1[51] = AI1[50] + 8;
    AI1[52] = AI1[51] / 8;
    AI1[53] = (AI1[52] > 0);
    AI1[54] = (AI1[53] ? AI1[52] : 0);
    AI1[55] = AI1[54] - AI1[30];
    AI1[56] = (AI1[55] > 0);
    AI1[57] = AI1[30] + AI1[55];
    AI1[58] = -1 * AI1[55];
    AI1[59] = AI1[56] ? AI1[30] : AI1[57];
    AI1[60] = AI1[56] ? AI1[55] : AI1[58];
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (selected + AI1[59]) > (void*) (vector + AI1[20]))
    || ((void*) (vector + 0) > (void*) (selected + AI1[60])));
    #pragma omp target data map(to: vector[0:AI1[20]]) map(tofrom: selected[AI1[59]:AI1[60]]) if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < s; i++) {
      for (j = i + 1; j < s; j++) {
        float distance, x, y;
        x = vector[i].x - vector[j].x;
        y = vector[i].y - vector[j].y;
        x = x * x;
        y = y * y;

        distance = x + y;
        distance = sqrt(distance);

        selected[i * s + j].value = distance;
        selected[i * s + j].position = j;

        selected[j * s + i].value = distance;
        selected[j * s + i].position = i;
      }
    }

/// for each line in matrix
/// order values
#pragma omp parallel for collapse(1)
    long long int AI2[20];
    AI2[0] = s + -1;
    AI2[1] = s * AI2[0];
    AI2[2] = 1 + AI2[1];
    AI2[3] = AI2[2] + AI2[0];
    AI2[4] = s + -2;
    AI2[5] = -1 * AI2[0];
    AI2[6] = AI2[4] + AI2[5];
    AI2[7] = AI2[3] + AI2[6];
    AI2[8] = 2 * AI2[7];
    AI2[9] = AI2[8] * 4;
    AI2[10] = AI2[1] + AI2[0];
    AI2[11] = 2 * AI2[10];
    AI2[12] = AI2[11] * 4;
    AI2[13] = AI2[9] > AI2[12];
    AI2[14] = (AI2[13] ? AI2[9] : AI2[12]);
    AI2[15] = (long long int) AI2[14];
    AI2[16] = AI2[15] + 8;
    AI2[17] = AI2[16] / 8;
    AI2[18] = (AI2[17] > 0);
    AI2[19] = (AI2[18] ? AI2[17] : 0);
    #pragma omp target data map(to: selected[0:AI2[19]]) 
    #pragma omp target
    for (i = 0; i < s; i++) {
      for (j = 0; j < s; j++) {
        for (m = j + 1; m < s; m++) {
          if (selected[i * s + j].value > selected[i * s + m].value) {
            sel_points aux;
            aux = selected[i * s + j];
            selected[i * s + j] = selected[i * s + m];
            selected[i * s + m] = aux;
          }
        }
      }
    }
  }
}

void k_nearest_cpu(int s, point *vector, sel_points *selected) {
  int i, j;
  long long int AI1[61];
  AI1[0] = s + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = 12 + AI1[1];
  AI1[3] = s + -2;
  AI1[4] = -1 * AI1[0];
  AI1[5] = AI1[3] + AI1[4];
  AI1[6] = 8 * AI1[5];
  AI1[7] = AI1[2] + AI1[6];
  AI1[8] = 4 + AI1[1];
  AI1[9] = 8 + AI1[1];
  AI1[10] = AI1[9] + AI1[6];
  AI1[11] = AI1[10] > AI1[1];
  AI1[12] = (AI1[11] ? AI1[10] : AI1[1]);
  AI1[13] = AI1[8] > AI1[12];
  AI1[14] = (AI1[13] ? AI1[8] : AI1[12]);
  AI1[15] = AI1[7] > AI1[14];
  AI1[16] = (AI1[15] ? AI1[7] : AI1[14]);
  AI1[17] = AI1[16] + 8;
  AI1[18] = AI1[17] / 8;
  AI1[19] = (AI1[18] > 0);
  AI1[20] = (AI1[19] ? AI1[18] : 0);
  AI1[21] = s * 8;
  AI1[22] = 2 * s;
  AI1[23] = AI1[22] * 4;
  AI1[24] = AI1[23] < 8;
  AI1[25] = (AI1[24] ? AI1[23] : 8);
  AI1[26] = AI1[21] < AI1[25];
  AI1[27] = (AI1[26] ? AI1[21] : AI1[25]);
  AI1[28] = AI1[27] / 8;
  AI1[29] = (AI1[28] > 0);
  AI1[30] = (AI1[29] ? AI1[28] : 0);
  AI1[31] = s + 1;
  AI1[32] = AI1[31] * AI1[0];
  AI1[33] = s + AI1[32];
  AI1[34] = s * AI1[5];
  AI1[35] = AI1[33] + AI1[34];
  AI1[36] = AI1[35] * 8;
  AI1[37] = 2 * AI1[35];
  AI1[38] = AI1[37] * 4;
  AI1[39] = 1 + AI1[32];
  AI1[40] = AI1[39] + AI1[5];
  AI1[41] = AI1[40] * 8;
  AI1[42] = 2 * AI1[40];
  AI1[43] = AI1[42] * 4;
  AI1[44] = AI1[41] > AI1[43];
  AI1[45] = (AI1[44] ? AI1[41] : AI1[43]);
  AI1[46] = AI1[38] > AI1[45];
  AI1[47] = (AI1[46] ? AI1[38] : AI1[45]);
  AI1[48] = AI1[36] > AI1[47];
  AI1[49] = (AI1[48] ? AI1[36] : AI1[47]);
  AI1[50] = (long long int) AI1[49];
  AI1[51] = AI1[50] + 8;
  AI1[52] = AI1[51] / 8;
  AI1[53] = (AI1[52] > 0);
  AI1[54] = (AI1[53] ? AI1[52] : 0);
  AI1[55] = AI1[54] - AI1[30];
  AI1[56] = (AI1[55] > 0);
  AI1[57] = AI1[30] + AI1[55];
  AI1[58] = -1 * AI1[55];
  AI1[59] = AI1[56] ? AI1[30] : AI1[57];
  AI1[60] = AI1[56] ? AI1[55] : AI1[58];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (selected + AI1[59]) > (void*) (vector + AI1[20]))
  || ((void*) (vector + 0) > (void*) (selected + AI1[60])));
  #pragma omp target data map(to: vector[0:AI1[20]]) map(tofrom: selected[AI1[59]:AI1[60]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < s; i++) {
    for (j = i + 1; j < s; j++) {
      float distance, x, y;
      x = vector[i].x - vector[j].x;
      y = vector[i].y - vector[j].y;
      x = x * x;
      y = y * y;

      distance = x + y;
      distance = sqrt(distance);

      selected[i * s + j].value = distance;
      selected[i * s + j].position = j;

      selected[j * s + i].value = distance;
      selected[j * s + i].position = i;
    }
  }
}

void order_points(int s, point *vector, sel_points *selected) {
  int i;
  long long int AI1[20];
  AI1[0] = s + -1;
  AI1[1] = s * AI1[0];
  AI1[2] = 1 + AI1[1];
  AI1[3] = AI1[2] + AI1[0];
  AI1[4] = s + -2;
  AI1[5] = -1 * AI1[0];
  AI1[6] = AI1[4] + AI1[5];
  AI1[7] = AI1[3] + AI1[6];
  AI1[8] = 2 * AI1[7];
  AI1[9] = AI1[8] * 4;
  AI1[10] = AI1[1] + AI1[0];
  AI1[11] = 2 * AI1[10];
  AI1[12] = AI1[11] * 4;
  AI1[13] = AI1[9] > AI1[12];
  AI1[14] = (AI1[13] ? AI1[9] : AI1[12]);
  AI1[15] = (long long int) AI1[14];
  AI1[16] = AI1[15] + 8;
  AI1[17] = AI1[16] / 8;
  AI1[18] = (AI1[17] > 0);
  AI1[19] = (AI1[18] ? AI1[17] : 0);
  #pragma omp target data map(to: selected[0:AI1[19]]) 
  #pragma omp target
  for (i = 0; i < s; i++) {
    /// for each line in matrix
    /// order values
    int j;
    for (j = 0; j < s; j++) {
      int m;
      for (m = j + 1; m < s; m++) {
        if (selected[i * s + j].value > selected[i * s + m].value) {
          sel_points aux;
          aux = selected[i * s + j];
          selected[i * s + j] = selected[i * s + m];
          selected[i * s + m] = aux;
        }
      }
    }
  }
}

void compareResults(sel_points *B, sel_points *B_GPU) {
  int i, j, fail;
  fail = 0;

  // Compare B and B_GPU
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (B + 0) > (void*) (B_GPU + 250000))
  || ((void*) (B_GPU + 0) > (void*) (B + 250000)));
  #pragma omp target data map(to: B[0:250000],B_GPU[0:250000])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      // Value
      if (percentDiff(B[i * SIZE + j].value, B_GPU[i * SIZE + j].value) >
          ERROR_THRESHOLD) {
        fail++;
      }
      // Position
      if (percentDiff(B[i * SIZE + j].position, B_GPU[i * SIZE + j].position) >
          ERROR_THRESHOLD) {
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
  double t_start, t_end;
  point *vector;
  sel_points *selected_cpu, *selected_gpu;

  vector = (point *)malloc(sizeof(point) * SIZE);
  selected_cpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);
  selected_gpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);

  int i;

  fprintf(stdout, "<< Nearest >>\n");

  t_start = rtclock();
  for (i = (var - 1); i < SIZE; i += var) {
    init(i, vector, selected_cpu);
    k_nearest_cpu(i, vector, selected_cpu);
    order_points(i, vector, selected_cpu);
  }
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  for (i = (var - 1); i < SIZE; i += var) {
    init(i, vector, selected_gpu);
    k_nearest_gpu(i, vector, selected_gpu);
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(selected_cpu, selected_gpu);

  free(selected_cpu);
  free(selected_gpu);
  free(vector);
  return 0;
}

