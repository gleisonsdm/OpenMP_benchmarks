#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define OPENMP
#include "../../common/rodiniaUtilFunctions.h"

#define GPU_DEVICE 1
#define ERROR_THRESHOLD 0.05

//#define NUM_THREAD 4

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int *input_itemsets, int *referrence, int max_rows, int max_cols,
             int penalty, int dev);

int maximum(int a, int b, int c) {

  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}

int blosum62[24][24] = {{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2,
                         -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
                        {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2,
                         -1, -1, -3, -2, -3, -1, 0, -1, -4},
                        {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1,
                         0, -4, -2, -3, 3, 0, -1, -4},
                        {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1,
                         0, -1, -4, -3, -3, 4, 1, -1, -4},
                        {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2,
                         -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0,
                         -1, -2, -1, -2, 0, 3, -1, -4},
                        {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0,
                         -1, -3, -2, -2, 1, 4, -1, -4},
                        {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3,
                         -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2,
                         -1, -2, -2, 2, -3, 0, 0, -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3,
                         -2, -1, -3, -1, 3, -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3,
                         -2, -1, -2, -1, 1, -4, -3, -1, -4},
                        {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1,
                         0, -1, -3, -2, -2, 0, 1, -1, -4},
                        {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2,
                         -1, -1, -1, -1, 1, -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4,
                         -2, -2, 1, 3, -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,
                         7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4,
                         1, -3, -2, -2, 0, 0, 0, -4},
                        {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2,
                         -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,
                         -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3,
                         -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2,
                         -2, 0, -3, -1, 4, -3, -2, -1, -4},
                        {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2,
                         0, -1, -4, -3, -3, 4, 1, -1, -4},
                        {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0,
                         -1, -3, -2, -2, 1, 4, -1, -4},
                        {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n",
          argv[0]);
  fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
  fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
  fprintf(stderr, "\t<num_threads>    - no. of threads\n");
  exit(1);
}

void compareResults(int *cpu, int *gpu, int max_rows, int max_cols) {
  int i, fail;
  fail = 0;

  // Compare B and B_GPU
  long long int AI1[7];
  AI1[0] = max_rows * max_cols;
  AI1[1] = AI1[0] + -1;
  AI1[2] = 4 * AI1[1];
  AI1[3] = AI1[2] + 4;
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (cpu + 0) > (void*) (gpu + AI1[6]))
  || ((void*) (gpu + 0) > (void*) (cpu + AI1[6])));
  #pragma omp target data map(to: cpu[0:AI1[6]],gpu[0:AI1[6]])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < max_rows * max_cols; i++) {
    if (percentDiff(gpu[i], cpu[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

void init(int *input_itemsets_cpu, int *input_itemsets_gpu, int *referrence_cpu,
          int *referrence_gpu, int max_rows, int max_cols, int penalty) {
  srand(7);

  long long int AI1[9];
  AI1[0] = max_cols + -1;
  AI1[1] = max_cols * AI1[0];
  AI1[2] = max_rows + -1;
  AI1[3] = AI1[1] + AI1[2];
  AI1[4] = AI1[3] * 4;
  AI1[5] = AI1[4] + 4;
  AI1[6] = AI1[5] / 4;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (input_itemsets_cpu + 0) > (void*) (input_itemsets_gpu + AI1[8]))
  || ((void*) (input_itemsets_gpu + 0) > (void*) (input_itemsets_cpu + AI1[8])));
  #pragma omp target data map(tofrom: input_itemsets_cpu[0:AI1[8]],input_itemsets_gpu[0:AI1[8]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets_cpu[i * max_cols + j] = 0;
      input_itemsets_gpu[i * max_cols + j] = 0;
    }
  }

  long long int AI2[18];
  AI2[0] = max_cols * 4;
  AI2[1] = AI2[0] / 4;
  AI2[2] = (AI2[1] > 0);
  AI2[3] = (AI2[2] ? AI2[1] : 0);
  AI2[4] = max_rows + -2;
  AI2[5] = max_cols * AI2[4];
  AI2[6] = max_cols + AI2[5];
  AI2[7] = AI2[6] * 4;
  AI2[8] = AI2[7] + 4;
  AI2[9] = AI2[8] / 4;
  AI2[10] = (AI2[9] > 0);
  AI2[11] = (AI2[10] ? AI2[9] : 0);
  AI2[12] = AI2[11] - AI2[3];
  AI2[13] = (AI2[12] > 0);
  AI2[14] = AI2[3] + AI2[12];
  AI2[15] = -1 * AI2[12];
  AI2[16] = AI2[13] ? AI2[3] : AI2[14];
  AI2[17] = AI2[13] ? AI2[12] : AI2[15];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (input_itemsets_cpu + AI2[16]) > (void*) (input_itemsets_gpu + AI2[17]))
  || ((void*) (input_itemsets_gpu + AI2[16]) > (void*) (input_itemsets_cpu + AI2[17])));
  #pragma omp target data map(tofrom: input_itemsets_cpu[AI2[16]:AI2[17]],input_itemsets_gpu[AI2[16]:AI2[17]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (int i = 1; i < max_rows; i++) { // please define your own sequence.
    int al = rand() % 10 + 1;
    input_itemsets_cpu[i * max_cols] = al;
    input_itemsets_gpu[i * max_cols] = al;
  }

  long long int AI3[13];
  AI3[0] = max_cols + -2;
  AI3[1] = 4 * AI3[0];
  AI3[2] = 4 + AI3[1];
  AI3[3] = AI3[2] + 4;
  AI3[4] = AI3[3] / 4;
  AI3[5] = (AI3[4] > 0);
  AI3[6] = (AI3[5] ? AI3[4] : 0);
  AI3[7] = AI3[6] - 1;
  AI3[8] = (AI3[7] > 0);
  AI3[9] = 1 + AI3[7];
  AI3[10] = -1 * AI3[7];
  AI3[11] = AI3[8] ? 1 : AI3[9];
  AI3[12] = AI3[8] ? AI3[7] : AI3[10];
  char RST_AI3 = 0;
  RST_AI3 |= !(((void*) (input_itemsets_cpu + AI3[11]) > (void*) (input_itemsets_gpu + AI3[12]))
  || ((void*) (input_itemsets_gpu + AI3[11]) > (void*) (input_itemsets_cpu + AI3[12])));
  #pragma omp target data map(tofrom: input_itemsets_cpu[AI3[11]:AI3[12]],input_itemsets_gpu[AI3[11]:AI3[12]]) if(!RST_AI3)
  #pragma omp target if(!RST_AI3)
  for (int j = 1; j < max_cols; j++) { // please define your own sequence.
    int al = rand() % 10 + 1;
    input_itemsets_cpu[j] = al;
    input_itemsets_gpu[j] = al;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      referrence_cpu[i * max_cols + j] =
          blosum62[input_itemsets_cpu[i * max_cols]][input_itemsets_cpu[j]];
      referrence_gpu[i * max_cols + j] =
          blosum62[input_itemsets_gpu[i * max_cols]][input_itemsets_gpu[j]];
    }
  }

  long long int AI4[25];
  AI4[0] = max_cols * 4;
  AI4[1] = 4 < AI4[0];
  AI4[2] = (AI4[1] ? 4 : AI4[0]);
  AI4[3] = AI4[2] / 4;
  AI4[4] = (AI4[3] > 0);
  AI4[5] = (AI4[4] ? AI4[3] : 0);
  AI4[6] = max_cols + -2;
  AI4[7] = 4 * AI4[6];
  AI4[8] = 4 + AI4[7];
  AI4[9] = max_rows + -2;
  AI4[10] = max_cols * AI4[9];
  AI4[11] = max_cols + AI4[10];
  AI4[12] = AI4[11] * 4;
  AI4[13] = AI4[8] > AI4[12];
  AI4[14] = (AI4[13] ? AI4[8] : AI4[12]);
  AI4[15] = AI4[14] + 4;
  AI4[16] = AI4[15] / 4;
  AI4[17] = (AI4[16] > 0);
  AI4[18] = (AI4[17] ? AI4[16] : 0);
  AI4[19] = AI4[18] - AI4[5];
  AI4[20] = (AI4[19] > 0);
  AI4[21] = AI4[5] + AI4[19];
  AI4[22] = -1 * AI4[19];
  AI4[23] = AI4[20] ? AI4[5] : AI4[21];
  AI4[24] = AI4[20] ? AI4[19] : AI4[22];
  char RST_AI4 = 0;
  RST_AI4 |= !(((void*) (input_itemsets_cpu + AI4[23]) > (void*) (input_itemsets_gpu + AI4[24]))
  || ((void*) (input_itemsets_gpu + AI4[23]) > (void*) (input_itemsets_cpu + AI4[24])));
  #pragma omp target data map(tofrom: input_itemsets_cpu[AI4[23]:AI4[24]],input_itemsets_gpu[AI4[23]:AI4[24]]) if(!RST_AI4)
  #pragma omp target if(!RST_AI4)
  for (int i = 1; i < max_rows; i++) {
    input_itemsets_cpu[i * max_cols] = -i * penalty;
    input_itemsets_gpu[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++) {
      input_itemsets_cpu[j] = -j * penalty;
      input_itemsets_gpu[j] = -j * penalty;
    }
  }
}

void runTest_GPU(int max_cols, int max_rows, int *input_itemsets,
                 int *referrence, int penalty) {
  int index, i, idx;
#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : referrence[0 : max_rows *max_cols])                \
    map(tofrom : input_itemsets[0 : max_rows *max_cols])
  {
    long long int AI1[72];
    AI1[0] = max_cols + 1;
    AI1[1] = AI1[0] * 4;
    AI1[2] = max_cols * 4;
    AI1[3] = AI1[2] < 0;
    AI1[4] = (AI1[3] ? AI1[2] : 0);
    AI1[5] = AI1[2] < AI1[4];
    AI1[6] = (AI1[5] ? AI1[2] : AI1[4]);
    AI1[7] = 4 < AI1[6];
    AI1[8] = (AI1[7] ? 4 : AI1[6]);
    AI1[9] = 4 < AI1[8];
    AI1[10] = (AI1[9] ? 4 : AI1[8]);
    AI1[11] = AI1[1] < AI1[10];
    AI1[12] = (AI1[11] ? AI1[1] : AI1[10]);
    AI1[13] = AI1[1] < AI1[12];
    AI1[14] = (AI1[13] ? AI1[1] : AI1[12]);
    AI1[15] = 0 < AI1[14];
    AI1[16] = (AI1[15] ? 0 : AI1[14]);
    AI1[17] = AI1[16] / 4;
    AI1[18] = (AI1[17] > 0);
    AI1[19] = (AI1[18] ? AI1[17] : 0);
    AI1[20] = max_cols + -3;
    AI1[21] = max_cols + -1;
    AI1[22] = AI1[21] * AI1[20];
    AI1[23] = AI1[20] + AI1[22];
    AI1[24] = AI1[23] * 4;
    AI1[25] = AI1[0] + AI1[20];
    AI1[26] = AI1[25] + AI1[22];
    AI1[27] = AI1[26] * 4;
    AI1[28] = 1 + AI1[20];
    AI1[29] = AI1[28] + AI1[22];
    AI1[30] = AI1[29] * 4;
    AI1[31] = max_cols + AI1[20];
    AI1[32] = AI1[31] + AI1[22];
    AI1[33] = AI1[32] * 4;
    AI1[34] = AI1[33] > AI1[24];
    AI1[35] = (AI1[34] ? AI1[33] : AI1[24]);
    AI1[36] = AI1[33] > AI1[35];
    AI1[37] = (AI1[36] ? AI1[33] : AI1[35]);
    AI1[38] = AI1[30] > AI1[37];
    AI1[39] = (AI1[38] ? AI1[30] : AI1[37]);
    AI1[40] = AI1[30] > AI1[39];
    AI1[41] = (AI1[40] ? AI1[30] : AI1[39]);
    AI1[42] = AI1[27] > AI1[41];
    AI1[43] = (AI1[42] ? AI1[27] : AI1[41]);
    AI1[44] = AI1[27] > AI1[43];
    AI1[45] = (AI1[44] ? AI1[27] : AI1[43]);
    AI1[46] = AI1[24] > AI1[45];
    AI1[47] = (AI1[46] ? AI1[24] : AI1[45]);
    AI1[48] = (long long int) AI1[47];
    AI1[49] = AI1[48] + 4;
    AI1[50] = AI1[49] / 4;
    AI1[51] = (AI1[50] > 0);
    AI1[52] = (AI1[51] ? AI1[50] : 0);
    AI1[53] = AI1[52] - AI1[19];
    AI1[54] = (AI1[53] > 0);
    AI1[55] = AI1[19] + AI1[53];
    AI1[56] = -1 * AI1[53];
    AI1[57] = AI1[54] ? AI1[19] : AI1[55];
    AI1[58] = AI1[54] ? AI1[53] : AI1[56];
    AI1[59] = AI1[1] / 4;
    AI1[60] = (AI1[59] > 0);
    AI1[61] = (AI1[60] ? AI1[59] : 0);
    AI1[62] = AI1[27] + 4;
    AI1[63] = AI1[62] / 4;
    AI1[64] = (AI1[63] > 0);
    AI1[65] = (AI1[64] ? AI1[63] : 0);
    AI1[66] = AI1[65] - AI1[61];
    AI1[67] = (AI1[66] > 0);
    AI1[68] = AI1[61] + AI1[66];
    AI1[69] = -1 * AI1[66];
    AI1[70] = AI1[67] ? AI1[61] : AI1[68];
    AI1[71] = AI1[67] ? AI1[66] : AI1[69];
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (input_itemsets + AI1[57]) > (void*) (referrence + AI1[71]))
    || ((void*) (referrence + AI1[70]) > (void*) (input_itemsets + AI1[58])));
    #pragma omp target data map(to: referrence[AI1[70]:AI1[71]]) map(tofrom: input_itemsets[AI1[57]:AI1[58]]) if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < max_cols - 2; i++) {

#pragma omp parallel for
      for (idx = 0; idx <= i; idx++) {
        index = (idx + 1) * max_cols + (i + 1 - idx);

        int k;
        if ((input_itemsets[index - 1 - max_cols] + referrence[index]) <=
            (input_itemsets[index - 1] - penalty))
          k = (input_itemsets[index - 1] - penalty);
        else
          k = (input_itemsets[index - 1 - max_cols] + referrence[index]);

        if (k <= (input_itemsets[index - max_cols] - penalty))
          input_itemsets[index] = (input_itemsets[index - max_cols] - penalty);
        else
          input_itemsets[index] = k;
      }
    }
  }

// Compute bottom-right matrix
#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : referrence[0 : max_rows *max_cols])                \
    map(tofrom : input_itemsets[0 : max_rows *max_cols])
  {
    long long int AI2[84];
    AI2[0] = max_cols + -3;
    AI2[1] = AI2[0] * max_cols;
    AI2[2] = AI2[1] + 1;
    AI2[3] = AI2[2] * 4;
    AI2[4] = max_cols + -2;
    AI2[5] = AI2[4] * max_cols;
    AI2[6] = AI2[5] + 2;
    AI2[7] = AI2[6] * 4;
    AI2[8] = AI2[1] + 2;
    AI2[9] = AI2[8] * 4;
    AI2[10] = AI2[5] + 1;
    AI2[11] = AI2[10] * 4;
    AI2[12] = AI2[11] < AI2[3];
    AI2[13] = (AI2[12] ? AI2[11] : AI2[3]);
    AI2[14] = AI2[11] < AI2[13];
    AI2[15] = (AI2[14] ? AI2[11] : AI2[13]);
    AI2[16] = AI2[9] < AI2[15];
    AI2[17] = (AI2[16] ? AI2[9] : AI2[15]);
    AI2[18] = AI2[9] < AI2[17];
    AI2[19] = (AI2[18] ? AI2[9] : AI2[17]);
    AI2[20] = AI2[7] < AI2[19];
    AI2[21] = (AI2[20] ? AI2[7] : AI2[19]);
    AI2[22] = AI2[7] < AI2[21];
    AI2[23] = (AI2[22] ? AI2[7] : AI2[21]);
    AI2[24] = AI2[3] < AI2[23];
    AI2[25] = (AI2[24] ? AI2[3] : AI2[23]);
    AI2[26] = AI2[25] / 4;
    AI2[27] = (AI2[26] > 0);
    AI2[28] = (AI2[27] ? AI2[26] : 0);
    AI2[29] = max_cols + -4;
    AI2[30] = AI2[2] + AI2[29];
    AI2[31] = 1 - max_cols;
    AI2[32] = -1 * AI2[29];
    AI2[33] = AI2[29] + AI2[32];
    AI2[34] = AI2[31] * AI2[33];
    AI2[35] = AI2[30] + AI2[34];
    AI2[36] = AI2[35] * 4;
    AI2[37] = AI2[6] + AI2[29];
    AI2[38] = AI2[37] + AI2[34];
    AI2[39] = AI2[38] * 4;
    AI2[40] = AI2[8] + AI2[29];
    AI2[41] = AI2[40] + AI2[34];
    AI2[42] = AI2[41] * 4;
    AI2[43] = AI2[10] + AI2[29];
    AI2[44] = AI2[43] + AI2[34];
    AI2[45] = AI2[44] * 4;
    AI2[46] = AI2[45] > AI2[36];
    AI2[47] = (AI2[46] ? AI2[45] : AI2[36]);
    AI2[48] = AI2[45] > AI2[47];
    AI2[49] = (AI2[48] ? AI2[45] : AI2[47]);
    AI2[50] = AI2[42] > AI2[49];
    AI2[51] = (AI2[50] ? AI2[42] : AI2[49]);
    AI2[52] = AI2[42] > AI2[51];
    AI2[53] = (AI2[52] ? AI2[42] : AI2[51]);
    AI2[54] = AI2[39] > AI2[53];
    AI2[55] = (AI2[54] ? AI2[39] : AI2[53]);
    AI2[56] = AI2[39] > AI2[55];
    AI2[57] = (AI2[56] ? AI2[39] : AI2[55]);
    AI2[58] = AI2[36] > AI2[57];
    AI2[59] = (AI2[58] ? AI2[36] : AI2[57]);
    AI2[60] = (long long int) AI2[59];
    AI2[61] = AI2[60] + 4;
    AI2[62] = AI2[61] / 4;
    AI2[63] = (AI2[62] > 0);
    AI2[64] = (AI2[63] ? AI2[62] : 0);
    AI2[65] = AI2[64] - AI2[28];
    AI2[66] = (AI2[65] > 0);
    AI2[67] = AI2[28] + AI2[65];
    AI2[68] = -1 * AI2[65];
    AI2[69] = AI2[66] ? AI2[28] : AI2[67];
    AI2[70] = AI2[66] ? AI2[65] : AI2[68];
    AI2[71] = AI2[7] / 4;
    AI2[72] = (AI2[71] > 0);
    AI2[73] = (AI2[72] ? AI2[71] : 0);
    AI2[74] = AI2[39] + 4;
    AI2[75] = AI2[74] / 4;
    AI2[76] = (AI2[75] > 0);
    AI2[77] = (AI2[76] ? AI2[75] : 0);
    AI2[78] = AI2[77] - AI2[73];
    AI2[79] = (AI2[78] > 0);
    AI2[80] = AI2[73] + AI2[78];
    AI2[81] = -1 * AI2[78];
    AI2[82] = AI2[79] ? AI2[73] : AI2[80];
    AI2[83] = AI2[79] ? AI2[78] : AI2[81];
    char RST_AI2 = 0;
    RST_AI2 |= !(((void*) (input_itemsets + AI2[69]) > (void*) (referrence + AI2[83]))
    || ((void*) (referrence + AI2[82]) > (void*) (input_itemsets + AI2[70])));
    #pragma omp target data map(to: referrence[AI2[82]:AI2[83]]) map(tofrom: input_itemsets[AI2[69]:AI2[70]]) if(!RST_AI2)
    #pragma omp target if(!RST_AI2)
    for (i = max_cols - 4; i >= 0; i--) {
#pragma omp parallel for
      for (idx = 0; idx <= i; idx++) {
        index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;

        int k;
        if ((input_itemsets[index - 1 - max_cols] + referrence[index]) <=
            (input_itemsets[index - 1] - penalty))
          k = (input_itemsets[index - 1] - penalty);
        else
          k = (input_itemsets[index - 1 - max_cols] + referrence[index]);

        if (k <= (input_itemsets[index - max_cols] - penalty))
          input_itemsets[index] = (input_itemsets[index - max_cols] - penalty);
        else
          input_itemsets[index] = k;
      }
    }
  }
}

void runTest_CPU(int max_cols, int max_rows, int *input_itemsets,
                 int *referrence, int penalty) {
  int index, i, idx;
  // printf("Processing top-left matrix\n");
  long long int AI1[72];
  AI1[0] = max_cols + 1;
  AI1[1] = AI1[0] * 4;
  AI1[2] = max_cols * 4;
  AI1[3] = AI1[2] < 0;
  AI1[4] = (AI1[3] ? AI1[2] : 0);
  AI1[5] = AI1[2] < AI1[4];
  AI1[6] = (AI1[5] ? AI1[2] : AI1[4]);
  AI1[7] = 4 < AI1[6];
  AI1[8] = (AI1[7] ? 4 : AI1[6]);
  AI1[9] = 4 < AI1[8];
  AI1[10] = (AI1[9] ? 4 : AI1[8]);
  AI1[11] = AI1[1] < AI1[10];
  AI1[12] = (AI1[11] ? AI1[1] : AI1[10]);
  AI1[13] = AI1[1] < AI1[12];
  AI1[14] = (AI1[13] ? AI1[1] : AI1[12]);
  AI1[15] = 0 < AI1[14];
  AI1[16] = (AI1[15] ? 0 : AI1[14]);
  AI1[17] = AI1[16] / 4;
  AI1[18] = (AI1[17] > 0);
  AI1[19] = (AI1[18] ? AI1[17] : 0);
  AI1[20] = max_cols + -3;
  AI1[21] = max_cols + -1;
  AI1[22] = AI1[21] * AI1[20];
  AI1[23] = AI1[20] + AI1[22];
  AI1[24] = AI1[23] * 4;
  AI1[25] = AI1[0] + AI1[20];
  AI1[26] = AI1[25] + AI1[22];
  AI1[27] = AI1[26] * 4;
  AI1[28] = 1 + AI1[20];
  AI1[29] = AI1[28] + AI1[22];
  AI1[30] = AI1[29] * 4;
  AI1[31] = max_cols + AI1[20];
  AI1[32] = AI1[31] + AI1[22];
  AI1[33] = AI1[32] * 4;
  AI1[34] = AI1[33] > AI1[24];
  AI1[35] = (AI1[34] ? AI1[33] : AI1[24]);
  AI1[36] = AI1[33] > AI1[35];
  AI1[37] = (AI1[36] ? AI1[33] : AI1[35]);
  AI1[38] = AI1[30] > AI1[37];
  AI1[39] = (AI1[38] ? AI1[30] : AI1[37]);
  AI1[40] = AI1[30] > AI1[39];
  AI1[41] = (AI1[40] ? AI1[30] : AI1[39]);
  AI1[42] = AI1[27] > AI1[41];
  AI1[43] = (AI1[42] ? AI1[27] : AI1[41]);
  AI1[44] = AI1[27] > AI1[43];
  AI1[45] = (AI1[44] ? AI1[27] : AI1[43]);
  AI1[46] = AI1[24] > AI1[45];
  AI1[47] = (AI1[46] ? AI1[24] : AI1[45]);
  AI1[48] = (long long int) AI1[47];
  AI1[49] = AI1[48] + 4;
  AI1[50] = AI1[49] / 4;
  AI1[51] = (AI1[50] > 0);
  AI1[52] = (AI1[51] ? AI1[50] : 0);
  AI1[53] = AI1[52] - AI1[19];
  AI1[54] = (AI1[53] > 0);
  AI1[55] = AI1[19] + AI1[53];
  AI1[56] = -1 * AI1[53];
  AI1[57] = AI1[54] ? AI1[19] : AI1[55];
  AI1[58] = AI1[54] ? AI1[53] : AI1[56];
  AI1[59] = AI1[1] / 4;
  AI1[60] = (AI1[59] > 0);
  AI1[61] = (AI1[60] ? AI1[59] : 0);
  AI1[62] = AI1[27] + 4;
  AI1[63] = AI1[62] / 4;
  AI1[64] = (AI1[63] > 0);
  AI1[65] = (AI1[64] ? AI1[63] : 0);
  AI1[66] = AI1[65] - AI1[61];
  AI1[67] = (AI1[66] > 0);
  AI1[68] = AI1[61] + AI1[66];
  AI1[69] = -1 * AI1[66];
  AI1[70] = AI1[67] ? AI1[61] : AI1[68];
  AI1[71] = AI1[67] ? AI1[66] : AI1[69];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (input_itemsets + AI1[57]) > (void*) (referrence + AI1[71]))
  || ((void*) (referrence + AI1[70]) > (void*) (input_itemsets + AI1[58])));
  #pragma omp target data map(to: referrence[AI1[70]:AI1[71]]) map(tofrom: input_itemsets[AI1[57]:AI1[58]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < max_cols - 2; i++) {
    for (idx = 0; idx <= i; idx++) {
      index = (idx + 1) * max_cols + (i + 1 - idx);

      int k;
      if ((input_itemsets[index - 1 - max_cols] + referrence[index]) <=
          (input_itemsets[index - 1] - penalty))
        k = (input_itemsets[index - 1] - penalty);
      else
        k = (input_itemsets[index - 1 - max_cols] + referrence[index]);

      if (k <= (input_itemsets[index - max_cols] - penalty))
        input_itemsets[index] = (input_itemsets[index - max_cols] - penalty);
      else
        input_itemsets[index] = k;
    }
  }

  // Compute bottom-right matrix
  long long int AI2[84];
  AI2[0] = max_cols + -3;
  AI2[1] = AI2[0] * max_cols;
  AI2[2] = AI2[1] + 1;
  AI2[3] = AI2[2] * 4;
  AI2[4] = max_cols + -2;
  AI2[5] = AI2[4] * max_cols;
  AI2[6] = AI2[5] + 2;
  AI2[7] = AI2[6] * 4;
  AI2[8] = AI2[1] + 2;
  AI2[9] = AI2[8] * 4;
  AI2[10] = AI2[5] + 1;
  AI2[11] = AI2[10] * 4;
  AI2[12] = AI2[11] < AI2[3];
  AI2[13] = (AI2[12] ? AI2[11] : AI2[3]);
  AI2[14] = AI2[11] < AI2[13];
  AI2[15] = (AI2[14] ? AI2[11] : AI2[13]);
  AI2[16] = AI2[9] < AI2[15];
  AI2[17] = (AI2[16] ? AI2[9] : AI2[15]);
  AI2[18] = AI2[9] < AI2[17];
  AI2[19] = (AI2[18] ? AI2[9] : AI2[17]);
  AI2[20] = AI2[7] < AI2[19];
  AI2[21] = (AI2[20] ? AI2[7] : AI2[19]);
  AI2[22] = AI2[7] < AI2[21];
  AI2[23] = (AI2[22] ? AI2[7] : AI2[21]);
  AI2[24] = AI2[3] < AI2[23];
  AI2[25] = (AI2[24] ? AI2[3] : AI2[23]);
  AI2[26] = AI2[25] / 4;
  AI2[27] = (AI2[26] > 0);
  AI2[28] = (AI2[27] ? AI2[26] : 0);
  AI2[29] = max_cols + -4;
  AI2[30] = AI2[2] + AI2[29];
  AI2[31] = 1 - max_cols;
  AI2[32] = -1 * AI2[29];
  AI2[33] = AI2[29] + AI2[32];
  AI2[34] = AI2[31] * AI2[33];
  AI2[35] = AI2[30] + AI2[34];
  AI2[36] = AI2[35] * 4;
  AI2[37] = AI2[6] + AI2[29];
  AI2[38] = AI2[37] + AI2[34];
  AI2[39] = AI2[38] * 4;
  AI2[40] = AI2[8] + AI2[29];
  AI2[41] = AI2[40] + AI2[34];
  AI2[42] = AI2[41] * 4;
  AI2[43] = AI2[10] + AI2[29];
  AI2[44] = AI2[43] + AI2[34];
  AI2[45] = AI2[44] * 4;
  AI2[46] = AI2[45] > AI2[36];
  AI2[47] = (AI2[46] ? AI2[45] : AI2[36]);
  AI2[48] = AI2[45] > AI2[47];
  AI2[49] = (AI2[48] ? AI2[45] : AI2[47]);
  AI2[50] = AI2[42] > AI2[49];
  AI2[51] = (AI2[50] ? AI2[42] : AI2[49]);
  AI2[52] = AI2[42] > AI2[51];
  AI2[53] = (AI2[52] ? AI2[42] : AI2[51]);
  AI2[54] = AI2[39] > AI2[53];
  AI2[55] = (AI2[54] ? AI2[39] : AI2[53]);
  AI2[56] = AI2[39] > AI2[55];
  AI2[57] = (AI2[56] ? AI2[39] : AI2[55]);
  AI2[58] = AI2[36] > AI2[57];
  AI2[59] = (AI2[58] ? AI2[36] : AI2[57]);
  AI2[60] = (long long int) AI2[59];
  AI2[61] = AI2[60] + 4;
  AI2[62] = AI2[61] / 4;
  AI2[63] = (AI2[62] > 0);
  AI2[64] = (AI2[63] ? AI2[62] : 0);
  AI2[65] = AI2[64] - AI2[28];
  AI2[66] = (AI2[65] > 0);
  AI2[67] = AI2[28] + AI2[65];
  AI2[68] = -1 * AI2[65];
  AI2[69] = AI2[66] ? AI2[28] : AI2[67];
  AI2[70] = AI2[66] ? AI2[65] : AI2[68];
  AI2[71] = AI2[7] / 4;
  AI2[72] = (AI2[71] > 0);
  AI2[73] = (AI2[72] ? AI2[71] : 0);
  AI2[74] = AI2[39] + 4;
  AI2[75] = AI2[74] / 4;
  AI2[76] = (AI2[75] > 0);
  AI2[77] = (AI2[76] ? AI2[75] : 0);
  AI2[78] = AI2[77] - AI2[73];
  AI2[79] = (AI2[78] > 0);
  AI2[80] = AI2[73] + AI2[78];
  AI2[81] = -1 * AI2[78];
  AI2[82] = AI2[79] ? AI2[73] : AI2[80];
  AI2[83] = AI2[79] ? AI2[78] : AI2[81];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (input_itemsets + AI2[69]) > (void*) (referrence + AI2[83]))
  || ((void*) (referrence + AI2[82]) > (void*) (input_itemsets + AI2[70])));
  #pragma omp target data map(to: referrence[AI2[82]:AI2[83]]) map(tofrom: input_itemsets[AI2[69]:AI2[70]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = max_cols - 4; i >= 0; i--) {
    for (idx = 0; idx <= i; idx++) {
      index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;

      int k;
      if ((input_itemsets[index - 1 - max_cols] + referrence[index]) <=
          (input_itemsets[index - 1] - penalty))
        k = (input_itemsets[index - 1] - penalty);
      else
        k = (input_itemsets[index - 1 - max_cols] + referrence[index]);

      if (k <= (input_itemsets[index - max_cols] - penalty))
        input_itemsets[index] = (input_itemsets[index - max_cols] - penalty);
      else
        input_itemsets[index] = k;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int *input_itemsets, int *referrence, int max_rows, int max_cols,
             int penalty, int dev) {

  // Compute top-left matrix
  if (dev == 0)
    runTest_CPU(max_cols, max_rows, input_itemsets, referrence, penalty);
  else
    runTest_GPU(max_cols, max_rows, input_itemsets, referrence, penalty);

//#define TRACEBACK
#ifdef TRACEBACK

  FILE *fpo = fopen("result.txt", "w");
  fprintf(fpo, "print traceback value GPU:\n");

  for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
    int nw, n, w, traceback;
    if (i == max_rows - 2 && j == max_rows - 2)
      fprintf(fpo, "%d ",
              input_itemsets[i * max_cols + j]); // print the first element
    if (i == 0 && j == 0)
      break;
    if (i > 0 && j > 0) {
      nw = input_itemsets[(i - 1) * max_cols + j - 1];
      w = input_itemsets[i * max_cols + j - 1];
      n = input_itemsets[(i - 1) * max_cols + j];
    } else if (i == 0) {
      nw = n = LIMIT;
      w = input_itemsets[i * max_cols + j - 1];
    } else if (j == 0) {
      nw = w = LIMIT;
      n = input_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    // traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + referrence[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if (traceback == new_nw)
      traceback = nw;
    if (traceback == new_w)
      traceback = w;
    if (traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if (traceback == nw) {
      i--;
      j--;
      continue;
    }

    else if (traceback == w) {
      j--;
      continue;
    }

    else if (traceback == n) {
      i--;
      continue;
    }

    else
      ;
  }

  fclose(fpo);

#endif
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  double t_start, t_end;
  int max_rows, max_cols, penalty;
  int *input_itemsets_cpu, *input_itemsets_gpu;
  int *referrence_cpu, *referrence_gpu;

  if (argc == 4) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
  } else {
    usage(argc, argv);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;

  input_itemsets_cpu = (int *)malloc(max_rows * max_cols * sizeof(int));
  input_itemsets_gpu = (int *)malloc(max_rows * max_cols * sizeof(int));

  referrence_cpu = (int *)malloc(max_rows * max_cols * sizeof(int));
  referrence_gpu = (int *)malloc(max_rows * max_cols * sizeof(int));

  if (!input_itemsets_cpu)
    fprintf(stderr, "error: can not allocate memory");

  init(input_itemsets_cpu, input_itemsets_gpu, referrence_cpu, referrence_gpu,
       max_rows, max_rows, penalty);

  printf("Start Needleman-Wunsch\n");

  t_start = rtclock();
  runTest(input_itemsets_cpu, referrence_cpu, max_rows, max_cols, penalty, 0);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  runTest(input_itemsets_gpu, referrence_gpu, max_rows, max_cols, penalty, 1);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(input_itemsets_cpu, input_itemsets_gpu, max_rows, max_cols);

  free(input_itemsets_cpu);
  free(input_itemsets_gpu);
  free(referrence_cpu);
  free(referrence_gpu);

  return EXIT_SUCCESS;
}

