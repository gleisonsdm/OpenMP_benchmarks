#include <stdio.h>
#include <omp.h>
#define GPU_DEVICE 1

void lud_omp_cpu(float *a, int size) {
  int i, j, k;
  float sum;

  long long int AI1[69];
  AI1[0] = size * 4;
  AI1[1] = AI1[0] < 0;
  AI1[2] = (AI1[1] ? AI1[0] : 0);
  AI1[3] = AI1[0] < AI1[2];
  AI1[4] = (AI1[3] ? AI1[0] : AI1[2]);
  AI1[5] = 0 < AI1[4];
  AI1[6] = (AI1[5] ? 0 : AI1[4]);
  AI1[7] = 0 < AI1[6];
  AI1[8] = (AI1[7] ? 0 : AI1[6]);
  AI1[9] = AI1[0] < AI1[8];
  AI1[10] = (AI1[9] ? AI1[0] : AI1[8]);
  AI1[11] = AI1[10] / 4;
  AI1[12] = (AI1[11] > 0);
  AI1[13] = (AI1[12] ? AI1[11] : 0);
  AI1[14] = size + 1;
  AI1[15] = size + -1;
  AI1[16] = AI1[14] * AI1[15];
  AI1[17] = size + AI1[16];
  AI1[18] = size + -2;
  AI1[19] = -1 * AI1[15];
  AI1[20] = AI1[18] + AI1[19];
  AI1[21] = size * AI1[20];
  AI1[22] = AI1[17] + AI1[21];
  AI1[23] = AI1[22] * 4;
  AI1[24] = AI1[16] * 4;
  AI1[25] = -1 + AI1[15];
  AI1[26] = size * AI1[25];
  AI1[27] = AI1[15] + AI1[26];
  AI1[28] = AI1[27] * 4;
  AI1[29] = size * AI1[15];
  AI1[30] = size + AI1[29];
  AI1[31] = AI1[30] + AI1[21];
  AI1[32] = AI1[31] + AI1[25];
  AI1[33] = AI1[32] * 4;
  AI1[34] = AI1[15] + AI1[19];
  AI1[35] = AI1[16] + AI1[34];
  AI1[36] = AI1[35] * 4;
  AI1[37] = AI1[15] + AI1[34];
  AI1[38] = AI1[37] + AI1[26];
  AI1[39] = AI1[38] * 4;
  AI1[40] = AI1[29] + AI1[25];
  AI1[41] = AI1[40] * 4;
  AI1[42] = AI1[41] > AI1[36];
  AI1[43] = (AI1[42] ? AI1[41] : AI1[36]);
  AI1[44] = AI1[39] > AI1[43];
  AI1[45] = (AI1[44] ? AI1[39] : AI1[43]);
  AI1[46] = AI1[36] > AI1[45];
  AI1[47] = (AI1[46] ? AI1[36] : AI1[45]);
  AI1[48] = AI1[23] > AI1[47];
  AI1[49] = (AI1[48] ? AI1[23] : AI1[47]);
  AI1[50] = AI1[33] > AI1[49];
  AI1[51] = (AI1[50] ? AI1[33] : AI1[49]);
  AI1[52] = AI1[28] > AI1[51];
  AI1[53] = (AI1[52] ? AI1[28] : AI1[51]);
  AI1[54] = AI1[24] > AI1[53];
  AI1[55] = (AI1[54] ? AI1[24] : AI1[53]);
  AI1[56] = AI1[23] > AI1[55];
  AI1[57] = (AI1[56] ? AI1[23] : AI1[55]);
  AI1[58] = (long long int) AI1[57];
  AI1[59] = AI1[58] + 4;
  AI1[60] = AI1[59] / 4;
  AI1[61] = (AI1[60] > 0);
  AI1[62] = (AI1[61] ? AI1[60] : 0);
  AI1[63] = AI1[62] - AI1[13];
  AI1[64] = (AI1[63] > 0);
  AI1[65] = AI1[13] + AI1[63];
  AI1[66] = -1 * AI1[63];
  AI1[67] = AI1[64] ? AI1[13] : AI1[65];
  AI1[68] = AI1[64] ? AI1[63] : AI1[66];
  #pragma omp target data map(tofrom: a[AI1[67]:AI1[68]])
  #pragma omp target
  for (i = 0; i < size; i++) {
    for (j = i; j < size; j++) {
      sum = a[i * size + j];
      for (k = 0; k < i; k++)
        sum -= a[i * size + k] * a[k * size + j];
      a[i * size + j] = sum;
    }

    for (j = i + 1; j < size; j++) {
      sum = a[j * size + i];
      for (k = 0; k < i; k++)
        sum -= a[j * size + k] * a[k * size + i];
      a[j * size + i] = sum / a[i * size + i];
    }
  }
}

void lud_omp_gpu(float *a, int size) {
  int i, j, k;
  float sum;

#pragma omp target device(GPU_DEVICE)
#pragma omp target map(tofrom : a[0 : size *size])
  {
    long long int AI1[69];
    AI1[0] = size * 4;
    AI1[1] = AI1[0] < 0;
    AI1[2] = (AI1[1] ? AI1[0] : 0);
    AI1[3] = AI1[0] < AI1[2];
    AI1[4] = (AI1[3] ? AI1[0] : AI1[2]);
    AI1[5] = 0 < AI1[4];
    AI1[6] = (AI1[5] ? 0 : AI1[4]);
    AI1[7] = 0 < AI1[6];
    AI1[8] = (AI1[7] ? 0 : AI1[6]);
    AI1[9] = AI1[0] < AI1[8];
    AI1[10] = (AI1[9] ? AI1[0] : AI1[8]);
    AI1[11] = AI1[10] / 4;
    AI1[12] = (AI1[11] > 0);
    AI1[13] = (AI1[12] ? AI1[11] : 0);
    AI1[14] = size + 1;
    AI1[15] = size + -1;
    AI1[16] = AI1[14] * AI1[15];
    AI1[17] = size + AI1[16];
    AI1[18] = size + -2;
    AI1[19] = -1 * AI1[15];
    AI1[20] = AI1[18] + AI1[19];
    AI1[21] = size * AI1[20];
    AI1[22] = AI1[17] + AI1[21];
    AI1[23] = AI1[22] * 4;
    AI1[24] = AI1[16] * 4;
    AI1[25] = -1 + AI1[15];
    AI1[26] = size * AI1[25];
    AI1[27] = AI1[15] + AI1[26];
    AI1[28] = AI1[27] * 4;
    AI1[29] = size * AI1[15];
    AI1[30] = size + AI1[29];
    AI1[31] = AI1[30] + AI1[21];
    AI1[32] = AI1[31] + AI1[25];
    AI1[33] = AI1[32] * 4;
    AI1[34] = AI1[15] + AI1[19];
    AI1[35] = AI1[16] + AI1[34];
    AI1[36] = AI1[35] * 4;
    AI1[37] = AI1[15] + AI1[34];
    AI1[38] = AI1[37] + AI1[26];
    AI1[39] = AI1[38] * 4;
    AI1[40] = AI1[29] + AI1[25];
    AI1[41] = AI1[40] * 4;
    AI1[42] = AI1[41] > AI1[36];
    AI1[43] = (AI1[42] ? AI1[41] : AI1[36]);
    AI1[44] = AI1[39] > AI1[43];
    AI1[45] = (AI1[44] ? AI1[39] : AI1[43]);
    AI1[46] = AI1[36] > AI1[45];
    AI1[47] = (AI1[46] ? AI1[36] : AI1[45]);
    AI1[48] = AI1[23] > AI1[47];
    AI1[49] = (AI1[48] ? AI1[23] : AI1[47]);
    AI1[50] = AI1[33] > AI1[49];
    AI1[51] = (AI1[50] ? AI1[33] : AI1[49]);
    AI1[52] = AI1[28] > AI1[51];
    AI1[53] = (AI1[52] ? AI1[28] : AI1[51]);
    AI1[54] = AI1[24] > AI1[53];
    AI1[55] = (AI1[54] ? AI1[24] : AI1[53]);
    AI1[56] = AI1[23] > AI1[55];
    AI1[57] = (AI1[56] ? AI1[23] : AI1[55]);
    AI1[58] = (long long int) AI1[57];
    AI1[59] = AI1[58] + 4;
    AI1[60] = AI1[59] / 4;
    AI1[61] = (AI1[60] > 0);
    AI1[62] = (AI1[61] ? AI1[60] : 0);
    AI1[63] = AI1[62] - AI1[13];
    AI1[64] = (AI1[63] > 0);
    AI1[65] = AI1[13] + AI1[63];
    AI1[66] = -1 * AI1[63];
    AI1[67] = AI1[64] ? AI1[13] : AI1[65];
    AI1[68] = AI1[64] ? AI1[63] : AI1[66];
    #pragma omp target data map(tofrom: a[AI1[67]:AI1[68]])
    #pragma omp target
    for (i = 0; i < size; i++) {
#pragma omp parallel for
      for (j = i; j < size; j++) {
        sum = a[i * size + j];
        for (k = 0; k < i; k++)
          sum -= a[i * size + k] * a[k * size + j];
        a[i * size + j] = sum;
      }

#pragma omp parallel for
      for (j = i + 1; j < size; j++) {
        sum = a[j * size + i];
        for (k = 0; k < i; k++)
          sum -= a[j * size + k] * a[k * size + i];
        a[j * size + i] = sum / a[i * size + i];
      }
    }
  }
}

