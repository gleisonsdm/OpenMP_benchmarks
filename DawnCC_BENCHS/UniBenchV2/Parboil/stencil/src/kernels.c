/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

void cpu_stencilGPU(float c0, float c1, float *A0, float *Anext, const int nx,
                    const int ny, const int nz) {

  int i, j, k;

#pragma omp target device(1)
#pragma omp target map(to : A0[ : nx *ny *nz])                                 \
    map(tofrom : Anext[ : nx *ny *nz])
#pragma omp parallel for
  long long int AI1[105];
  AI1[0] = ny + 1;
  AI1[1] = AI1[0] * nx;
  AI1[2] = AI1[1] + 1;
  AI1[3] = AI1[2] * 4;
  AI1[4] = AI1[1] * 4;
  AI1[5] = AI1[1] + 2;
  AI1[6] = AI1[5] * 4;
  AI1[7] = nx * ny;
  AI1[8] = AI1[7] + 1;
  AI1[9] = AI1[8] * 4;
  AI1[10] = ny + 2;
  AI1[11] = AI1[10] * nx;
  AI1[12] = AI1[11] + 1;
  AI1[13] = AI1[12] * 4;
  AI1[14] = nx + 1;
  AI1[15] = AI1[14] * 4;
  AI1[16] = 2 * ny;
  AI1[17] = AI1[16] + 1;
  AI1[18] = AI1[17] * nx;
  AI1[19] = AI1[18] + 1;
  AI1[20] = AI1[19] * 4;
  AI1[21] = AI1[15] < AI1[20];
  AI1[22] = (AI1[21] ? AI1[15] : AI1[20]);
  AI1[23] = AI1[13] < AI1[22];
  AI1[24] = (AI1[23] ? AI1[13] : AI1[22]);
  AI1[25] = AI1[9] < AI1[24];
  AI1[26] = (AI1[25] ? AI1[9] : AI1[24]);
  AI1[27] = AI1[6] < AI1[26];
  AI1[28] = (AI1[27] ? AI1[6] : AI1[26]);
  AI1[29] = AI1[4] < AI1[28];
  AI1[30] = (AI1[29] ? AI1[4] : AI1[28]);
  AI1[31] = AI1[3] < AI1[30];
  AI1[32] = (AI1[31] ? AI1[3] : AI1[30]);
  AI1[33] = AI1[32] / 4;
  AI1[34] = (AI1[33] > 0);
  AI1[35] = (AI1[34] ? AI1[33] : 0);
  AI1[36] = nz + -3;
  AI1[37] = AI1[7] * AI1[36];
  AI1[38] = AI1[2] + AI1[37];
  AI1[39] = ny + -3;
  AI1[40] = nx * AI1[39];
  AI1[41] = AI1[38] + AI1[40];
  AI1[42] = nx + -3;
  AI1[43] = AI1[41] + AI1[42];
  AI1[44] = AI1[43] * 4;
  AI1[45] = AI1[1] + AI1[37];
  AI1[46] = AI1[45] + AI1[40];
  AI1[47] = AI1[46] + AI1[42];
  AI1[48] = AI1[47] * 4;
  AI1[49] = AI1[5] + AI1[37];
  AI1[50] = AI1[49] + AI1[40];
  AI1[51] = AI1[50] + AI1[42];
  AI1[52] = AI1[51] * 4;
  AI1[53] = AI1[8] + AI1[37];
  AI1[54] = AI1[53] + AI1[40];
  AI1[55] = AI1[54] + AI1[42];
  AI1[56] = AI1[55] * 4;
  AI1[57] = AI1[12] + AI1[37];
  AI1[58] = AI1[57] + AI1[40];
  AI1[59] = AI1[58] + AI1[42];
  AI1[60] = AI1[59] * 4;
  AI1[61] = AI1[14] + AI1[37];
  AI1[62] = AI1[61] + AI1[40];
  AI1[63] = AI1[62] + AI1[42];
  AI1[64] = AI1[63] * 4;
  AI1[65] = AI1[19] + AI1[37];
  AI1[66] = AI1[65] + AI1[40];
  AI1[67] = AI1[66] + AI1[42];
  AI1[68] = AI1[67] * 4;
  AI1[69] = AI1[64] > AI1[68];
  AI1[70] = (AI1[69] ? AI1[64] : AI1[68]);
  AI1[71] = AI1[60] > AI1[70];
  AI1[72] = (AI1[71] ? AI1[60] : AI1[70]);
  AI1[73] = AI1[56] > AI1[72];
  AI1[74] = (AI1[73] ? AI1[56] : AI1[72]);
  AI1[75] = AI1[52] > AI1[74];
  AI1[76] = (AI1[75] ? AI1[52] : AI1[74]);
  AI1[77] = AI1[48] > AI1[76];
  AI1[78] = (AI1[77] ? AI1[48] : AI1[76]);
  AI1[79] = AI1[44] > AI1[78];
  AI1[80] = (AI1[79] ? AI1[44] : AI1[78]);
  AI1[81] = (long long int) AI1[80];
  AI1[82] = AI1[81] + 4;
  AI1[83] = AI1[82] / 4;
  AI1[84] = (AI1[83] > 0);
  AI1[85] = (AI1[84] ? AI1[83] : 0);
  AI1[86] = AI1[85] - AI1[35];
  AI1[87] = (AI1[86] > 0);
  AI1[88] = AI1[35] + AI1[86];
  AI1[89] = -1 * AI1[86];
  AI1[90] = AI1[87] ? AI1[35] : AI1[88];
  AI1[91] = AI1[87] ? AI1[86] : AI1[89];
  AI1[92] = AI1[3] / 4;
  AI1[93] = (AI1[92] > 0);
  AI1[94] = (AI1[93] ? AI1[92] : 0);
  AI1[95] = AI1[44] + 4;
  AI1[96] = AI1[95] / 4;
  AI1[97] = (AI1[96] > 0);
  AI1[98] = (AI1[97] ? AI1[96] : 0);
  AI1[99] = AI1[98] - AI1[94];
  AI1[100] = (AI1[99] > 0);
  AI1[101] = AI1[94] + AI1[99];
  AI1[102] = -1 * AI1[99];
  AI1[103] = AI1[100] ? AI1[94] : AI1[101];
  AI1[104] = AI1[100] ? AI1[99] : AI1[102];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A0 + AI1[90]) > (void*) (Anext + AI1[104]))
  || ((void*) (Anext + AI1[103]) > (void*) (A0 + AI1[91])));
  #pragma omp target data map(to: A0[AI1[90]:AI1[91]]) map(tofrom: Anext[AI1[103]:AI1[104]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (k = 1; k < nz - 1; k++) {
    for (j = 1; j < ny - 1; j++) {
      for (i = 1; i < nx - 1; i++) {
        Anext[Index3D(nx, ny, i, j, k)] = (A0[Index3D(nx, ny, i, j, k + 1)] +
                                           A0[Index3D(nx, ny, i, j, k - 1)] +
                                           A0[Index3D(nx, ny, i, j + 1, k)] +
                                           A0[Index3D(nx, ny, i, j - 1, k)] +
                                           A0[Index3D(nx, ny, i + 1, j, k)] +
                                           A0[Index3D(nx, ny, i - 1, j, k)]) *
                                              c1 -
                                          A0[Index3D(nx, ny, i, j, k)] * c0;
      }
    }
  }
}

void cpu_stencilCPU(float c0, float c1, float *A0, float *Anext, const int nx,
                    const int ny, const int nz) {

  int i, j, k;
  long long int AI1[105];
  AI1[0] = ny + 1;
  AI1[1] = AI1[0] * nx;
  AI1[2] = AI1[1] + 1;
  AI1[3] = AI1[2] * 4;
  AI1[4] = AI1[1] * 4;
  AI1[5] = AI1[1] + 2;
  AI1[6] = AI1[5] * 4;
  AI1[7] = nx * ny;
  AI1[8] = AI1[7] + 1;
  AI1[9] = AI1[8] * 4;
  AI1[10] = ny + 2;
  AI1[11] = AI1[10] * nx;
  AI1[12] = AI1[11] + 1;
  AI1[13] = AI1[12] * 4;
  AI1[14] = nx + 1;
  AI1[15] = AI1[14] * 4;
  AI1[16] = 2 * ny;
  AI1[17] = AI1[16] + 1;
  AI1[18] = AI1[17] * nx;
  AI1[19] = AI1[18] + 1;
  AI1[20] = AI1[19] * 4;
  AI1[21] = AI1[15] < AI1[20];
  AI1[22] = (AI1[21] ? AI1[15] : AI1[20]);
  AI1[23] = AI1[13] < AI1[22];
  AI1[24] = (AI1[23] ? AI1[13] : AI1[22]);
  AI1[25] = AI1[9] < AI1[24];
  AI1[26] = (AI1[25] ? AI1[9] : AI1[24]);
  AI1[27] = AI1[6] < AI1[26];
  AI1[28] = (AI1[27] ? AI1[6] : AI1[26]);
  AI1[29] = AI1[4] < AI1[28];
  AI1[30] = (AI1[29] ? AI1[4] : AI1[28]);
  AI1[31] = AI1[3] < AI1[30];
  AI1[32] = (AI1[31] ? AI1[3] : AI1[30]);
  AI1[33] = AI1[32] / 4;
  AI1[34] = (AI1[33] > 0);
  AI1[35] = (AI1[34] ? AI1[33] : 0);
  AI1[36] = nz + -3;
  AI1[37] = AI1[7] * AI1[36];
  AI1[38] = AI1[2] + AI1[37];
  AI1[39] = ny + -3;
  AI1[40] = nx * AI1[39];
  AI1[41] = AI1[38] + AI1[40];
  AI1[42] = nx + -3;
  AI1[43] = AI1[41] + AI1[42];
  AI1[44] = AI1[43] * 4;
  AI1[45] = AI1[1] + AI1[37];
  AI1[46] = AI1[45] + AI1[40];
  AI1[47] = AI1[46] + AI1[42];
  AI1[48] = AI1[47] * 4;
  AI1[49] = AI1[5] + AI1[37];
  AI1[50] = AI1[49] + AI1[40];
  AI1[51] = AI1[50] + AI1[42];
  AI1[52] = AI1[51] * 4;
  AI1[53] = AI1[8] + AI1[37];
  AI1[54] = AI1[53] + AI1[40];
  AI1[55] = AI1[54] + AI1[42];
  AI1[56] = AI1[55] * 4;
  AI1[57] = AI1[12] + AI1[37];
  AI1[58] = AI1[57] + AI1[40];
  AI1[59] = AI1[58] + AI1[42];
  AI1[60] = AI1[59] * 4;
  AI1[61] = AI1[14] + AI1[37];
  AI1[62] = AI1[61] + AI1[40];
  AI1[63] = AI1[62] + AI1[42];
  AI1[64] = AI1[63] * 4;
  AI1[65] = AI1[19] + AI1[37];
  AI1[66] = AI1[65] + AI1[40];
  AI1[67] = AI1[66] + AI1[42];
  AI1[68] = AI1[67] * 4;
  AI1[69] = AI1[64] > AI1[68];
  AI1[70] = (AI1[69] ? AI1[64] : AI1[68]);
  AI1[71] = AI1[60] > AI1[70];
  AI1[72] = (AI1[71] ? AI1[60] : AI1[70]);
  AI1[73] = AI1[56] > AI1[72];
  AI1[74] = (AI1[73] ? AI1[56] : AI1[72]);
  AI1[75] = AI1[52] > AI1[74];
  AI1[76] = (AI1[75] ? AI1[52] : AI1[74]);
  AI1[77] = AI1[48] > AI1[76];
  AI1[78] = (AI1[77] ? AI1[48] : AI1[76]);
  AI1[79] = AI1[44] > AI1[78];
  AI1[80] = (AI1[79] ? AI1[44] : AI1[78]);
  AI1[81] = (long long int) AI1[80];
  AI1[82] = AI1[81] + 4;
  AI1[83] = AI1[82] / 4;
  AI1[84] = (AI1[83] > 0);
  AI1[85] = (AI1[84] ? AI1[83] : 0);
  AI1[86] = AI1[85] - AI1[35];
  AI1[87] = (AI1[86] > 0);
  AI1[88] = AI1[35] + AI1[86];
  AI1[89] = -1 * AI1[86];
  AI1[90] = AI1[87] ? AI1[35] : AI1[88];
  AI1[91] = AI1[87] ? AI1[86] : AI1[89];
  AI1[92] = AI1[3] / 4;
  AI1[93] = (AI1[92] > 0);
  AI1[94] = (AI1[93] ? AI1[92] : 0);
  AI1[95] = AI1[44] + 4;
  AI1[96] = AI1[95] / 4;
  AI1[97] = (AI1[96] > 0);
  AI1[98] = (AI1[97] ? AI1[96] : 0);
  AI1[99] = AI1[98] - AI1[94];
  AI1[100] = (AI1[99] > 0);
  AI1[101] = AI1[94] + AI1[99];
  AI1[102] = -1 * AI1[99];
  AI1[103] = AI1[100] ? AI1[94] : AI1[101];
  AI1[104] = AI1[100] ? AI1[99] : AI1[102];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A0 + AI1[90]) > (void*) (Anext + AI1[104]))
  || ((void*) (Anext + AI1[103]) > (void*) (A0 + AI1[91])));
  #pragma omp target data map(to: A0[AI1[90]:AI1[91]]) map(tofrom: Anext[AI1[103]:AI1[104]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (k = 1; k < nz - 1; k++) {
    for (j = 1; j < ny - 1; j++) {
      for (i = 1; i < nx - 1; i++) {
        Anext[Index3D(nx, ny, i, j, k)] = (A0[Index3D(nx, ny, i, j, k + 1)] +
                                           A0[Index3D(nx, ny, i, j, k - 1)] +
                                           A0[Index3D(nx, ny, i, j + 1, k)] +
                                           A0[Index3D(nx, ny, i, j - 1, k)] +
                                           A0[Index3D(nx, ny, i + 1, j, k)] +
                                           A0[Index3D(nx, ny, i - 1, j, k)]) *
                                              c1 -
                                          A0[Index3D(nx, ny, i, j, k)] * c0;
      }
    }
  }
}

