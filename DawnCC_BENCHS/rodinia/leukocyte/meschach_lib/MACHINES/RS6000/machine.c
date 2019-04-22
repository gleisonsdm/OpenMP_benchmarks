
/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
**
**			     Meschach Library
**
** This Meschach Library is provided "as is" without any express
** or implied warranty of any kind with respect to this software.
** In particular the authors shall not be liable for any direct,
** indirect, special, incidental or consequential damages arising
** in any way from use of the software.
**
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
***************************************************************************/

/*
  This file contains basic routines which are used by the functions
  in matrix.a etc.
  These are the routines that should be modified in order to take
  full advantage of specialised architectures (pipelining, vector
  processors etc).
  */
static char *rcsid = "$Header: "
                     "/usr/local/home/des/meschach/meschach/RCS/machine.c,v "
                     "1.3 1991/08/29 06:42:11 des Exp $";

#include "machine.h"

/* __ip__ -- inner product */
double __ip__(dp1, dp2, len) register double *dp1, *dp2;
int len;
{
  register int len4;
  register int i;
  register double sum0, sum1, sum2, sum3;

  sum0 = sum1 = sum2 = sum3 = 0.0;

  len4 = len / 4;
  len = len % 4;

  long long int AI1[16];
  AI1[0] = len / 4;
  AI1[1] = AI1[0] + -1;
  AI1[2] = 32 * AI1[1];
  AI1[3] = 24 + AI1[2];
  AI1[4] = 16 + AI1[2];
  AI1[5] = 8 + AI1[2];
  AI1[6] = AI1[5] > AI1[2];
  AI1[7] = (AI1[6] ? AI1[5] : AI1[2]);
  AI1[8] = AI1[4] > AI1[7];
  AI1[9] = (AI1[8] ? AI1[4] : AI1[7]);
  AI1[10] = AI1[3] > AI1[9];
  AI1[11] = (AI1[10] ? AI1[3] : AI1[9]);
  AI1[12] = AI1[11] + 8;
  AI1[13] = AI1[12] / 8;
  AI1[14] = (AI1[13] > 0);
  AI1[15] = (AI1[14] ? AI1[13] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (dp2 + AI1[15]))
  || ((void*) (dp2 + 0) > (void*) (dp1 + AI1[15])));
  #pragma omp target data map(to: dp1[0:AI1[15]],dp2[0:AI1[15]])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len4; i++) {
    sum0 += dp1[4 * i] * dp2[4 * i];
    sum1 += dp1[4 * i + 1] * dp2[4 * i + 1];
    sum2 += dp1[4 * i + 2] * dp2[4 * i + 2];
    sum3 += dp1[4 * i + 3] * dp2[4 * i + 3];
  }
  sum0 += sum1 + sum2 + sum3;
  dp1 += 4 * len4;
  dp2 += 4 * len4;

  long long int AI2[20];
  AI2[0] = len / 4;
  AI2[1] = 4 * AI2[0];
  AI2[2] = AI2[1] * 8;
  AI2[3] = AI2[2] / 8;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  AI2[6] = len % 4;
  AI2[7] = AI2[6] + -1;
  AI2[8] = 8 * AI2[7];
  AI2[9] = AI2[2] + AI2[8];
  AI2[10] = AI2[9] + 8;
  AI2[11] = AI2[10] / 8;
  AI2[12] = (AI2[11] > 0);
  AI2[13] = (AI2[12] ? AI2[11] : 0);
  AI2[14] = AI2[13] - AI2[5];
  AI2[15] = (AI2[14] > 0);
  AI2[16] = AI2[5] + AI2[14];
  AI2[17] = -1 * AI2[14];
  AI2[18] = AI2[15] ? AI2[5] : AI2[16];
  AI2[19] = AI2[15] ? AI2[14] : AI2[17];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (dp1 + AI2[18]) > (void*) (dp2 + AI2[19]))
  || ((void*) (dp2 + AI2[18]) > (void*) (dp1 + AI2[19])));
  #pragma omp target data map(tofrom: dp1[AI2[18]:AI2[19]],dp2[AI2[18]:AI2[19]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < len; i++)
    sum0 += (*dp1++) * (*dp2++);

  return sum0;
}

/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
void __mltadd__(dp1, dp2, s, len) register double *dp1, *dp2, s;
register int len;
{
  register int i, len4;

  len4 = len / 4;
  len = len % 4;
  long long int AI1[30];
  AI1[0] = len / 4;
  AI1[1] = AI1[0] + -1;
  AI1[2] = 32 * AI1[1];
  AI1[3] = 24 + AI1[2];
  AI1[4] = 16 + AI1[2];
  AI1[5] = 8 + AI1[2];
  AI1[6] = AI1[5] > AI1[2];
  AI1[7] = (AI1[6] ? AI1[5] : AI1[2]);
  AI1[8] = AI1[5] > AI1[7];
  AI1[9] = (AI1[8] ? AI1[5] : AI1[7]);
  AI1[10] = AI1[4] > AI1[9];
  AI1[11] = (AI1[10] ? AI1[4] : AI1[9]);
  AI1[12] = AI1[4] > AI1[11];
  AI1[13] = (AI1[12] ? AI1[4] : AI1[11]);
  AI1[14] = AI1[3] > AI1[13];
  AI1[15] = (AI1[14] ? AI1[3] : AI1[13]);
  AI1[16] = AI1[3] > AI1[15];
  AI1[17] = (AI1[16] ? AI1[3] : AI1[15]);
  AI1[18] = AI1[17] + 8;
  AI1[19] = AI1[18] / 8;
  AI1[20] = (AI1[19] > 0);
  AI1[21] = (AI1[20] ? AI1[19] : 0);
  AI1[22] = AI1[4] > AI1[7];
  AI1[23] = (AI1[22] ? AI1[4] : AI1[7]);
  AI1[24] = AI1[3] > AI1[23];
  AI1[25] = (AI1[24] ? AI1[3] : AI1[23]);
  AI1[26] = AI1[25] + 8;
  AI1[27] = AI1[26] / 8;
  AI1[28] = (AI1[27] > 0);
  AI1[29] = (AI1[28] ? AI1[27] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (dp2 + AI1[29]))
  || ((void*) (dp2 + 0) > (void*) (dp1 + AI1[21])));
  #pragma omp target data map(to: dp2[0:AI1[29]]) map(tofrom: dp1[0:AI1[21]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len4; i++) {
    dp1[4 * i] += s * dp2[4 * i];
    dp1[4 * i + 1] += s * dp2[4 * i + 1];
    dp1[4 * i + 2] += s * dp2[4 * i + 2];
    dp1[4 * i + 3] += s * dp2[4 * i + 3];
  }
  dp1 += 4 * len4;
  dp2 += 4 * len4;

  long long int AI2[20];
  AI2[0] = len / 4;
  AI2[1] = 4 * AI2[0];
  AI2[2] = AI2[1] * 8;
  AI2[3] = AI2[2] / 8;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  AI2[6] = len % 4;
  AI2[7] = AI2[6] + -1;
  AI2[8] = 8 * AI2[7];
  AI2[9] = AI2[2] + AI2[8];
  AI2[10] = AI2[9] + 8;
  AI2[11] = AI2[10] / 8;
  AI2[12] = (AI2[11] > 0);
  AI2[13] = (AI2[12] ? AI2[11] : 0);
  AI2[14] = AI2[13] - AI2[5];
  AI2[15] = (AI2[14] > 0);
  AI2[16] = AI2[5] + AI2[14];
  AI2[17] = -1 * AI2[14];
  AI2[18] = AI2[15] ? AI2[5] : AI2[16];
  AI2[19] = AI2[15] ? AI2[14] : AI2[17];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (dp1 + AI2[18]) > (void*) (dp2 + AI2[19]))
  || ((void*) (dp2 + AI2[18]) > (void*) (dp1 + AI2[19])));
  #pragma omp target data map(tofrom: dp1[AI2[18]:AI2[19]],dp2[AI2[18]:AI2[19]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < len; i++)
    (*dp1++) += s * (*dp2++);
}

/* __smlt__ scalar multiply array c.f. sv_mlt() */
void __smlt__(dp, s, out, len) register double *dp, s, *out;
register int len;
{
  register int i;
  long long int AI1[6];
  AI1[0] = len + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = AI1[1] + 8;
  AI1[3] = AI1[2] / 8;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (dp + 0) > (void*) (out + AI1[5]))
  || ((void*) (out + 0) > (void*) (dp + AI1[5])));
  #pragma omp target data map(tofrom: dp[0:AI1[5]],out[0:AI1[5]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++)
    (*out++) = s * (*dp++);
}

/* __add__ -- add arrays c.f. v_add() */
void __add__(dp1, dp2, out, len) register double *dp1, *dp2, *out;
register int len;
{
  register int i;
  long long int AI1[6];
  AI1[0] = len + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = AI1[1] + 8;
  AI1[3] = AI1[2] / 8;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (dp2 + AI1[5]))
  || ((void*) (dp2 + 0) > (void*) (dp1 + AI1[5])));
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (out + AI1[5]))
  || ((void*) (out + 0) > (void*) (dp1 + AI1[5])));
  RST_AI1 |= !(((void*) (dp2 + 0) > (void*) (out + AI1[5]))
  || ((void*) (out + 0) > (void*) (dp2 + AI1[5])));
  #pragma omp target data map(tofrom: dp1[0:AI1[5]],dp2[0:AI1[5]],out[0:AI1[5]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++)
    (*out++) = (*dp1++) + (*dp2++);
}

/* __sub__ -- subtract arrays c.f. v_sub() */
void __sub__(dp1, dp2, out, len) register double *dp1, *dp2, *out;
register int len;
{
  register int i;
  long long int AI1[6];
  AI1[0] = len + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = AI1[1] + 8;
  AI1[3] = AI1[2] / 8;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (dp2 + AI1[5]))
  || ((void*) (dp2 + 0) > (void*) (dp1 + AI1[5])));
  RST_AI1 |= !(((void*) (dp1 + 0) > (void*) (out + AI1[5]))
  || ((void*) (out + 0) > (void*) (dp1 + AI1[5])));
  RST_AI1 |= !(((void*) (dp2 + 0) > (void*) (out + AI1[5]))
  || ((void*) (out + 0) > (void*) (dp2 + AI1[5])));
  #pragma omp target data map(tofrom: dp1[0:AI1[5]],dp2[0:AI1[5]],out[0:AI1[5]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++)
    (*out++) = (*dp1++) - (*dp2++);
}

/* __zero__ -- zeros an array of double precision numbers */
void __zero__(dp, len) register double *dp;
register int len;
{
  /* if a double precision zero is equivalent to a string of nulls */
  MEM_ZERO((char *)dp, len * sizeof(double));
  /* else, need to zero the array entry by entry */
  /*************************************************
    while ( len-- )
    *dp++ = 0.0;
    *************************************************/
}

/***********************************************************************
 ******			Faster versions				********
 ***********************************************************************/

/* __ip4__ -- compute 4 inner products in one go */
void __ip4__(v0, v1, v2, v3, w, out, len) double *v0, *v1, *v2, *v3, *w;
double out[4];
int len;
{
  register int i, len2;
  register double sum00, sum10, sum20, sum30, w_val0;
  register double sum01, sum11, sum21, sum31, w_val1;

  len2 = len / 2;
  len = len % 2;
  sum00 = sum10 = sum20 = sum30 = 0.0;
  sum01 = sum11 = sum21 = sum31 = 0.0;
  long long int AI1[10];
  AI1[0] = len / 2;
  AI1[1] = AI1[0] + -1;
  AI1[2] = 16 * AI1[1];
  AI1[3] = 8 + AI1[2];
  AI1[4] = AI1[3] > AI1[2];
  AI1[5] = (AI1[4] ? AI1[3] : AI1[2]);
  AI1[6] = AI1[5] + 8;
  AI1[7] = AI1[6] / 8;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = (AI1[8] ? AI1[7] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v1 + AI1[9]))
  || ((void*) (v1 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v2 + AI1[9]))
  || ((void*) (v2 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (w + AI1[9]))
  || ((void*) (w + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v2 + AI1[9]))
  || ((void*) (v2 + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (w + AI1[9]))
  || ((void*) (w + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v2 + AI1[9])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (w + AI1[9]))
  || ((void*) (w + 0) > (void*) (v2 + AI1[9])));
  RST_AI1 |= !(((void*) (v3 + 0) > (void*) (w + AI1[9]))
  || ((void*) (w + 0) > (void*) (v3 + AI1[9])));
  #pragma omp target data map(to: v0[0:AI1[9]],v1[0:AI1[9]],v2[0:AI1[9]],v3[0:AI1[9]],w[0:AI1[9]])  if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len2; i++) {
    w_val0 = w[2 * i];
    w_val1 = w[2 * i + 1];
    sum00 += v0[2 * i] * w_val0;
    sum01 += v0[2 * i + 1] * w_val1;
    sum10 += v1[2 * i] * w_val0;
    sum11 += v1[2 * i + 1] * w_val1;
    sum20 += v2[2 * i] * w_val0;
    sum21 += v2[2 * i + 1] * w_val1;
    sum30 += v3[2 * i] * w_val0;
    sum31 += v3[2 * i + 1] * w_val1;
  }
  w += 2 * len2;
  v0 += 2 * len2;
  v1 += 2 * len2;
  v2 += 2 * len2;
  v3 += 2 * len2;
  long long int AI2[20];
  AI2[0] = len / 2;
  AI2[1] = 2 * AI2[0];
  AI2[2] = AI2[1] * 8;
  AI2[3] = AI2[2] / 8;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  AI2[6] = len % 2;
  AI2[7] = AI2[6] + -1;
  AI2[8] = 8 * AI2[7];
  AI2[9] = AI2[2] + AI2[8];
  AI2[10] = AI2[9] + 8;
  AI2[11] = AI2[10] / 8;
  AI2[12] = (AI2[11] > 0);
  AI2[13] = (AI2[12] ? AI2[11] : 0);
  AI2[14] = AI2[13] - AI2[5];
  AI2[15] = (AI2[14] > 0);
  AI2[16] = AI2[5] + AI2[14];
  AI2[17] = -1 * AI2[14];
  AI2[18] = AI2[15] ? AI2[5] : AI2[16];
  AI2[19] = AI2[15] ? AI2[14] : AI2[17];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v1 + AI2[19]))
  || ((void*) (v1 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v2 + AI2[19]))
  || ((void*) (v2 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (v2 + AI2[19]))
  || ((void*) (v2 + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v2 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v2 + AI2[19])));
  RST_AI2 |= !(((void*) (v2 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v2 + AI2[19])));
  RST_AI2 |= !(((void*) (v3 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v3 + AI2[19])));
  #pragma omp target data map(to: v0[AI2[18]:AI2[19]],v1[AI2[18]:AI2[19]],v2[AI2[18]:AI2[19]],v3[AI2[18]:AI2[19]],w[AI2[18]:AI2[19]])  if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < len; i++) {
    w_val0 = w[i];
    sum00 += v0[i] * w_val0;
    sum10 += v1[i] * w_val0;
    sum20 += v2[i] * w_val0;
    sum30 += v3[i] * w_val0;
  }
  out[0] = sum00 + sum01;
  out[1] = sum10 + sum11;
  out[2] = sum20 + sum21;
  out[3] = sum30 + sum31;
}

/* __lc4__ -- linear combinations: w <- w+a[0]*v0+ ... + a[3]*v3 */
void __lc4__(v0, v1, v2, v3, w, a, len) double *v0, *v1, *v2, *v3, *w;
double a[4];
int len;
{
  register int i, len2;
  register double a0, a1, a2, a3, tmp0, tmp1;

  len2 = len / 2;
  len = len % 2;

  a0 = a[0];
  a1 = a[1];
  a2 = a[2];
  a3 = a[3];
  long long int AI1[18];
  AI1[0] = len / 2;
  AI1[1] = AI1[0] + -1;
  AI1[2] = 16 * AI1[1];
  AI1[3] = 8 + AI1[2];
  AI1[4] = AI1[3] > AI1[2];
  AI1[5] = (AI1[4] ? AI1[3] : AI1[2]);
  AI1[6] = AI1[5] + 8;
  AI1[7] = AI1[6] / 8;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = (AI1[8] ? AI1[7] : 0);
  AI1[10] = AI1[2] > AI1[5];
  AI1[11] = (AI1[10] ? AI1[2] : AI1[5]);
  AI1[12] = AI1[3] > AI1[11];
  AI1[13] = (AI1[12] ? AI1[3] : AI1[11]);
  AI1[14] = AI1[13] + 8;
  AI1[15] = AI1[14] / 8;
  AI1[16] = (AI1[15] > 0);
  AI1[17] = (AI1[16] ? AI1[15] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v1 + AI1[9]))
  || ((void*) (v1 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v2 + AI1[9]))
  || ((void*) (v2 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (w + AI1[17]))
  || ((void*) (w + 0) > (void*) (v0 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v2 + AI1[9]))
  || ((void*) (v2 + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (w + AI1[17]))
  || ((void*) (w + 0) > (void*) (v1 + AI1[9])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (v3 + AI1[9]))
  || ((void*) (v3 + 0) > (void*) (v2 + AI1[9])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (w + AI1[17]))
  || ((void*) (w + 0) > (void*) (v2 + AI1[9])));
  RST_AI1 |= !(((void*) (v3 + 0) > (void*) (w + AI1[17]))
  || ((void*) (w + 0) > (void*) (v3 + AI1[9])));
  #pragma omp target data map(to: v0[0:AI1[9]],v1[0:AI1[9]],v2[0:AI1[9]],v3[0:AI1[9]]) map(tofrom: w[0:AI1[17]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len2; i++) {
    tmp0 = w[2 * i] + a0 * v0[2 * i];
    tmp1 = w[2 * i + 1] + a0 * v0[2 * i + 1];
    tmp0 += a1 * v1[2 * i];
    tmp1 += a1 * v1[2 * i + 1];
    tmp0 += a2 * v2[2 * i];
    tmp1 += a2 * v2[2 * i + 1];
    tmp0 += a3 * v3[2 * i];
    tmp1 += a3 * v3[2 * i + 1];
    w[2 * i] = tmp0;
    w[2 * i + 1] = tmp1;
  }
  w += 2 * len2;
  v0 += 2 * len2;
  v1 += 2 * len2;
  v2 += 2 * len2;
  v3 += 2 * len2;
  long long int AI2[20];
  AI2[0] = len / 2;
  AI2[1] = 2 * AI2[0];
  AI2[2] = AI2[1] * 8;
  AI2[3] = AI2[2] / 8;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  AI2[6] = len % 2;
  AI2[7] = AI2[6] + -1;
  AI2[8] = 8 * AI2[7];
  AI2[9] = AI2[2] + AI2[8];
  AI2[10] = AI2[9] + 8;
  AI2[11] = AI2[10] / 8;
  AI2[12] = (AI2[11] > 0);
  AI2[13] = (AI2[12] ? AI2[11] : 0);
  AI2[14] = AI2[13] - AI2[5];
  AI2[15] = (AI2[14] > 0);
  AI2[16] = AI2[5] + AI2[14];
  AI2[17] = -1 * AI2[14];
  AI2[18] = AI2[15] ? AI2[5] : AI2[16];
  AI2[19] = AI2[15] ? AI2[14] : AI2[17];
  char RST_AI2 = 0;
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v1 + AI2[19]))
  || ((void*) (v1 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v2 + AI2[19]))
  || ((void*) (v2 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v0 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v0 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (v2 + AI2[19]))
  || ((void*) (v2 + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v1 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v1 + AI2[19])));
  RST_AI2 |= !(((void*) (v2 + AI2[18]) > (void*) (v3 + AI2[19]))
  || ((void*) (v3 + AI2[18]) > (void*) (v2 + AI2[19])));
  RST_AI2 |= !(((void*) (v2 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v2 + AI2[19])));
  RST_AI2 |= !(((void*) (v3 + AI2[18]) > (void*) (w + AI2[19]))
  || ((void*) (w + AI2[18]) > (void*) (v3 + AI2[19])));
  #pragma omp target data map(to: v0[AI2[18]:AI2[19]],v1[AI2[18]:AI2[19]],v2[AI2[18]:AI2[19]],v3[AI2[18]:AI2[19]]) map(tofrom: w[AI2[18]:AI2[19]]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < len; i++)
    w[i] += a0 * v0[i] + a1 * v1[i] + a2 * v2[i] + a3 * v3[i];
}

/* __ma4__ -- multiply and add with 4 vectors: vi <- vi + ai*w */
void __ma4__(v0, v1, v2, v3, w, a, len) double *v0, *v1, *v2, *v3, *w;
double a[4];
int len;
{
  register int i;
  register double a0, a1, a2, a3, w0, w1, w2, w3;

  a0 = a[0];
  a1 = a[1];
  a2 = a[2];
  a3 = a[3];
  long long int AI1[6];
  AI1[0] = len + -1;
  AI1[1] = 8 * AI1[0];
  AI1[2] = AI1[1] + 8;
  AI1[3] = AI1[2] / 8;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (a + 0) > (void*) (v0 + AI1[5]))
  || ((void*) (v0 + 0) > (void*) (a + 4)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (v1 + AI1[5]))
  || ((void*) (v1 + 0) > (void*) (a + 4)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (v2 + AI1[5]))
  || ((void*) (v2 + 0) > (void*) (a + 4)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (v3 + AI1[5]))
  || ((void*) (v3 + 0) > (void*) (a + 4)));
  RST_AI1 |= !(((void*) (a + 0) > (void*) (w + AI1[5]))
  || ((void*) (w + 0) > (void*) (a + 4)));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v1 + AI1[5]))
  || ((void*) (v1 + 0) > (void*) (v0 + AI1[5])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v2 + AI1[5]))
  || ((void*) (v2 + 0) > (void*) (v0 + AI1[5])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (v3 + AI1[5]))
  || ((void*) (v3 + 0) > (void*) (v0 + AI1[5])));
  RST_AI1 |= !(((void*) (v0 + 0) > (void*) (w + AI1[5]))
  || ((void*) (w + 0) > (void*) (v0 + AI1[5])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v2 + AI1[5]))
  || ((void*) (v2 + 0) > (void*) (v1 + AI1[5])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (v3 + AI1[5]))
  || ((void*) (v3 + 0) > (void*) (v1 + AI1[5])));
  RST_AI1 |= !(((void*) (v1 + 0) > (void*) (w + AI1[5]))
  || ((void*) (w + 0) > (void*) (v1 + AI1[5])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (v3 + AI1[5]))
  || ((void*) (v3 + 0) > (void*) (v2 + AI1[5])));
  RST_AI1 |= !(((void*) (v2 + 0) > (void*) (w + AI1[5]))
  || ((void*) (w + 0) > (void*) (v2 + AI1[5])));
  RST_AI1 |= !(((void*) (v3 + 0) > (void*) (w + AI1[5]))
  || ((void*) (w + 0) > (void*) (v3 + AI1[5])));
  #pragma omp target data map(to: w[0:AI1[5]]) map(tofrom: a[0:4],v0[0:AI1[5]],v1[0:AI1[5]],v2[0:AI1[5]],v3[0:AI1[5]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++) {
    w0 = w[i];
    v0[i] += a0 * w0;
    v1[i] += a1 * w0;
    v2[i] += a2 * w0;
    v3[i] += a3 * w0;
  }
}

