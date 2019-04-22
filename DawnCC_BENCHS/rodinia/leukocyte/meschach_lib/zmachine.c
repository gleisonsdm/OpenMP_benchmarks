
/**************************************************************************
**
** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
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
  involving complex vectors.
  These are the routines that should be modified in order to take
  full advantage of specialised architectures (pipelining, vector
  processors etc).
  */
static char *rcsid = "$Id: zmachine.c,v 1.1 1994/01/13 04:25:41 des Exp $";

#include <math.h>
#include "machine.h"
#include "zmatrix.h"

/* __zconj__ -- complex conjugate */
#ifndef ANSI_C
void __zconj__(zp, len) complex *zp;
int len;
#else
void __zconj__(complex zp[], int len)
#endif
{
  int i;

  long long int AI1[7];
  AI1[0] = len + -1;
  AI1[1] = 16 * AI1[0];
  AI1[2] = 8 + AI1[1];
  AI1[3] = AI1[2] + 16;
  AI1[4] = AI1[3] / 16;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  #pragma omp target data map(tofrom: zp[0:AI1[6]])
  #pragma omp target
  for (i = 0; i < len; i++)
    zp[i].im = -zp[i].im;
}

/* __zip__ -- inner product
        -- computes sum_i zp1[i].zp2[i] if flag == 0
                    sum_i zp1[i]*.zp2[i] if flag != 0 */
#ifndef ANSI_C
complex __zip__(zp1, zp2, len, flag) complex *zp1, *zp2;
int flag, len;
#else
complex __zip__(const complex *zp1, const complex *zp2, int len, int flag)
#endif
{
  complex sum;
  int i;

  sum.re = sum.im = 0.0;
  if (flag) {
    long long int AI1[21];
    AI1[0] = len + -1;
    AI1[1] = 16 * AI1[0];
    AI1[2] = 8 + AI1[1];
    AI1[3] = AI1[2] > AI1[1];
    AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
    AI1[5] = AI1[1] > AI1[4];
    AI1[6] = (AI1[5] ? AI1[1] : AI1[4]);
    AI1[7] = AI1[2] > AI1[6];
    AI1[8] = (AI1[7] ? AI1[2] : AI1[6]);
    AI1[9] = AI1[8] + 16;
    AI1[10] = AI1[9] / 16;
    AI1[11] = (AI1[10] > 0);
    AI1[12] = (AI1[11] ? AI1[10] : 0);
    AI1[13] = AI1[2] > AI1[4];
    AI1[14] = (AI1[13] ? AI1[2] : AI1[4]);
    AI1[15] = AI1[1] > AI1[14];
    AI1[16] = (AI1[15] ? AI1[1] : AI1[14]);
    AI1[17] = AI1[16] + 16;
    AI1[18] = AI1[17] / 16;
    AI1[19] = (AI1[18] > 0);
    AI1[20] = (AI1[19] ? AI1[18] : 0);
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI1[20]))
    || ((void*) (zp2 + 0) > (void*) (zp1 + AI1[12])));
    #pragma omp target data map(to: zp1[0:AI1[12]],zp2[0:AI1[20]])  if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < len; i++) {
      sum.re += zp1[i].re * zp2[i].re + zp1[i].im * zp2[i].im;
      sum.im += zp1[i].re * zp2[i].im - zp1[i].im * zp2[i].re;
    }
  } else {
    long long int AI2[21];
    AI2[0] = len + -1;
    AI2[1] = 16 * AI2[0];
    AI2[2] = 8 + AI2[1];
    AI2[3] = AI2[2] > AI2[1];
    AI2[4] = (AI2[3] ? AI2[2] : AI2[1]);
    AI2[5] = AI2[1] > AI2[4];
    AI2[6] = (AI2[5] ? AI2[1] : AI2[4]);
    AI2[7] = AI2[2] > AI2[6];
    AI2[8] = (AI2[7] ? AI2[2] : AI2[6]);
    AI2[9] = AI2[8] + 16;
    AI2[10] = AI2[9] / 16;
    AI2[11] = (AI2[10] > 0);
    AI2[12] = (AI2[11] ? AI2[10] : 0);
    AI2[13] = AI2[2] > AI2[4];
    AI2[14] = (AI2[13] ? AI2[2] : AI2[4]);
    AI2[15] = AI2[1] > AI2[14];
    AI2[16] = (AI2[15] ? AI2[1] : AI2[14]);
    AI2[17] = AI2[16] + 16;
    AI2[18] = AI2[17] / 16;
    AI2[19] = (AI2[18] > 0);
    AI2[20] = (AI2[19] ? AI2[18] : 0);
    char RST_AI2 = 0;
    RST_AI2 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI2[20]))
    || ((void*) (zp2 + 0) > (void*) (zp1 + AI2[12])));
    #pragma omp target data map(to: zp1[0:AI2[12]],zp2[0:AI2[20]])  if(!RST_AI2)
    #pragma omp target if(!RST_AI2)
    for (i = 0; i < len; i++) {
      sum.re += zp1[i].re * zp2[i].re - zp1[i].im * zp2[i].im;
      sum.im += zp1[i].re * zp2[i].im + zp1[i].im * zp2[i].re;
    }
  }

  return sum;
}

/* __zmltadd__ -- scalar multiply and add i.e. complex saxpy
        -- computes zp1[i] += s.zp2[i]  if flag == 0
        -- computes zp1[i] += s.zp2[i]* if flag != 0 */
#ifndef ANSI_C
void __zmltadd__(zp1, zp2, s, len, flag) complex *zp1, *zp2, s;
int flag, len;
#else
void __zmltadd__(complex *zp1, const complex *zp2, complex s, int len, int flag)
#endif
{
  int i;
  LongReal t_re, t_im;

  if (!flag) {
    long long int AI2[21];
    AI2[0] = len + -1;
    AI2[1] = 16 * AI2[0];
    AI2[2] = 8 + AI2[1];
    AI2[3] = AI2[2] > AI2[1];
    AI2[4] = (AI2[3] ? AI2[2] : AI2[1]);
    AI2[5] = AI2[1] > AI2[4];
    AI2[6] = (AI2[5] ? AI2[1] : AI2[4]);
    AI2[7] = AI2[2] > AI2[6];
    AI2[8] = (AI2[7] ? AI2[2] : AI2[6]);
    AI2[9] = AI2[8] + 16;
    AI2[10] = AI2[9] / 16;
    AI2[11] = (AI2[10] > 0);
    AI2[12] = (AI2[11] ? AI2[10] : 0);
    AI2[13] = AI2[2] > AI2[4];
    AI2[14] = (AI2[13] ? AI2[2] : AI2[4]);
    AI2[15] = AI2[1] > AI2[14];
    AI2[16] = (AI2[15] ? AI2[1] : AI2[14]);
    AI2[17] = AI2[16] + 16;
    AI2[18] = AI2[17] / 16;
    AI2[19] = (AI2[18] > 0);
    AI2[20] = (AI2[19] ? AI2[18] : 0);
    char RST_AI2 = 0;
    RST_AI2 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI2[20]))
    || ((void*) (zp2 + 0) > (void*) (zp1 + AI2[12])));
    #pragma omp target data map(to: zp2[0:AI2[20]]) map(tofrom: zp1[0:AI2[12]]) if(!RST_AI2)
    #pragma omp target if(!RST_AI2)
    for (i = 0; i < len; i++) {
      t_re = zp1[i].re + s.re * zp2[i].re - s.im * zp2[i].im;
      t_im = zp1[i].im + s.re * zp2[i].im + s.im * zp2[i].re;
      zp1[i].re = t_re;
      zp1[i].im = t_im;
    }
  } else {
    long long int AI1[21];
    AI1[0] = len + -1;
    AI1[1] = 16 * AI1[0];
    AI1[2] = 8 + AI1[1];
    AI1[3] = AI1[2] > AI1[1];
    AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
    AI1[5] = AI1[1] > AI1[4];
    AI1[6] = (AI1[5] ? AI1[1] : AI1[4]);
    AI1[7] = AI1[2] > AI1[6];
    AI1[8] = (AI1[7] ? AI1[2] : AI1[6]);
    AI1[9] = AI1[8] + 16;
    AI1[10] = AI1[9] / 16;
    AI1[11] = (AI1[10] > 0);
    AI1[12] = (AI1[11] ? AI1[10] : 0);
    AI1[13] = AI1[2] > AI1[4];
    AI1[14] = (AI1[13] ? AI1[2] : AI1[4]);
    AI1[15] = AI1[1] > AI1[14];
    AI1[16] = (AI1[15] ? AI1[1] : AI1[14]);
    AI1[17] = AI1[16] + 16;
    AI1[18] = AI1[17] / 16;
    AI1[19] = (AI1[18] > 0);
    AI1[20] = (AI1[19] ? AI1[18] : 0);
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI1[20]))
    || ((void*) (zp2 + 0) > (void*) (zp1 + AI1[12])));
    #pragma omp target data map(to: zp2[0:AI1[20]]) map(tofrom: zp1[0:AI1[12]]) if(!RST_AI1)
    #pragma omp target if(!RST_AI1)
    for (i = 0; i < len; i++) {
      t_re = zp1[i].re + s.re * zp2[i].re + s.im * zp2[i].im;
      t_im = zp1[i].im - s.re * zp2[i].im + s.im * zp2[i].re;
      zp1[i].re = t_re;
      zp1[i].im = t_im;
    }
  }
}

/* __zmlt__ scalar complex multiply array c.f. sv_mlt() */
#ifndef ANSI_C
void __zmlt__(zp, s, out, len) complex *zp, s, *out;
register int len;
#else
void __zmlt__(const complex *zp, complex s, complex *out, int len)
#endif
{
  int i;
  LongReal t_re, t_im;

  long long int AI1[17];
  AI1[0] = len + -1;
  AI1[1] = 16 * AI1[0];
  AI1[2] = 8 + AI1[1];
  AI1[3] = AI1[2] > AI1[1];
  AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
  AI1[5] = AI1[2] > AI1[4];
  AI1[6] = (AI1[5] ? AI1[2] : AI1[4]);
  AI1[7] = AI1[1] > AI1[6];
  AI1[8] = (AI1[7] ? AI1[1] : AI1[6]);
  AI1[9] = AI1[8] + 16;
  AI1[10] = AI1[9] / 16;
  AI1[11] = (AI1[10] > 0);
  AI1[12] = (AI1[11] ? AI1[10] : 0);
  AI1[13] = AI1[4] + 16;
  AI1[14] = AI1[13] / 16;
  AI1[15] = (AI1[14] > 0);
  AI1[16] = (AI1[15] ? AI1[14] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (out + 0) > (void*) (zp + AI1[12]))
  || ((void*) (zp + 0) > (void*) (out + AI1[16])));
  #pragma omp target data map(to: zp[0:AI1[12]]) map(tofrom: out[0:AI1[16]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++) {
    t_re = s.re * zp[i].re - s.im * zp[i].im;
    t_im = s.re * zp[i].im + s.im * zp[i].re;
    out[i].re = t_re;
    out[i].im = t_im;
  }
}

/* __zadd__ -- add complex arrays c.f. v_add() */
#ifndef ANSI_C
void __zadd__(zp1, zp2, out, len) complex *zp1, *zp2, *out;
int len;
#else
void __zadd__(const complex *zp1, const complex *zp2, complex *out, int len)
#endif
{
  int i;
  long long int AI1[9];
  AI1[0] = len + -1;
  AI1[1] = 16 * AI1[0];
  AI1[2] = 8 + AI1[1];
  AI1[3] = AI1[2] > AI1[1];
  AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
  AI1[5] = AI1[4] + 16;
  AI1[6] = AI1[5] / 16;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (out + 0) > (void*) (zp1 + AI1[8]))
  || ((void*) (zp1 + 0) > (void*) (out + AI1[8])));
  RST_AI1 |= !(((void*) (out + 0) > (void*) (zp2 + AI1[8]))
  || ((void*) (zp2 + 0) > (void*) (out + AI1[8])));
  RST_AI1 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI1[8]))
  || ((void*) (zp2 + 0) > (void*) (zp1 + AI1[8])));
  #pragma omp target data map(to: zp1[0:AI1[8]],zp2[0:AI1[8]]) map(tofrom: out[0:AI1[8]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++) {
    out[i].re = zp1[i].re + zp2[i].re;
    out[i].im = zp1[i].im + zp2[i].im;
  }
}

/* __zsub__ -- subtract complex arrays c.f. v_sub() */
#ifndef ANSI_C
void __zsub__(zp1, zp2, out, len) complex *zp1, *zp2, *out;
int len;
#else
void __zsub__(const complex *zp1, const complex *zp2, complex *out, int len)
#endif
{
  int i;
  long long int AI1[9];
  AI1[0] = len + -1;
  AI1[1] = 16 * AI1[0];
  AI1[2] = 8 + AI1[1];
  AI1[3] = AI1[2] > AI1[1];
  AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
  AI1[5] = AI1[4] + 16;
  AI1[6] = AI1[5] / 16;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (out + 0) > (void*) (zp1 + AI1[8]))
  || ((void*) (zp1 + 0) > (void*) (out + AI1[8])));
  RST_AI1 |= !(((void*) (out + 0) > (void*) (zp2 + AI1[8]))
  || ((void*) (zp2 + 0) > (void*) (out + AI1[8])));
  RST_AI1 |= !(((void*) (zp1 + 0) > (void*) (zp2 + AI1[8]))
  || ((void*) (zp2 + 0) > (void*) (zp1 + AI1[8])));
  #pragma omp target data map(to: zp1[0:AI1[8]],zp2[0:AI1[8]]) map(tofrom: out[0:AI1[8]]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < len; i++) {
    out[i].re = zp1[i].re - zp2[i].re;
    out[i].im = zp1[i].im - zp2[i].im;
  }
}

/* __zzero__ -- zeros an array of complex numbers */
#ifndef ANSI_C
void __zzero__(zp, len) complex *zp;
int len;
#else
void __zzero__(complex *zp, int len)
#endif
{
  /* if a Real precision zero is equivalent to a string of nulls */
  MEM_ZERO((char *)zp, len * sizeof(complex));
  /* else, need to zero the array entry by entry */
  /******************************
  while ( len-- )
  {
      zp->re = zp->im = 0.0;
      zp++;
  }
  ******************************/
}

