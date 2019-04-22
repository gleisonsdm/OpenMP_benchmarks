/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>
#define PI 3.1415926535897932
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
  return (float)(end_time - start_time) / (1000 * 1000);
}
/**
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value
* > input value
*/
double roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}
/**
* Set values of the 3D array to a newValue if that value is equal to the
* testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY,
           int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}
/**
* Generates a uniformly distributed random number using the provided seed and
* GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((double)M));
}
/**
* Generates a normally distributed random number using the Box-Muller
* transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller
* algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value
* for normal random distribution
*/
double randn(int *seed, int index) {
  /*Box-Muller algorithm*/
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2 * PI * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}
/**
* Sets values of 3D matrix using randomly generated numbers from a normal
* distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            (int)(5 * randn(seed, 0));
      }
    }
  }
}
/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  long long int AI1[10];
  AI1[0] = 2 * radius;
  AI1[1] = AI1[0] + -1;
  AI1[2] = AI1[0] + -2;
  AI1[3] = AI1[1] * AI1[2];
  AI1[4] = AI1[3] + AI1[2];
  AI1[5] = AI1[4] * 4;
  AI1[6] = AI1[5] + 4;
  AI1[7] = AI1[6] / 4;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = (AI1[8] ? AI1[7] : 0);
  #pragma omp target data map(tofrom: disk[0:AI1[9]])
  #pragma omp target
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      double distance = sqrt(pow((double)(x - radius + 1), 2) +
                             pow((double)(y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
    }
  }
}
/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
  int startX = posX - error;
  #pragma omp target data 
  #pragma omp target
  while (startX < 0)
    startX++;
  int startY = posY - error;
  #pragma omp target data 
  #pragma omp target
  while (startY < 0)
    startY++;
  int endX = posX + error;
  #pragma omp target data 
  #pragma omp target
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  #pragma omp target data 
  #pragma omp target
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance =
          sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}
/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}
/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void videoSequence(int *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY / 2.0);
  int y0 = (int)roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  int *newMatrix = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 -
* (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
  double likelihoodSum = 0.0;
  int y;
  for (y = 0; y < numOnes; y++)
    likelihoodSum +=
        (pow((I[ind[y]] - 100), 2) - pow((I[ind[y]] - 228), 2)) / 50.0;
  return likelihoodSum;
}
/**
* Finds the first element in the CDF that is greater than or equal to the
* provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the
* last index
*/
int findIndex(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}
/**
* Finds the first element in the CDF that is greater than or equal to the
* provided value and returns that index
* @note This function uses binary search before switching to sequential search
* @param CDF The CDF
* @param beginIndex The index to start searching from
* @param endIndex The index to stop searching
* @param value The value to find
* @return The index of value in the CDF; if value is never found, returns the
* last index
* @warning Use at your own risk; not fully tested
*/
int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
  if (endIndex < beginIndex)
    return -1;
  int middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
  /*check the value*/
  if (CDF[middleIndex] >= value) {
    /*check that it's good*/
    if (middleIndex == 0)
      return middleIndex;
    else if (CDF[middleIndex - 1] < value)
      return middleIndex;
    else if (CDF[middleIndex - 1] == value) {
      while (middleIndex > 0 && CDF[middleIndex - 1] == value)
        middleIndex--;
      return middleIndex;
    }
  }
  if (CDF[middleIndex] > value)
    return findIndexBin(CDF, beginIndex, middleIndex + 1, value);
  return findIndexBin(CDF, middleIndex - 1, endIndex, value);
}
/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In
* addition, it references a provided MATLAB function which takes the video, the
* objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
*/
void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles) {

  int max_size = IszX * IszY * Nfr;
  long long start = get_time();
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)malloc(diameter * diameter * sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  #pragma omp target data map(tofrom: disk[0:80])
  #pragma omp target
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  double *objxy = (double *)malloc(countOnes * 2 * sizeof(double));
  getneighbors(disk, countOnes, objxy, radius);

  long long get_neighbors = get_time();
  printf("TIME TO GET NEIGHBORS TOOK: %f\n",
         elapsed_time(start, get_neighbors));
  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)malloc(sizeof(double) * Nparticles);
#pragma omp parallel for shared(weights, Nparticles) private(x)
  long long int AI2[6];
  AI2[0] = Nparticles + -1;
  AI2[1] = 8 * AI2[0];
  AI2[2] = AI2[1] + 1;
  AI2[3] = AI2[2] / 8;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  #pragma omp target data map(tofrom: weights[0:AI2[5]])
  #pragma omp target
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }
  long long get_weights = get_time();
  printf("TIME TO GET WEIGHTSTOOK: %f\n",
         elapsed_time(get_neighbors, get_weights));
  // initial likelihood to 0.0
  double *likelihood = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayX = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayY = (double *)malloc(sizeof(double) * Nparticles);
  double *xj = (double *)malloc(sizeof(double) * Nparticles);
  double *yj = (double *)malloc(sizeof(double) * Nparticles);
  double *CDF = (double *)malloc(sizeof(double) * Nparticles);
  double *u = (double *)malloc(sizeof(double) * Nparticles);
  int *ind = (int *)malloc(sizeof(int) * countOnes * Nparticles);
#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
  long long int AI3[6];
  AI3[0] = Nparticles + -1;
  AI3[1] = 8 * AI3[0];
  AI3[2] = AI3[1] + 1;
  AI3[3] = AI3[2] / 8;
  AI3[4] = (AI3[3] > 0);
  AI3[5] = (AI3[4] ? AI3[3] : 0);
  char RST_AI3 = 0;
  RST_AI3 |= !(((void*) (arrayX + 0) > (void*) (arrayY + AI3[5]))
  || ((void*) (arrayY + 0) > (void*) (arrayX + AI3[5])));
  #pragma omp target data map(tofrom: arrayX[0:AI3[5]],arrayY[0:AI3[5]]) if(!RST_AI3)
  #pragma omp target if(!RST_AI3)
  for (x = 0; x < Nparticles; x++) {
    arrayX[x] = xe;
    arrayY[x] = ye;
  }
  int k;

  printf("TIME TO SET ARRAYS TOOK: %f\n",
         elapsed_time(get_weights, get_time()));
  int indX, indY;
  for (k = 1; k < Nfr; k++) {
    long long set_arrays = get_time();
// apply motion model
// draws sample from motion model (random walk). The only prior information
// is that the object moves 2x as fast as in the y direction
#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
    for (x = 0; x < Nparticles; x++) {
      arrayX[x] += 1 + 5 * randn(seed, x);
      arrayY[x] += -2 + 2 * randn(seed, x);
    }
    long long error = get_time();
    printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
// particle filter likelihood
#pragma omp parallel for shared(likelihood, I, arrayX, arrayY, objxy,          \
                                ind) private(x, y, indX, indY)
    for (x = 0; x < Nparticles; x++) {
      // compute the likelihood: remember our assumption is that you know
      // foreground and the background image intensity distribution.
      // Notice that we consider here a likelihood ratio, instead of
      // p(z|x). It is possible in this case. why? a hometask for you.
      // calc ind
      for (y = 0; y < countOnes; y++) {
        indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
        indY = roundDouble(arrayY[x]) + objxy[y * 2];
        ind[x * countOnes + y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
        if (ind[x * countOnes + y] >= max_size)
          ind[x * countOnes + y] = 0;
      }
      likelihood[x] = 0;
      for (y = 0; y < countOnes; y++)
        likelihood[x] += (pow((I[ind[x * countOnes + y]] - 100), 2) -
                          pow((I[ind[x * countOnes + y]] - 228), 2)) /
                         50.0;
      likelihood[x] = likelihood[x] / ((double)countOnes);
    }
    long long likelihood_time = get_time();
    printf("TIME TO GET LIKELIHOODS TOOK: %f\n",
           elapsed_time(error, likelihood_time));
// update & normalize weights
// using equation (63) of Arulampalam Tutorial
#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
    long long int AI4[6];
    AI4[0] = Nparticles + -1;
    AI4[1] = 8 * AI4[0];
    AI4[2] = AI4[1] + 1;
    AI4[3] = AI4[2] / 8;
    AI4[4] = (AI4[3] > 0);
    AI4[5] = (AI4[4] ? AI4[3] : 0);
    char RST_AI4 = 0;
    RST_AI4 |= !(((void*) (likelihood + 0) > (void*) (weights + AI4[5]))
    || ((void*) (weights + 0) > (void*) (likelihood + AI4[5])));
    #pragma omp target data map(tofrom: likelihood[0:AI4[5]],weights[0:AI4[5]]) if(!RST_AI4)
    #pragma omp target if(!RST_AI4)
    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] * exp(likelihood[x]);
    }
    long long exponential = get_time();
    printf("TIME TO GET EXP TOOK: %f\n",
           elapsed_time(likelihood_time, exponential));
    double sumWeights = 0;
#pragma omp parallel for private(x) reduction(+ : sumWeights)
    long long int AI5[6];
    AI5[0] = Nparticles + -1;
    AI5[1] = 8 * AI5[0];
    AI5[2] = AI5[1] + 1;
    AI5[3] = AI5[2] / 8;
    AI5[4] = (AI5[3] > 0);
    AI5[5] = (AI5[4] ? AI5[3] : 0);
    #pragma omp target data map(tofrom: weights[0:AI5[5]])
    #pragma omp target
    for (x = 0; x < Nparticles; x++) {
      sumWeights += weights[x];
    }
    long long sum_time = get_time();
    printf("TIME TO SUM WEIGHTS TOOK: %f\n",
           elapsed_time(exponential, sum_time));
#pragma omp parallel for shared(sumWeights, weights) private(x)
    long long int AI6[6];
    AI6[0] = Nparticles + -1;
    AI6[1] = 8 * AI6[0];
    AI6[2] = AI6[1] + 1;
    AI6[3] = AI6[2] / 8;
    AI6[4] = (AI6[3] > 0);
    AI6[5] = (AI6[4] ? AI6[3] : 0);
    #pragma omp target data map(tofrom: weights[0:AI6[5]])
    #pragma omp target
    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] / sumWeights;
    }
    long long normalize = get_time();
    printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n",
           elapsed_time(sum_time, normalize));
    xe = 0;
    ye = 0;
// estimate the object location by expected values
#pragma omp parallel for private(x) reduction(+ : xe, ye)
    long long int AI7[6];
    AI7[0] = Nparticles + -1;
    AI7[1] = 8 * AI7[0];
    AI7[2] = AI7[1] + 1;
    AI7[3] = AI7[2] / 8;
    AI7[4] = (AI7[3] > 0);
    AI7[5] = (AI7[4] ? AI7[3] : 0);
    char RST_AI7 = 0;
    RST_AI7 |= !(((void*) (arrayX + 0) > (void*) (arrayY + AI7[5]))
    || ((void*) (arrayY + 0) > (void*) (arrayX + AI7[5])));
    RST_AI7 |= !(((void*) (arrayX + 0) > (void*) (weights + AI7[5]))
    || ((void*) (weights + 0) > (void*) (arrayX + AI7[5])));
    RST_AI7 |= !(((void*) (arrayY + 0) > (void*) (weights + AI7[5]))
    || ((void*) (weights + 0) > (void*) (arrayY + AI7[5])));
    #pragma omp target data map(tofrom: arrayX[0:AI7[5]],arrayY[0:AI7[5]],weights[0:AI7[5]]) if(!RST_AI7)
    #pragma omp target if(!RST_AI7)
    for (x = 0; x < Nparticles; x++) {
      xe += arrayX[x] * weights[x];
      ye += arrayY[x] * weights[x];
    }
    long long move_time = get_time();
    printf("TIME TO MOVE OBJECT TOOK: %f\n",
           elapsed_time(normalize, move_time));
    printf("XE: %lf\n", xe);
    printf("YE: %lf\n", ye);
    double distance = sqrt(pow((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                           pow((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
    printf("%lf\n", distance);
    // display(hold off for now)

    // pause(hold off for now)

    // resampling

    CDF[0] = weights[0];
    long long int AI8[19];
    AI8[0] = Nparticles + -2;
    AI8[1] = 8 * AI8[0];
    AI8[2] = 8 + AI8[1];
    AI8[3] = AI8[2] + 1;
    AI8[4] = AI8[3] / 8;
    AI8[5] = (AI8[4] > 0);
    AI8[6] = (AI8[5] ? AI8[4] : 0);
    AI8[7] = AI8[6] - 1;
    AI8[8] = (AI8[7] > 0);
    AI8[9] = 1 + AI8[7];
    AI8[10] = -1 * AI8[7];
    AI8[11] = AI8[8] ? 1 : AI8[9];
    AI8[12] = AI8[8] ? AI8[7] : AI8[10];
    AI8[13] = AI8[2] > AI8[1];
    AI8[14] = (AI8[13] ? AI8[2] : AI8[1]);
    AI8[15] = AI8[14] + 1;
    AI8[16] = AI8[15] / 8;
    AI8[17] = (AI8[16] > 0);
    AI8[18] = (AI8[17] ? AI8[16] : 0);
    char RST_AI8 = 0;
    RST_AI8 |= !(((void*) (CDF + 0) > (void*) (weights + AI8[12]))
    || ((void*) (weights + AI8[11]) > (void*) (CDF + AI8[18])));
    #pragma omp target data map(tofrom: CDF[0:AI8[18]],weights[AI8[11]:AI8[12]]) if(!RST_AI8)
    #pragma omp target if(!RST_AI8)
    for (x = 1; x < Nparticles; x++) {
      CDF[x] = weights[x] + CDF[x - 1];
    }
    long long cum_sum = get_time();
    printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
    double u1 = (1 / ((double)(Nparticles))) * randu(seed, 0);
#pragma omp parallel for shared(u, u1, Nparticles) private(x)
    long long int AI9[6];
    AI9[0] = Nparticles + -1;
    AI9[1] = 8 * AI9[0];
    AI9[2] = AI9[1] + 1;
    AI9[3] = AI9[2] / 8;
    AI9[4] = (AI9[3] > 0);
    AI9[5] = (AI9[4] ? AI9[3] : 0);
    #pragma omp target data map(tofrom: u[0:AI9[5]])
    #pragma omp target
    for (x = 0; x < Nparticles; x++) {
      u[x] = u1 + x / ((double)(Nparticles));
    }
    long long u_time = get_time();
    printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
    int j, i;

#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX,            \
                                arrayY) private(i, j)
    for (j = 0; j < Nparticles; j++) {
      i = findIndex(CDF, Nparticles, u[j]);
      if (i == -1)
        i = Nparticles - 1;
      xj[j] = arrayX[i];
      yj[j] = arrayY[i];
    }
    long long xyj_time = get_time();
    printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n",
           elapsed_time(u_time, xyj_time));

    //#pragma omp parallel for shared(weights, Nparticles) private(x)
    long long int AI10[6];
    AI10[0] = Nparticles + -1;
    AI10[1] = 8 * AI10[0];
    AI10[2] = AI10[1] + 1;
    AI10[3] = AI10[2] / 8;
    AI10[4] = (AI10[3] > 0);
    AI10[5] = (AI10[4] ? AI10[3] : 0);
    char RST_AI10 = 0;
    RST_AI10 |= !(((void*) (arrayX + 0) > (void*) (arrayY + AI10[5]))
    || ((void*) (arrayY + 0) > (void*) (arrayX + AI10[5])));
    RST_AI10 |= !(((void*) (arrayX + 0) > (void*) (weights + AI10[5]))
    || ((void*) (weights + 0) > (void*) (arrayX + AI10[5])));
    RST_AI10 |= !(((void*) (arrayX + 0) > (void*) (xj + AI10[5]))
    || ((void*) (xj + 0) > (void*) (arrayX + AI10[5])));
    RST_AI10 |= !(((void*) (arrayX + 0) > (void*) (yj + AI10[5]))
    || ((void*) (yj + 0) > (void*) (arrayX + AI10[5])));
    RST_AI10 |= !(((void*) (arrayY + 0) > (void*) (weights + AI10[5]))
    || ((void*) (weights + 0) > (void*) (arrayY + AI10[5])));
    RST_AI10 |= !(((void*) (arrayY + 0) > (void*) (xj + AI10[5]))
    || ((void*) (xj + 0) > (void*) (arrayY + AI10[5])));
    RST_AI10 |= !(((void*) (arrayY + 0) > (void*) (yj + AI10[5]))
    || ((void*) (yj + 0) > (void*) (arrayY + AI10[5])));
    RST_AI10 |= !(((void*) (weights + 0) > (void*) (xj + AI10[5]))
    || ((void*) (xj + 0) > (void*) (weights + AI10[5])));
    RST_AI10 |= !(((void*) (weights + 0) > (void*) (yj + AI10[5]))
    || ((void*) (yj + 0) > (void*) (weights + AI10[5])));
    RST_AI10 |= !(((void*) (xj + 0) > (void*) (yj + AI10[5]))
    || ((void*) (yj + 0) > (void*) (xj + AI10[5])));
    #pragma omp target data map(tofrom: arrayX[0:AI10[5]],arrayY[0:AI10[5]],weights[0:AI10[5]],xj[0:AI10[5]],yj[0:AI10[5]]) if(!RST_AI10)
    #pragma omp target if(!RST_AI10)
    for (x = 0; x < Nparticles; x++) {
      // reassign arrayX and arrayY
      arrayX[x] = xj[x];
      arrayY[x] = yj[x];
      weights[x] = 1 / ((double)(Nparticles));
    }
    long long reset = get_time();
    printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
  }
  free(disk);
  free(objxy);
  free(weights);
  free(likelihood);
  free(xj);
  free(yj);
  free(arrayX);
  free(arrayY);
  free(CDF);
  free(u);
  free(ind);
}
int main(int argc, char *argv[]) {

  char *usage = "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  // check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  // check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") ||
      strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  // converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }
  // establish seed
  int *seed = (int *)malloc(sizeof(int) * Nparticles);
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = time(0) * i;
  // malloc matrix
  int *I = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  long long start = get_time();
  // call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
  // call particle filter
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f\n",
         elapsed_time(endVideoSequence, endParticleFilter));
  printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

  free(seed);
  free(I);
  return 0;
}

