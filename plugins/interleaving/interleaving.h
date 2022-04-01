#ifndef __INTERLEAVING_H__
#define __INTERLEAVING_H__

#include "cudaUtility.h"
#include <cublas_v2.h>

/**
 * Function for upsampling float32 image or feature map using nearest neighbor interpolation
 * @ingroup util
 */
template <typename T>
cudaError_t cudaInterleave(T           *input1,
                           T           *input2,
                           T           *input3,
                           T           *input4,
                           int          nChannels,
                           int          inputHeight,
                           int          inputWidth,
                           T           *output,
                           cudaStream_t stream);

#endif
