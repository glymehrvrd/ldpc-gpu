#pragma once

#include "stdlib.h"
#include "stdio.h"
#include "cuda_runtime.h"

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
		cudaDeviceReset();
		// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
