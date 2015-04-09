/* *
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "sparsematrix.h"
#include "utils.h"

static void *cuda_chk_alloc(const size_t n, /* Number of elements */
	const size_t size /* Size of each element */
	)
{
	void *p;

	cudaError_t error = cudaMalloc((void **)&p, n * size);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc returned error code %d, line(%d)\n", error,
			__LINE__);
		exit(EXIT_FAILURE);
	}

	return p;
}

// x and y is sorted by row
__global__ void initVariableNode(sparseMatrix * const H, float* const l)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= H->n_cols)
		return;

	for (entry *e = sm_first_in_col(H, idx);
		!sm_at_end(e);
		e = sm_next_in_col(H, e))
	{
		e->q = l[idx];
	}
}

__global__ void iterCheckNode(sparseMatrix * const H)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= H->n_rows)
		return;

	for (entry * e = sm_first_in_row(H, idx);
		!sm_at_end(e);
		e = sm_next_in_row(H, e))
	{
		float product = 1;
		for (entry * p = sm_first_in_row(H, idx);
			!sm_at_end(p);
			p = sm_next_in_row(H, p))
		{
			if (p == e)
				continue;
			product *= tanh(p->q / 2);
		}
		e->r = 2 * atanh(product);
		if (isinf(e->r))
		{
			if (e->r < 0)
				e->r = -150;
			else
				e->r = 150;
		}
	}
}

__global__ void iterVariableNode(sparseMatrix * const H, float* const l)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= H->n_cols)
		return;

	float sum = l[idx];
	for (entry *e = sm_first_in_col(H, idx);
		!sm_at_end(e);
		e = sm_next_in_col(H, e))
	{
		sum += e->r;
	}
	for (entry *e = sm_first_in_col(H, idx);
		!sm_at_end(e);
		e = sm_next_in_col(H, e))
	{
		e->q = sum - e->r;
	}
}


__global__ void updateLikelihood(sparseMatrix * const H, float* const l, float* const Q)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= H->n_cols)
		return;

	float sum = l[idx];
	for (entry *e = sm_first_in_col(H, idx);
		!sm_at_end(e);
		e = sm_next_in_col(H, e))
	{
		sum += e->r;
	}
	Q[idx] = sum;
}

__global__ void hardDecision(size_t const N, float* const Q, char* const codeword)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	if (Q[idx] >= 0){
		codeword[idx] = '0';
	}
	else{
		codeword[idx] = '1';
	}
}

__global__ void check(sparseMatrix * const H, char* const codeword, int* const c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= H->n_rows)
		return;

	int sum = 0;
	for (entry * e = sm_first_in_row(H, idx);
		!sm_at_end(e);
		e = sm_next_in_row(H, e))
	{
		sum += codeword[e->col] - '0';
	}
	atomicAdd(c, sum % 2);
}

size_t prprp_decode(size_t M, size_t N, size_t nnz, sparseMatrix * const d_H, float* const d_lratio, float *d_Q, char* const d_codeword, size_t const max_iter)
{
	// check output
	int *d_c = (int *)cuda_chk_alloc(1, sizeof(int));

	/* launch kernel */
	size_t i;
	initVariableNode << <ceil(static_cast<float>(N) / 512.0f), 512 >> >(d_H, d_lratio);
	for (i = 0;; i++)
	{
		iterCheckNode << <ceil(static_cast<float>(M) / 512.0f), 512 >> >(d_H);
		checkCudaErrors(cudaGetLastError());

		iterVariableNode << <ceil(static_cast<float>(N) / 512.0f), 512 >> >(d_H, d_lratio);
		checkCudaErrors(cudaGetLastError());

		updateLikelihood << <ceil(static_cast<float>(N) / 512.0f), 512 >> >(d_H, d_lratio, d_Q);
		checkCudaErrors(cudaGetLastError());

		hardDecision << <ceil(static_cast<float>(N) / 512.0f), 512 >> >(N, d_Q, d_codeword);
		checkCudaErrors(cudaGetLastError());

		int c = 0;
		checkCudaErrors(cudaMemset(d_c, 0, sizeof(int)));

		check << <ceil(static_cast<float>(M) / 512.0f), 512 >> >(d_H, d_codeword, d_c);
		checkCudaErrors(cudaGetLastError());;

		checkCudaErrors(cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
		printf("c is %d\n", c);
		if (i == max_iter || c == 0)
		{
			break;
		}
	}

	/* free device memory */
	checkCudaErrors(cudaFree(d_c));

	return i;
}

sparseMatrix *readSparseMatrix(char *path){
	FILE *f = fopen(path, "r");
	if (f == NULL)
	{
		fprintf(stderr, "Can't open parity check file: %s\n", path);
		fclose(f);
		exit(1);
	}

	int M, N, nnz;
	fscanf(f, "%d %d %d", &M, &N, &nnz);

	int *x = (int *)malloc(nnz*sizeof(int));
	int *y = (int *)malloc(nnz*sizeof(int));
	for (int i = 0; i < nnz; i++)
	{
		fscanf(f, "%d %d", x + i, y + i);
	}
	fclose(f);

	return createSparseMatrix(x, y, M, N, nnz);
}

int main(int argc, char **argv)
{
	// check arg count
	if (argc != 4)
	{
		return 1;
	}

	char *pchk_path = argv[1];
	char *rfile_path = argv[2];
	char *dfile_path = argv[3];


	// open input file
	FILE *rfile, *dfile;
	rfile = fopen(rfile_path, "r");
	if (rfile == NULL)
	{
		fclose(rfile);
		exit(EXIT_FAILURE);
	}
	dfile = fopen(dfile_path, "w+");
	if (dfile == NULL)
	{
		fclose(dfile);
		exit(EXIT_FAILURE);
	}


	// read parity check file into host memory
	sparseMatrix *H = readSparseMatrix(pchk_path);

	// abbreviations
	int M = H->n_rows;
	int N = H->n_cols;
	int nnz = H->nnz;

	/* allocate host memory */
	char *codeword = (char *)malloc(N * sizeof(char)+1);
	float *lratio = (float *)malloc(N * sizeof(float));

	/* allocate device memory */
	// sparse matrix
	sparseMatrix *d_H = (sparseMatrix *)cuda_chk_alloc(1, sizeof(sparseMatrix));
	// log-likelihood ratio
	float *d_lratio = (float *)cuda_chk_alloc(N, sizeof(float));
	// Q
	float *d_Q = (float *)cuda_chk_alloc(N, sizeof(float));
	// hard decision output
	char *d_codeword = (char *)cuda_chk_alloc(N, sizeof(char));

	// copy sparse matrix into device
	cudaCopySparseMatrixH2D(d_H, H);

	// read each block, decode and write
	for (int block_id = 0;; block_id++)
	{
		// read likelihood ratio
		for (int i = 0; i < N; i++)
		{
			int c = fscanf(rfile, "%f", &lratio[i]);
			if (c == EOF)
			{
				if (i > 0)
				{
					printf("Warning: Short block (%d long) at end of received file ignored\n", i);
				}
				goto done;
			}
		}

		/* copy from host to device */
		checkCudaErrors(cudaMemcpy(d_lratio, lratio, N * sizeof(float), cudaMemcpyHostToDevice));
		/* set initial values */
		checkCudaErrors(cudaMemset(d_codeword, 0, N * sizeof(char)));
		checkCudaErrors(cudaMemset(d_Q, 0, N * sizeof(float)));

		// decode
		size_t iters = prprp_decode(M, N, nnz, d_H, d_lratio, d_Q, d_codeword, 50);

		// write output
		checkCudaErrors(cudaMemcpy(codeword, d_codeword, N * sizeof(char), cudaMemcpyDeviceToHost));
		fprintf(dfile, "%s\n", codeword);
	}

done:
	// free file handle
	fclose(rfile);
	fclose(dfile);

	// free host memory
	freeSparseMatrix(H);
	free(codeword);
	free(lratio);

	// free device memory
	cudaFreeSparseMatrix(d_H);
	checkCudaErrors(cudaFree(d_lratio));
	checkCudaErrors(cudaFree(d_Q));
	checkCudaErrors(cudaFree(d_codeword));

	// reset device and wait for exit
	cudaDeviceReset();
	return 0;
}
