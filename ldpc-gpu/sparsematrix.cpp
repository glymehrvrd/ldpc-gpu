
#include "string.h"
#include "utils.h"
#include "sparsematrix.h"


sparseMatrix *createSparseMatrix(int *row, int *col, int n_rows, int n_cols, int nnz)
{
	entry *mem = (entry *)malloc(sizeof(entry)*(nnz + n_rows + n_cols));
	entry *rows = mem;
	entry *cols = mem + n_rows;
	entry *e = mem + n_rows + n_cols;

	memset(mem, 0, sizeof(entry)*(nnz + n_rows + n_cols));
	// initialize row header
	for (int i = 0; i < n_rows; i++)
	{
		rows[i].left = rows + i - mem;
		rows[i].right = rows + i - mem;
		rows[i].up = rows + (n_rows + i - 1) % n_rows - mem;
		rows[i].down = rows + (i + 1) % n_rows - mem;
		rows[i].row = i;
		rows[i].col = -1;
	}

	// intialize column headers
	for (int i = 0; i < n_cols; i++)
	{
		cols[i].left = cols + (n_cols + i - 1) % n_cols - mem;
		cols[i].right = cols + (i + 1) % n_cols - mem;
		cols[i].up = cols + i - mem;
		cols[i].down = cols + i - mem;
		cols[i].row = -1;
		cols[i].col = i;
	}

	for (int i = 0; i < nnz; i++)
	{
		int q;
		entry *p;
		entry *t = e + i;

		// initialize data
		t->row = row[i];
		t->col = col[i];
		t->q = 0;
		t->r = 0;
		t->Q = 0;

		// insert into row
		q = row[i];
		p = rows + row[i];
		while (p->right != q && (mem + p->right)->col < col[i])
		{
			p = mem + p->right;
		}
		t->left = p - mem;
		t->right = p->right;
		(mem + p->right)->left = t - mem;
		p->right = t - mem;

		// insert into column
		q = col[i] + n_rows;
		p = cols + col[i];
		while (p->down != q && (mem + p->down)->row < row[i])
		{
			p = mem + p->down;
		}
		t->up = p - mem;
		t->down = p->down;
		(mem + p->down)->up = t - mem;
		p->down = t - mem;
	}

	sparseMatrix *m = (sparseMatrix *)malloc(sizeof(sparseMatrix));
	m->n_rows = n_rows;
	m->n_cols = n_cols;
	m->rows = rows;
	m->cols = cols;
	m->nnz = nnz;
	m->e = e;
	m->mem = mem;
	return m;
}

void freeSparseMatrix(sparseMatrix *H)
{
	free(H->mem);
	free(H);
}

void cudaFreeSparseMatrix(sparseMatrix *H)
{
	sparseMatrix h_H;
	cudaMemcpy(&h_H, H, sizeof(sparseMatrix), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(h_H.mem));
	checkCudaErrors(cudaFree(H));
}

void cudaCopySparseMatrixH2D(sparseMatrix *dst, sparseMatrix *src)
{
	size_t rows_size = sizeof(entry)* src->n_rows;
	size_t cols_size = sizeof(entry)* src->n_cols;
	size_t e_size = sizeof(entry)* src->nnz;
	size_t mem_size = sizeof(entry)* (src->n_rows + src->n_cols + src->nnz);

	entry *d_mem;
	checkCudaErrors(cudaMalloc(&d_mem, mem_size));
	sparseMatrix tmp;
	tmp.n_rows = src->n_rows;
	tmp.n_cols = src->n_cols;
	tmp.nnz = src->nnz;
	tmp.rows = d_mem;
	tmp.cols = d_mem + src->n_rows;
	tmp.e = d_mem + src->n_rows + src->n_cols;
	tmp.mem = d_mem;

	checkCudaErrors(cudaMemcpy(d_mem, src->mem, mem_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dst, &tmp, sizeof(sparseMatrix), cudaMemcpyHostToDevice));
}