#include "stdlib.h"

struct entry
{
	int row, col;
	int left, right, up, down;
	double q, r, Q;
};

struct sparseMatrix		/* Representation of a sparse matrix */
{
	int n_rows;		  /* Number of rows in the matrix */
	int n_cols;		  /* Number of columns in the matrix */
	int nnz;

	entry *rows;	  /* Pointer to array of row headers */
	entry *cols;	  /* Pointer to array of column headers */
	entry *e;
	entry *mem;
};

#define sm_first_in_row(m,i) ((m)->mem + (m)->rows[i].right) /* Find the first   */
#define sm_first_in_col(m,j) ((m)->mem + (m)->cols[j].down)  /* or last entry in */
#define sm_last_in_row(m,i) ((m)->mem + (m)->rows[i].left)   /* a row or column  */
#define sm_last_in_col(m,j) ((m)->mem + (m)->cols[j].up)

#define sm_next_in_row(m,e) ((m)->mem + (e)->right)  /* Move from one entry to     */
#define sm_next_in_col(m,e) ((m)->mem + (e)->down)   /* another in any of the four */
#define sm_prev_in_row(m,e) ((m)->mem + (e)->left)   /* possible directions        */
#define sm_prev_in_col(m,e) ((m)->mem + (e)->up)   

#define sm_at_end(e) ((e)->row<0 || (e)->col<0) /* See if we've reached the end     */

#define sm_row(e) ((e)->row)      /* Find out the row or column index */
#define sm_col(e) ((e)->col)      /* of an entry (indexes start at 0) */

#define sm_rows(m) ((m)->n_rows)  /* Get the number of rows or columns*/
#define sm_cols(m) ((m)->n_cols)  /* in a matrix                      */

sparseMatrix *createSparseMatrix(int *row, int *col, int n_rows, int n_cols, int nnz);
void freeSparseMatrix(sparseMatrix *H);
void cudaFreeSparseMatrix(sparseMatrix *H);
void cudaCopySparseMatrixH2D(sparseMatrix *dst, sparseMatrix *src);