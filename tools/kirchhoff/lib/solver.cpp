/*
 * solver.cpp
 *	Core function to solve a sparse matrix equation Ax=B using Intel MKL
 *  A is an upper triangular of a symmetric square matrix in CSR format
 */


#include<cstdlib>
#include<cstdio>
#include "mkl.h"
//#include "mkl_dss.h"	// Direct Sparse Solver
//#include "mkl_spblas.h"	// sparse matrix-vector multiplication
//#include "mkl_cblas.h"	// vector-vector operations
//#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
    
int dsolve_dss(	int		NCOLS,		//	Total number of rows of square matrix A
				int		NNZ,		//	Total number of nonzero elements of A
				int*	Rptr_A,		//	Row pointers of nonzero elements of A
				int*	Inds_A,		//	Column indices of nonzero elements of A
				double*	ANZ_REAL,	//	LHS matrix A - nonzero elements in CSR format
				double*	RHS_REAL,	//	RHS vector
				double*	SOL_REAL	//	solution vector - output
				) {

	// Assemble complex array from real and imaginary parts:
	double*		LHS = new double[NNZ];
	for(long long int i = 0; i<NNZ; i++) {
		LHS[i] = ANZ_REAL[i];
	}

	double*		RHS = new double[NCOLS];
	for(long long int i = 0; i<NCOLS; i++) {
			RHS[i] = RHS_REAL[i];
	}
	// Solution
	double*		SOL = new double[NCOLS];
	MKL_INT	crt_opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT	sym_opt = MKL_DSS_SYMMETRIC_STRUCTURE;
	MKL_INT	fac_opt = MKL_DSS_INDEFINITE;

	_MKL_DSS_HANDLE_t	handle;
	MKL_INT				error;

	//	initialize solver and set to C-style array indexing:
	error = dss_create(	handle,	crt_opt);

	//	pass locations of nonzero elements of LHS:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_define_structure( handle, sym_opt, Rptr_A, NCOLS, NCOLS, Inds_A, NNZ );
	} else { printf("Error: dss_create failed\n"); }

	//	reorder LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_reorder( handle, crt_opt, 0 );
	} else { printf("Error: define_structure failed\n"); }

	//	Factor LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_factor_real( handle, fac_opt, LHS);
	} else { printf("Error: reorder failed\n"); }

	//	Solve sparse equation:
	if ( error == MKL_DSS_SUCCESS ) {
		MKL_INT nRhs = 1;
		error = dss_solve_real( handle, crt_opt, RHS, nRhs, SOL);
	} else { printf("Error:  factor failed\n"); }

	if ( error == MKL_DSS_SUCCESS ) {
		// Assign real and imaginary parts results to return to python:
		for(int i = 0; i<NCOLS; i++)	{
			SOL_REAL[i] = SOL[i];
		}

        // Compute residual:
		double*		RHS1 = new double[NCOLS];
		char trans = 'N';
		mkl_cspblas_dcsrgemv(&trans,&NCOLS,LHS,Rptr_A,Inds_A,SOL,RHS1);
		cblas_daxpy(NCOLS,-1.0,RHS,1,RHS1,1);
		double norm = cblas_dnrm2(NCOLS,RHS1,1);
		printf("Computed norm(B-X)= %13.10e\n",norm);
        delete [] RHS1;


	} else { printf("Error: solve failed\n"); }
	
	// Free memory
    dss_delete( handle, crt_opt);
    delete [] RHS;
    delete [] LHS;
    delete [] SOL;

	return error;	//MKL_DSS_SUCCESS = 0
}


int zsolve_dss_ge(	int		NCOLS,		//	Total number of rows of square matrix A
				int		NNZ,		//	Total number of nonzero elements of A
				int*	Rptr_A,		//	Row pointers of nonzero elements of A
				int*	Inds_A,		//	Column indices of nonzero elements of A
				double*	ANZ_REAL,	//	LHS matrix A - nonzero elements in CSR format
				double*	ANZ_IMAG,	//
				double*	RHS_REAL,	//	RHS vector
				double*	RHS_IMAG,	//
				double*	SOL_REAL,	//	solution vector - output
				double*	SOL_IMAG	//
				) {

	// Assemble complex array from real and imaginary parts:
	MKL_Complex16*		LHS = new MKL_Complex16[NNZ];
	for(long long int i = 0; i<NNZ; i++) {
		LHS[i].real = ANZ_REAL[i];
		LHS[i].imag = ANZ_IMAG[i];
	}

	MKL_Complex16*		RHS = new MKL_Complex16[NCOLS];
	for(long long int i = 0; i<NCOLS; i++) {
			RHS[i].real = RHS_REAL[i];
			RHS[i].imag = RHS_IMAG[i];
	}
	// Solution
	MKL_Complex16*		SOL = new MKL_Complex16[NCOLS];
	MKL_INT	crt_opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT	sym_opt = MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX;
	MKL_INT	fac_opt = MKL_DSS_INDEFINITE;

	_MKL_DSS_HANDLE_t	handle;
	MKL_INT				error;

	//	initialize solver and set to C-style array indexing:
	error = dss_create(	handle,	crt_opt);

	//	pass locations of nonzero elements of LHS:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_define_structure( handle, sym_opt, Rptr_A, NCOLS, NCOLS, Inds_A, NNZ );
	} else { printf("Error: dss_create failed\n"); }

	//	reorder LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_reorder( handle, crt_opt, 0 );
	} else { printf("Error: define_structure failed\n"); }

	//	Factor LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_factor_complex( handle, fac_opt, LHS);
	} else { printf("Error: reorder failed\n"); }

	//	Solve sparse equation:
	if ( error == MKL_DSS_SUCCESS ) {
		MKL_INT nRhs = 1;
		error = dss_solve_complex( handle, crt_opt, RHS, nRhs, SOL);
	} else { printf("Error:  factor failed\n"); }

	if ( error == MKL_DSS_SUCCESS ) {
		// Assign real and imaginary parts results to return to python:
		for(int i = 0; i<NCOLS; i++)	{
			SOL_REAL[i] = SOL[i].real;
			SOL_IMAG[i] = SOL[i].imag;
		}
		
        // Compute residual:
		MKL_Complex16*		RHS1 = new MKL_Complex16[NCOLS];
		MKL_Complex16 scl; scl.real = -1.0; scl.imag = 0.0;
		char trans = 'N';
		mkl_cspblas_zcsrgemv(&trans,&NCOLS,LHS,Rptr_A,Inds_A,SOL,RHS1);
		cblas_zaxpy(NCOLS,&scl,RHS,1,RHS1,1);
		double norm = cblas_dznrm2(NCOLS,RHS1,1);
		printf("Computed norm(B-X)= %13.10e\n",norm);
        delete [] RHS1;

	} else { printf("Error: solve_complex failed\n"); }
	
	// deallocate internal data structures:
	dss_delete( handle, crt_opt);
    delete [] RHS;
    delete [] LHS;
    delete [] SOL;

	return error;	//MKL_DSS_SUCCESS = 0
}

int zsolve_dss_sy(	int		NCOLS,		//	Total number of rows of square matrix A
				int		NNZ,		//	Total number of nonzero elements of A
				int*	Rptr_A,		//	Row pointers of nonzero elements of A
				int*	Inds_A,		//	Column indices of nonzero elements of A
				double*	ANZ_REAL,	//	LHS matrix A - nonzero elements in CSR format
				double*	ANZ_IMAG,	//
				double*	RHS_REAL,	//	RHS vector
				double*	RHS_IMAG,	//
				double*	SOL_REAL,	//	solution vector - output
				double*	SOL_IMAG	//
				) {

	// Assemble complex array from real and imaginary parts:
	MKL_Complex16*		LHS = new MKL_Complex16[NNZ];
	for(long long int i = 0; i<NNZ; i++) {
		LHS[i].real = ANZ_REAL[i];
		LHS[i].imag = ANZ_IMAG[i];
	}

	MKL_Complex16*		RHS = new MKL_Complex16[NCOLS];
	for(long long int i = 0; i<NCOLS; i++) {
			RHS[i].real = RHS_REAL[i];
			RHS[i].imag = RHS_IMAG[i];
	}
	// Solution
	MKL_Complex16*		SOL = new MKL_Complex16[NCOLS];
	MKL_INT	crt_opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT	sym_opt = MKL_DSS_SYMMETRIC_COMPLEX;
	MKL_INT	fac_opt = MKL_DSS_INDEFINITE;

	_MKL_DSS_HANDLE_t	handle;
	MKL_INT				error;

	//	initialize solver and set to C-style array indexing:
	error = dss_create(	handle,	crt_opt);

	//	pass locations of nonzero elements of LHS:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_define_structure( handle, sym_opt, Rptr_A, NCOLS, NCOLS, Inds_A, NNZ );
	} else { printf("Error: dss_create failed\n"); }

	//	reorder LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_reorder( handle, crt_opt, 0 );
	} else { printf("Error: define_structure failed\n"); }

	//	Factor LHS matrix:
	if ( error == MKL_DSS_SUCCESS ) {
		error = dss_factor_complex( handle, fac_opt, LHS);
	} else { printf("Error: reorder failed\n"); }

	//	Solve sparse equation:
	if ( error == MKL_DSS_SUCCESS ) {
		MKL_INT nRhs = 1;
		error = dss_solve_complex( handle, crt_opt, RHS, nRhs, SOL);
	} else { printf("Error:  factor failed\n"); }

	if ( error == MKL_DSS_SUCCESS ) {
		// Assign real and imaginary parts results to return to python:
		for(int i = 0; i<NCOLS; i++)	{
			SOL_REAL[i] = SOL[i].real;
			SOL_IMAG[i] = SOL[i].imag;
		}

        // Compute residual:
		MKL_Complex16*		RHS1 = new MKL_Complex16[NCOLS];
		MKL_Complex16 scl; scl.real = -1.0; scl.imag = 0.0;
		char uplo = 'U';
		mkl_cspblas_zcsrsymv(&uplo,&NCOLS,LHS,Rptr_A,Inds_A,SOL,RHS1);
		cblas_zaxpy(NCOLS,&scl,RHS,1,RHS1,1);
		double norm = cblas_dznrm2(NCOLS,RHS1,1);
        delete [] RHS1;
		printf("Computed norm(B-X)= %13.10e\n",norm);


	} else { printf("Error: solve_complex failed\n"); }
	
	// deallocate internal data structures:
	dss_delete( handle, crt_opt);
    delete [] RHS;
    delete [] LHS;
    delete [] SOL;

	return error;	//MKL_DSS_SUCCESS = 0
}

int zsolve_pardiso(	int		NCOLS,		//	Total number of rows of square matrix A
				int		NNZ,		//	Total number of nonzero elements of A
				int*	Rptr_A,		//	Row pointers of nonzero elements of A
				int*	Inds_A,		//	Column indices of nonzero elements of A
				double*	ANZ_REAL,	//	LHS matrix A - nonzero elements in CSR format
				double*	ANZ_IMAG,	//
				double*	RHS_REAL,	//	RHS vector
				double*	RHS_IMAG,	//
				double*	SOL_REAL,	//	solution vector - output
				double*	SOL_IMAG	//
				) {

	// PARDISO settings:
	MKL_INT	maxfct		= 1;	// Max factors in memory
	MKL_INT	mnum		= 1;	// N of matrix to solve
	MKL_INT	msglvl		= 0;	// No statistical output
	MKL_INT	mtype		= 6;	// Complex symmetric matrix
	MKL_INT	nRhs		= 1;	// number of right hand sides
	MKL_INT	phase		= 13;	// Analysis, numerical factorization & solve
	MKL_INT perm		= 0;	// perm parameter is ignored unless specified by iparm
	void	*pt[64];			// internal memory pointer - initialize with 0
	//MKL_INT	pt[64]	= {0};		// pardiso settings
	MKL_INT	iparm[64]	= {0};		// pardiso settings
	// Control parameters:
	iparm[0] = 1;	// Do not use defaults
	iparm[1] = 2;	// Fill-in reducing ordering: OpenMP version of nested dissection algorithm
	//iparm[1] = 3;	// Fill-in reducing ordering: OpenMP version of nested dissection algorithm
	iparm[9] = 8;	// permutation threshold E-8 (default for symmetric matrix)
	iparm[20] = 1;	// 1x1 and 2x2 Bunch-Kaufman pivoting - default for mtype=6
	iparm[34] = 1;	// 0-based indexing

	for (int i = 0; i < 64; i++ ) { pt[i] = 0; }

	// Assemble complex array from real and imaginary parts:
	MKL_Complex16*		LHS = new MKL_Complex16[NNZ];
	for(long long int i = 0; i<NNZ; i++) {
		LHS[i].real = ANZ_REAL[i];
		LHS[i].imag = ANZ_IMAG[i];
	}

	MKL_Complex16*		RHS = new MKL_Complex16[NCOLS];
	for(long long int i = 0; i<NCOLS; i++) {
			RHS[i].real = RHS_REAL[i];
			RHS[i].imag = RHS_IMAG[i];
	}
	// Solution
	MKL_Complex16*		SOL = new MKL_Complex16[NCOLS];
	MKL_INT	error = 1;

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &NCOLS, LHS, Rptr_A, Inds_A, &perm, &nRhs, iparm, &msglvl, RHS, SOL, &error );

	if ( error == 0 ) {
		// Assign real and imaginary parts results to return to python:
		for(int i = 0; i<NCOLS; i++)	{
			SOL_REAL[i] = SOL[i].real;
			SOL_IMAG[i] = SOL[i].imag;
		}

        // Compute residual:
		MKL_Complex16*		RHS1 = new MKL_Complex16[NCOLS];
		MKL_Complex16 scl; scl.real = -1.0; scl.imag = 0.0;
		char uplo = 'U';
		mkl_cspblas_zcsrsymv(&uplo,&NCOLS,LHS,Rptr_A,Inds_A,SOL,RHS1);
		cblas_zaxpy(NCOLS,&scl,RHS,1,RHS1,1);
		double norm = cblas_dznrm2(NCOLS,RHS1,1);
        delete [] RHS1;
		printf("Computed norm(B-X)= %13.10e\n",norm);


	} else { printf("Error: solve_complex failed\n"); }

	// deallocate internal data structures:
	double	ddum = 0.0;
	phase = -1;
	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &NCOLS, &ddum, Rptr_A, Inds_A, &perm, &nRhs, iparm, &msglvl, &ddum, &ddum, &error );
    delete [] RHS;
    delete [] LHS;
    delete [] SOL;

	return error;
}

#ifdef __cplusplus
}
#endif


