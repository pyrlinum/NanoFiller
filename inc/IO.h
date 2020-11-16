// This file describes functions used to set up generation
#pragma once
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>
#include <math.h>
#include "CNT.h"
using namespace std;

// IO functions:

float *read_density(const char* filename, int *ext, float *dim);	// read probability density
void write_dat(const char *filename, int size, float *data);
void write_dat(const char *filename, int size, double *data);
void write_dat(const char *filename, int size, short *data);
void write_dat(const char *filename, int size, int *data);
void write_dat(const char *filename, int size, unsigned int *data);
void write_mat(const char *filename, int size, double *data);
void write_NInput_dat(const char *filename, int Ncol, int size, float *data);
void write_CNTvtk(const char *filename, int cntNum, CNT_t *cnt_arr);			// print created cnts in vtk format
void write_CNT_ORTvtk(const char *filename, int cntNum, CNT_t *cnt_arr);
void write_CNTlmp(const char *filename, int cntNum, CNT_t *cnt_arr, float *box, float density);	// print created cnts in lammps data format
void write_CNTdat(const char *filename, int cntNum, float *cnt_arr);			// print created cnts in text format
void write_Meshvtk(int size[3], float *uni_mesh );		// print readed mesh
void write_Meshvtk(const char* filename, int size[3],float phsDm[3], float *uni_mesh );		// print readed mesh
void write_Meshvtk(const char* filename, int size[3], int *uni_mesh );		// print readed mesh
void write_RNDMeshvtk( int dim, float *rnd_mesh);
void write_vtkVectorField(const char* filename, int dim[3], float *vecfield);	// print vectro field
void write_Mathematica2Di(const char *filename, int size[2], unsigned int *data);
void write_Mathematica2Df(const char *filename, int size[2], float *data);
void write_Mathematica2D(const char *filename, int size[2], double *data);
void write_gnuplotMatrix(const char *filename, int size[2], float *data);
void write_gnuplot2Dfunc(const char *filename, int size[2], float step[2], float *data);
void write_sparse(const char *filename, int size, int *rows,int *cols, float* data);
void write_sparse2(const char *filename, int size, unsigned int *rows, unsigned int *cols, float* data1, float* data2);

int	writeVTKimage(int ext[2], int* data);

