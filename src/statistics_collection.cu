//====================================================================================================================
//										 <<< Kernels to collect statistical data: >>>
//====================================================================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "simPrms.h"
#include "IO.h"
#include "array_kern.h"

// type of data to analyse:
#define STAT_NONE	0		// do nothing (for test purposes)
#define	STAT_CENT	1		// compute center-center distance
#define STAT_ENDP	2		// compute end-end distance
#define STAT_ANGL	3		// compute average orientation angle and angle dispersion
#define STAT_DIST	4		// compute minimum distances
#define STAT_NCON	5		// compute number of connections (within maxDist)

//CUDA constants:
__constant__ int	numBins;					// number of bins to bild histogrtams
__constant__ float	binInterval;				// width of bin interval
__constant__ int	statGridExt[3];				// extents of grid to collect statistical data;
__constant__ int	grdExt_stat[3];				// extents of domain cells' grid
__constant__ float	phsDim_stat[3];				// physical dimensions of sample
__constant__ float	statGridStep[3];			// length of statGrid cell side;
__constant__ float	grdStep[3];					// length of domain cells' grid side;
__constant__ float	maxContactDist;				// maximum distance for tunneling connection;
__constant__ int	glbPtchSz_stat;				//  = numCNT
__constant__ float	float_precision;			// precision for float operations
__constant__ float3	left_elec_R,right_elec_R;	// position of electrode planes

#include "statKernels.h"

// cpu-part functions:
int writeResults(char*	fname,int arrSize,int nbins,float *averages,float *dispertions,unsigned int *counts,unsigned int *records);
//int collectResults(char*	fname,int arrSize,int nbins,float *averages,float *dispertions,unsigned int *counts,unsigned int *records);
// auxilary functions:
float sphericBelt_area(float theta0, float theta1);

//--------------------------------------------------------------------------------------------------------------------

int collect_vdw(	float range_lim,
					simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];
	
	printf("Collecting exculed Area: \n");

	// set constants:
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block(BLOCK);
	dim3	grid(statGridSize);
	dim3	grid_aux( (unsigned int) ceil(grid.x*1./block.x) );
	int		shrMem = (1)*devProp.warpSize*sizeof(unsigned int);

	// records data:
	float	*d_area = cuda_init<float>(0.0f,statGridSize);
	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);
		
		cudaEvent_t start,stop;
		float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = 0; int zh = rangeOrd;
		int yl = 0; int yh = rangeOrd;
		int xl = 0; int xh = rangeOrd;
		char3 stride = make_char3(0,0,0);
		
		for(stride.z=zl;stride.z<=zh;stride.z++) {
			for(stride.y=yl;stride.y<=yh;stride.y++) {
				for(stride.x=xl;stride.x<=xh;stride.x++) {
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_VdW_naive<<<grid,block,shrMem>>>(	range_lim,
																		0,
																		stride,
																		Simul.Domains.d_dmnAddr,
																		Simul.Domains.d_dmnOcc,
																		Simul.d_result,
																		d_area);
				}
				xl = -xh;                 
			}
			yl = -yh;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;

	shrMem = block.x*sizeof(double);
	double	*h_accArea = (double*) malloc (grid_aux.x*1*sizeof(double));
	double	*d_accArea; cudaMalloc(&d_accArea,grid_aux.x*1*sizeof(double));

	for(int j=0;j<2;j++)
		reduceStatData_Dbl<<<grid_aux,block,shrMem>>>( j, 1, d_area, d_accArea, statGridSize);
	cuErr = cudaMemcpy(h_accArea,d_accArea,grid_aux.x*1*sizeof(double),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));
	cudaFree(d_accArea);
	cudaFree(d_area);

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux.x;i++)	{
		for(int j=0;j<1;j++)
			h_accArea[0*1+j] += h_accArea[i*1+j];
	}

	printf("Calculated van der Waals energy per CNT: %f \n",h_accArea[0]/Simul.numCNT );
	free(h_accArea);
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

}

//--------------------------------------------------------------------------------------------------------------------

int collect_exclSurfArea(	float range_lim,
							simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];
	
	printf("Collecting exculed Area with the interaction distance %f: \n",range_lim);

	// set constants:
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block(BLOCK);
	dim3	grid(statGridSize);
	dim3	grid_aux((unsigned int) ceil(grid.x*1./block.x));
	int		shrMem = (2)*devProp.warpSize*sizeof(unsigned int);

	// records data:
	float	*d_area = cuda_init<float>(0.0f,statGridSize*2);
	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);
		
		cudaEvent_t start,stop;
		float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = 0; int zh = rangeOrd;
		int yl = 0; int yh = rangeOrd;
		int xl = 0; int xh = rangeOrd;
		char3 stride = make_char3(0,0,0);
		
		for(stride.z=zl;stride.z<=zh;stride.z++) {
			for(stride.y=yl;stride.y<=yh;stride.y++) {
				for(stride.x=xl;stride.x<=xh;stride.x++) {
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_exclSurf<<<grid,block,shrMem>>>(	range_lim,
																	0,
																	stride,
																	Simul.Domains.d_dmnAddr,
																	Simul.Domains.d_dmnOcc,
																	Simul.d_result,
																	d_area);
				}
				xl = -xh;                 
			}
			yl = -yh;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;

	shrMem = block.x*sizeof(double);
	double	*h_accArea = (double*) malloc (grid_aux.x*2*sizeof(double));
	double	*d_accArea; cudaMalloc(&d_accArea,grid_aux.x*2*sizeof(double));

	for(int j=0;j<2;j++)
		reduceStatData_Dbl<<<grid_aux,block,shrMem>>>( j, 2, d_area, d_accArea, statGridSize);
	cuErr = cudaMemcpy(h_accArea,d_accArea,grid_aux.x*2*sizeof(double),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux.x;i++)	{
		for(int j=0;j<2;j++)
			h_accArea[0*2+j] += h_accArea[i*2+j];
	}

	printf("Accumulated excluded surface: %f / %f = %f %% \n",h_accArea[1],h_accArea[0],h_accArea[1]/h_accArea[0]*100 );
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 
int collect_exclSurfArea2(	float range_lim,
							simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];
	
	printf("Collecting exculed Area with the interaction distance %f: \n",range_lim);

	// set constants:
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block(BLOCK);
	dim3	grid(statGridSize);
	dim3	grid_aux((unsigned int) ceil(grid.x*1./block.x));
	int		shrMem = (3)*devProp.warpSize*sizeof(unsigned int);

	// records data:
	float	*d_area = cuda_init<float>(0.0f,statGridSize*3);
	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);
		
		cudaEvent_t start,stop;
		float time;

		char3 stride = make_char3(0,0,0);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = 0; int zh = rangeOrd;
		int yl = 0; int yh = rangeOrd;
		int xl = 0; int xh = rangeOrd;
		
		for(stride.z=zl;stride.z<=zh;stride.z++) {
			for(stride.y=yl;stride.y<=yh;stride.y++) {
				for(stride.x=xl;stride.x<=xh;stride.x++) {
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_exclSurf2<<<grid,block,shrMem>>>(	range_lim,
																	0,
																	stride,
																	Simul.Domains.d_dmnAddr,
																	Simul.Domains.d_dmnOcc,
																	Simul.d_result,
																	d_area);
				}
				xl = -xh;                 
			}
			yl = -yh;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;

	shrMem = block.x*sizeof(double);
	double	*h_accArea = (double*) malloc (grid_aux.x*3*sizeof(double));
	double	*d_accArea; cudaMalloc(&d_accArea,grid_aux.x*3*sizeof(double));

	for(int j=0;j<3;j++)
		reduceStatData_Dbl<<<grid_aux,block,shrMem>>>( j, 3, d_area, d_accArea, statGridSize);
	cuErr = cudaMemcpy(h_accArea,d_accArea,grid_aux.x*3*sizeof(double),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux.x;i++)	{
		for(int j=0;j<3;j++)
			h_accArea[0*3+j] += h_accArea[i*3+j];
	}

	printf("Accumulated excluded surface: \n" );
	printf("Parallel: %f / %f = %f %% \n",h_accArea[1],h_accArea[0],h_accArea[1]/h_accArea[0]*100 );
	printf("Skew    : %f / %f = %f %% \n",h_accArea[2],h_accArea[0],h_accArea[2]/h_accArea[0]*100 );
	printf("Total   : %f / %f = %f %% \n",h_accArea[2]+h_accArea[1],h_accArea[0],(h_accArea[2]+h_accArea[1])/h_accArea[0]*100 );
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 

int collect_numContacts(	float range_lim,
							simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];
	
	printf("Collecting number of contacts with the interaction distance %f: \n",range_lim);

	// set constants:
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block;	block.x = BLOCK; block.y = 1; block.z = 1;
	dim3	grid;	grid.x = statGridSize; grid.y = 1; grid.z = 1;
	int		shrMem = (2)*devProp.warpSize*sizeof(unsigned int);

	dim3	grid_aux; grid_aux.x = (unsigned int) ceil(((float)grid.x)/block.x); grid_aux.y = 1; grid_aux.z = 1;

	// records data:
	float	*d_area = cuda_init<float>(0.0f,statGridSize*2);
	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);
		
		cudaEvent_t start,stop;
		float time;

		char3 stride = make_char3(0,0,0);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = 0; int zh = rangeOrd;
		int yl = 0; int yh = rangeOrd;
		int xl = 0; int xh = rangeOrd;
		
		for(stride.z=zl;stride.z<=zh;stride.z++) {
			for(stride.y=yl;stride.y<=yh;stride.y++) {
				for(stride.x=xl;stride.x<=xh;stride.x++) {
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_Contacts<<<grid,block,shrMem>>>(	range_lim,
																	0,
																	stride,
																	Simul.Domains.d_dmnAddr,
																	Simul.Domains.d_dmnOcc,
																	Simul.d_result,
																	d_area);
				}
				xl = -xh;                 
			}
			yl = -yh;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;

	shrMem = block.x*sizeof(double);
	double	*h_accCont = (double*) malloc (grid_aux.x*2*sizeof(double));
	double	*d_accCont; cudaMalloc(&d_accCont,grid_aux.x*2*sizeof(double));

	for(int j=0;j<2;j++)
		reduceStatData_Dbl<<<grid_aux,block,shrMem>>>( j, 2, d_area, d_accCont, statGridSize);
	cuErr = cudaMemcpy(h_accCont,d_accCont,grid_aux.x*2*sizeof(double),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux.x;i++)	{
		for(int j=0;j<2;j++)
			h_accCont[0*2+j] += h_accCont[i*2+j];
	}

	printf("Average number of contacts per inclusion: %f / %i = %f \n",2*h_accCont[1],Simul.numCNT,2*h_accCont[1]/Simul.numCNT );
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 
//--------------------------------------------------------------------------------------------------------------------
int collect_CntctPerInc(	float range_lim,
							simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];
	
	printf("Collecting number of contacts with the interaction distance %f: \n",range_lim);

	// set constants:
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block;	block.x = devProp.maxThreadsPerBlock; block.y = 1; block.z = 1;
	dim3	grid;	grid.x = statGridSize; grid.y = 1; grid.z = 1;
	
	// records data:
	unsigned int	statData_size = (unsigned int) ceil((float)Simul.Domains.ttlCNT/block.x)*block.x;
	unsigned int	*d_cntcts = cuda_init<unsigned int>(0,statData_size);

		cudaEvent_t start,stop;
		float time;

		char3 stride = make_char3(0,0,0);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = -rangeOrd; int zh = rangeOrd;
		int yl = -rangeOrd; int yh = rangeOrd;
		int xl = -rangeOrd; int xh = rangeOrd;
		
		for(stride.z=zl;stride.z<=zh;stride.z++) 
			for(stride.y=yl;stride.y<=yh;stride.y++) 
				for(stride.x=xl;stride.x<=xh;stride.x++) 
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_CntctPerInc<<<grid,block>>>(	range_lim,
																0,
																stride,
																Simul.Domains.d_dmnAddr,
																Simul.Domains.d_dmnOcc,
																Simul.d_result,
																d_cntcts);
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	/* //To test only:
	unsigned int	*h_cntcts = (unsigned int*) malloc(statData_size*sizeof(unsigned int));
	cuErr = cudaMemcpy(h_cntcts ,d_cntcts ,statData_size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));
	for(unsigned int i=1;i<statData_size;i++)
			h_cntcts[0] += h_cntcts[i];
	printf(" Total number of contacts per inclusion: %i / %i = %f \n",h_cntcts[0],Simul.numCNT,(float)h_cntcts[0]/Simul.numCNT );
	free(h_cntcts);
	//*/

	int	shrMem;
	// first reduction
	dim3	grid_aux1;
	shrMem = block.x*sizeof(unsigned int);
	grid_aux1.x = statData_size/block.x; grid_aux1.y = 1; grid_aux1.z = 1;
	unsigned int	*d_iniCont; cudaMalloc(&d_iniCont,statData_size*sizeof(unsigned int));
	reduceStatData_Int<<<grid_aux1,block,shrMem>>>( 0, 1, d_cntcts, d_iniCont, statData_size);
	
	// second reduction
	dim3	grid_aux2;
	shrMem = block.x*sizeof(unsigned long long int); 
	grid_aux2.x = ceil((float)grid_aux1.x/block.x);  grid_aux2.y = 1; grid_aux2.z = 1;
	unsigned long long int	*h_accCont = (unsigned long long int*) malloc (grid_aux2.x*sizeof(unsigned long long int));
	unsigned long long int	*d_accCont; cudaMalloc(&d_accCont,grid_aux2.x*sizeof(unsigned long long int));

	reduceStatData_Long<<<grid_aux2,block,shrMem>>>( 0, 1, d_iniCont, d_accCont, grid_aux1.x);
	cuErr = cudaMemcpy(h_accCont,d_accCont,grid_aux2.x*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux2.x;i++)
			h_accCont[0] += h_accCont[i];
	printf(" Total number of contacts per inclusion: %llu / %i = %f \n",h_accCont[0],Simul.numCNT,(float)h_accCont[0]/Simul.numCNT );
	free(h_accCont); cudaFree(d_accCont);

	// Find Max:
	// first reduction
	shrMem = block.x*sizeof(unsigned int);
	maxStatData_Int<<<grid_aux1,block,shrMem>>>( 0, 1, d_cntcts, d_iniCont, statData_size);

	// second reduction
	unsigned int	*h_maxCont = (unsigned int*) malloc (grid_aux2.x*sizeof(unsigned int));
	unsigned int	*d_maxCont; cudaMalloc(&d_maxCont,grid_aux2.x*sizeof(unsigned int));

	maxStatData_Int<<<grid_aux2,block,shrMem>>>( 0, 1, d_iniCont, d_maxCont, grid_aux1.x);
	cuErr = cudaMemcpy(h_maxCont,d_maxCont,grid_aux2.x*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux2.x;i++)	
		h_maxCont[0] = ( h_maxCont[0] >= h_maxCont[i] ? h_maxCont[0] : h_maxCont[i] ) ;
	printf("Maximum number of contacts per inclusion: %i \n",h_maxCont[0]);
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return h_maxCont[0];
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 

//--------------------------------------------------------------------------------------------------------------------
struct	nematic_func {
	const float3 ort;
	nematic_func(float3 _ort) : ort(_ort) {}

	__host__ __device__	float operator()(const thrust::tuple<float,float,float>& P) {
		float res = 0.0;

		float X = thrust::get<0>(P);
		float Y = thrust::get<1>(P);
		float Z = thrust::get<2>(P);
		float lim = -1+1.0e-6;
		if ((X>lim)&&(Y>lim)&&(Z>lim)) {
			// inclusion exists:
			float cosT = ort.x*X+ort.y*Y+ort.z*Z;
			res = (3*cosT*cosT-1.0)/2.0;
		}
		return res;
	}
};

typedef thrust::device_vector<float>::iterator	fIter;
typedef thrust::tuple<fIter,fIter,fIter>		fIter3;
typedef thrust::zip_iterator<fIter3>			f3ZipIter;

float get_nematic_order(	float3 	vec,
							simPrms	*Simul	) {
	// calculates nematic order parameter <S> = <(3*cos(theta)^2-1)/2>

	thrust::device_ptr<float> inc_crd(Simul->d_result);
	thrust::device_ptr<float> inc_vec_X(inc_crd + 3*Simul->Domains.ttlCNT);
	thrust::device_ptr<float> inc_vec_Y(inc_crd + 4*Simul->Domains.ttlCNT);
	thrust::device_ptr<float> inc_vec_Z(inc_crd + 5*Simul->Domains.ttlCNT);

	thrust::device_vector<float>	P_nem( Simul->Domains.ttlCNT, 0.0);
	f3ZipIter	inc_vec_beg = thrust::make_zip_iterator( thrust::make_tuple(	inc_vec_X,
																				inc_vec_Y,
																				inc_vec_Z));
	f3ZipIter	inc_vec_end = thrust::make_zip_iterator( thrust::make_tuple(	inc_vec_X + Simul->Domains.ttlCNT,
																				inc_vec_Y + Simul->Domains.ttlCNT,
																				inc_vec_Z + Simul->Domains.ttlCNT));

	thrust::transform(	inc_vec_beg, inc_vec_end, P_nem.begin(), nematic_func(vec));

	return thrust::reduce(P_nem.begin(), P_nem.end())/Simul->numCNT;

}


int collect_1D_statDistr_weightedFloat(	float range_lim,
										const char *fname,
										int	bins1D,
										simPrms	Simul)
{
	cudaError cuErr;
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];

	float dX		= (acosf(-1.0f)+Simul.TOL)/2/bins1D;
	
	printf("Collecting 1D statistics with the interaction distance %f: \n",range_lim);

	// set constants:
	cudaMemcpyToSymbol(numBins,&bins1D,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(binInterval,&dX,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_stat,Simul.physDim,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(float_precision,&Simul.TOL,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block;	block.x = BLOCK; block.y = 1; block.z = 1;
	dim3	grid;	grid.x = statGridSize; grid.y = 1; grid.z = 1;
	int		shrMem = (bins1D+1)*devProp.warpSize*sizeof(unsigned int);

	dim3	grid_aux; grid_aux.x = (unsigned int) ceil(((float)grid.x)/block.x); grid_aux.y = 1; grid_aux.z = 1;

	// records data:
	unsigned int	*d_counts = cuda_init<unsigned int>(0,statGridSize);
	float			*d_records = cuda_init<float>(0.0f,statGridSize*bins1D);
	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);
		
		cudaEvent_t start,stop;
		float time;

		char3 stride = make_char3(0,0,0);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int rangeOrd = (int) ceil((Simul.l+range_lim)/Simul.physDim[0]*Simul.Domains.ext[0]);
		printf("Scan up to order: %i Simul.l = %f \n",rangeOrd,Simul.l);

		int zl = 0; int zh = rangeOrd;
		int yl = 0; int yh = rangeOrd;
		int xl = 0; int xh = rangeOrd;
		
		for(stride.z=zl;stride.z<=zh;stride.z++) {
			for(stride.y=yl;stride.y<=yh;stride.y++) {
				for(stride.x=xl;stride.x<=xh;stride.x++) {
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						statKernel_MUT_ANGLE_weightedFloat<<<grid,block,shrMem>>>(	range_lim,
																					0,
																					stride,
																					Simul.Domains.d_dmnAddr,
																					Simul.Domains.d_dmnOcc,
																					Simul.d_result,
																					d_records,
																					d_counts);
				}
				xl = -xh;
			}
			yl = -yh;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Statistical Kernel - Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));
	
	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;
	//grid.x = ceil(statGridSize/1024.0f); grid.y = 1; grid.z = 1;

	shrMem = block.x*sizeof(unsigned long long int);
	unsigned long long int	*h_accCnts = (unsigned long long int*) malloc (grid_aux.x*sizeof(unsigned long long int));
	unsigned long long int	*d_accCnts = cuda_init<unsigned long long int>(0,grid_aux.x);
	reduceStatData_Long<<<grid_aux,block,shrMem>>>( 0, 1, d_counts, d_accCnts, statGridSize);
	
	cuErr = cudaMemcpy(h_accCnts,d_accCnts,grid_aux.x*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess)
		printf("ERROR WHILE REDUCING COUNTS: %s !!!\n",cudaGetErrorString(cuErr));
	else
		for(unsigned int i=1;i<grid_aux.x;i++)	h_accCnts[0] += h_accCnts[i];

	shrMem = block.x*sizeof(double);
	double	*h_accRecs = (double*) malloc (grid_aux.x*bins1D*sizeof(double));
	double	*d_accRecs; cudaMalloc(&d_accRecs,grid_aux.x*bins1D*sizeof(double));

	for(int j=0;j<bins1D;j++)
		reduceStatData_Dbl<<<grid_aux,block,shrMem>>>( j, bins1D, d_records, d_accRecs, statGridSize);
	cuErr = cudaMemcpy(h_accRecs,d_accRecs,grid_aux.x*bins1D*sizeof(double),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING RECORDS: %s !!!\n",cudaGetErrorString(cuErr));

	// accumulate and normalize bin data:
	for(unsigned int i=1;i<grid_aux.x;i++)	{
		for(int j=0;j<bins1D;j++)
			h_accRecs[0*bins1D+j] += h_accRecs[i*bins1D+j];
	}
	double sum = 0.0f;
	for(int j=0;j<bins1D;j++)
			sum += h_accRecs[j];

	printf("accumulated counts %llu-%f\n",h_accCnts[0],sum );
	//write_dat(fname,bins1D,h_accRecs);
	write_mat(fname,bins1D,h_accRecs);
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 
//--------------------------------------------------------------------------------------------------------------------
// function to collect rho(phi,theta):

int collect_PhiTheta_distr(const char *fname,
							int	numBinsX,
							int numBinsY,
							simPrms	Simul)
{
	int statGridExt[3];
		statGridExt[0] = Simul.Domains.ext[0];
		statGridExt[1] = Simul.Domains.ext[1];
		statGridExt[2] = Simul.Domains.ext[2];
	int statGridSize = statGridExt[0]*statGridExt[1]*statGridExt[2];

	float h_dPhi		= acosf(-1.0f)/numBinsX;
	float h_dTheta	= acosf(-1.0f)/numBinsY;
	int	h_numBins2D[2];
		h_numBins2D[0] = numBinsX;
		h_numBins2D[1] = numBinsY;
	int	bins2D_total = numBinsX*numBinsY;
	
	printf("Collecting rho(phi,theta) statistics \n");
	printf("dPhi = %f dTheta = %f \n",h_dPhi,h_dTheta);

	// set constants:
	cudaMemcpyToSymbol(numBins2D,&h_numBins2D,2*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dPhi,&h_dPhi,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dTheta,&h_dTheta,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);

	// records data:
	unsigned int *d_records; cudaMalloc(&d_records,statGridSize*bins2D_total*sizeof(unsigned int));
	unsigned int *h_records = (unsigned int*) malloc(statGridSize*bins2D_total*sizeof(unsigned int));
	// kernell call:
	dim3	block;	block.x = BLOCK; block.y = 1; block.z = 1;
	dim3	grid;	grid.x = statGridSize; grid.y = 1; grid.z = 1;
	int		shrMem = bins2D_total*sizeof(unsigned int);

	printf("block = %i grid = %i shrMem = %i \n",block.x,grid.x,shrMem);

		cudaEvent_t start,stop;
		cudaError cuErr;
		float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

					statKernel_ANGL<<<grid,block,shrMem>>>(	0,
															Simul.Domains.d_dmnAddr,
															Simul.Domains.d_dmnOcc,
															Simul.d_result,
															d_records);
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Statistical Kernel - Total Time: %f ms \n",time );
		cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE RUNNING KERNELL: %s !!!\n",cudaGetErrorString(cuErr));

	
	// Collect records:
	block.x = 1024; block.y = 1; block.z = 1;
	grid.x = (unsigned int) ceil(statGridSize/1024.0f); grid.y = 1; grid.z = 1;
	shrMem = block.x*sizeof(unsigned int);
	unsigned int	*h_accRecs = (unsigned int*) malloc (grid.x*bins2D_total*sizeof(unsigned int));
	unsigned int	*d_accRecs; cudaMalloc(&d_accRecs,grid.x*bins2D_total*sizeof(unsigned int));

	for(int j=0;j<bins2D_total;j++)	reduceStatData_Int<<<grid,block,shrMem>>>( j,bins2D_total, d_records, d_accRecs, statGridSize);


	cudaMemcpy(h_accRecs,d_accRecs,grid.x*bins2D_total*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cuErr = cudaGetLastError();
	if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING: %s !!!\n",cudaGetErrorString(cuErr));

	for(unsigned int i=1;i<grid.x;i++)
		for(int j=0;j<bins2D_total;j++)
			h_accRecs[0*bins2D_total+j] += h_accRecs[i*bins2D_total+j];

	write_Mathematica2Di(fname, h_numBins2D, h_accRecs);
	//write_gnuplotMatrix("rho_gnu.dat", h_numBins2D, rho);
	//float	steps[2];
	//		steps[0] = h_dPhi;
	//		steps[1] = h_dTheta;
	//write_gnuplot2Dfunc("rho_gnu2D.dat",h_numBins2D, steps, rho);
	
	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cuErr));
		return -1;
	}

} 
//--------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------
// function to call statistics kernel:
int collectStats(int	type,
				 int	h_numBins,
				 float	sampleRange,
				 char*	fname,
				 simPrms	Simul)
{
	int h_statGridExt[3];
	float statScale_crrtd[3];
		h_statGridExt[0] = (int) floor(Simul.physDim[0]/sampleRange);	 statScale_crrtd[0] = Simul.physDim[0]/h_statGridExt[0];
		h_statGridExt[1] = (int) floor(Simul.physDim[1]/sampleRange);	 statScale_crrtd[1] = Simul.physDim[1]/h_statGridExt[1];
		h_statGridExt[2] = (int) floor(Simul.physDim[2]/sampleRange);	 statScale_crrtd[2] = Simul.physDim[2]/h_statGridExt[2];
	int statGridSize = h_statGridExt[0]*h_statGridExt[1]*h_statGridExt[2];
	float interval = sampleRange/h_numBins;
	printf("sample range = %f h_numBins = %i interval = %f \n",sampleRange,h_numBins,interval);
	// records data:
	unsigned int *h_records = (unsigned int*) malloc(statGridSize*h_numBins*sizeof(unsigned int));
	unsigned int *d_records; cudaMalloc(&d_records,statGridSize*h_numBins*sizeof(unsigned int));
	for(int i=0;i<statGridSize*h_numBins;i++) h_records[i] = 0;
	cudaMemcpy(d_records,h_records,statGridSize*h_numBins*sizeof(int),cudaMemcpyHostToDevice);

	// avg data:
	float *h_avgVal = (float*) malloc(statGridSize*sizeof(float));
	float *d_avgVal; cudaMalloc(&d_avgVal,statGridSize*sizeof(float));
	for(int i=0;i<statGridSize;i++) h_avgVal[i] = 0;
	cudaMemcpy(d_avgVal,h_avgVal,statGridSize*sizeof(float),cudaMemcpyHostToDevice);

	// disp data:
	float *h_dispVal = (float*) malloc(statGridSize*sizeof(float));
	float *d_dispVal; cudaMalloc(&d_dispVal,statGridSize*sizeof(float));
	for(int i=0;i<statGridSize;i++) h_dispVal[i] = 0;
	cudaMemcpy(d_dispVal,h_dispVal,statGridSize*sizeof(float),cudaMemcpyHostToDevice);

	// counts data:
	unsigned int *h_counts = (unsigned int*) malloc(statGridSize*sizeof(unsigned int));
	unsigned int *d_counts; cudaMalloc(&d_counts,statGridSize*sizeof(unsigned int));
	for(int i=0;i<statGridSize;i++) h_counts[i] = 0;
	cudaMemcpy(d_counts,h_counts,statGridSize*sizeof(unsigned int),cudaMemcpyHostToDevice);

	// set constants:
	cudaMemcpyToSymbol(numBins,&h_numBins,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(binInterval,&interval,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(statGridExt,h_statGridExt,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt_stat,Simul.Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(statGridStep,statScale_crrtd,3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdStep,&Simul.Domains.edge,3*sizeof(float),0,cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(maxContactDist,&sampleRange,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_stat,&Simul.Domains.ttlCNT,sizeof(float),0,cudaMemcpyHostToDevice);


	// kernell call:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	dim3	block;	block.x = BLOCK; block.y = 1; block.z = 1;
	dim3	grid;	grid.x = statGridSize; grid.y = 1; grid.z = 1;
	int		shrMem = devProp.warpSize*(h_numBins*sizeof(int)+2*sizeof(float)+sizeof(unsigned int));
	printf("shared memory allocated: %i \n",shrMem);

		cudaEvent_t start,stop;
		float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int		dmnCellsPatch[3];
		for(int i=0;i<3;i++)
			dmnCellsPatch[i] = ((int) ceil(Simul.Domains.ext[i]/h_statGridExt[i] + 2*sampleRange/Simul.Domains.edge[i]))%(Simul.Domains.ext[i]);
		printf( "dmnCellsPatch = %i x %i x %i \n",dmnCellsPatch[0],dmnCellsPatch[1],dmnCellsPatch[2] );
		int3	displ = make_int3(1,1,1);
		for(displ.x=0;displ.x<dmnCellsPatch[0];displ.x++)
			for(displ.y=0;displ.y<dmnCellsPatch[1];displ.y++)
				for(displ.z=0;displ.z<dmnCellsPatch[2];displ.z++)
					statKernel_CENT<<<grid,block,shrMem>>>(	0,
															displ,
															Simul.Domains.d_dmnAddr,
															Simul.Domains.d_dmnOcc,
															Simul.d_result,
															d_records,
															d_avgVal,
															d_dispVal,
															d_counts);

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Statistical Kernel - Total Time: %f ms \n",time );
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cudaGetLastError()));

	// read back results:
	cudaMemcpy(h_records,d_records,statGridSize*h_numBins*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_avgVal,d_avgVal,statGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dispVal,d_dispVal,statGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_counts,d_counts,statGridSize*sizeof(unsigned int),cudaMemcpyDeviceToHost);

	long long int sum = 0;
	for(int i=0;i<statGridSize;i++) sum += h_counts[i];
	printf( "Statistical Kernel - Total counts: %lld  \n",sum );
	double sumD = 0.0;
	double sumB = 0.0;
	for(int i=0;i<statGridSize;i++) {
		sumB +=	h_avgVal[i];
		sumD += h_avgVal[i]/h_counts[i];
	}
	printf( "Statistical Kernel - Total averages: %f and %f \n",sumD/statGridSize,sumB/sum );

	//writeResults(fname,statGridSize,h_numBins,h_avgVal,h_dispVal,h_counts,h_records);

	//collect counts:
	block.x = 1024; block.y = 1; block.z = 1;
	grid.x = (unsigned int) ceil(statGridSize/1024.0f); grid.y = 1; grid.z = 1;
	shrMem = block.x*sizeof(unsigned int);
	int cllctGridSize = (int) ceil(statGridSize/1024.0f);
	unsigned int	*h_accCnts = (unsigned int*) malloc (cllctGridSize*sizeof(unsigned int));
	unsigned int	*d_accCnts; cudaMalloc(&d_accCnts,cllctGridSize*sizeof(unsigned int));
	reduceStatData_Int<<<grid,block,shrMem>>>( 0,1, d_counts, d_accCnts, statGridSize);
	cudaMemcpy(h_accCnts,d_accCnts,cllctGridSize*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	unsigned long long int accumulated_cnts = 0;
	for(int i=0;i<cllctGridSize;i++)	accumulated_cnts += h_accCnts[i];
	printf( "Statistical Kernel - Collected counts: %lld  \n",accumulated_cnts );

	// Collect averages:
	shrMem = block.x*sizeof(double);
	double	*h_accAvgs = (double*) malloc (cllctGridSize*sizeof(double));
	double	*d_accAvgs; cudaMalloc(&d_accAvgs,cllctGridSize*sizeof(double));
	reduceStatData_Float<<<grid,block,shrMem>>>( d_avgVal,d_counts, d_accAvgs, statGridSize);
	cudaMemcpy(h_accAvgs,d_accAvgs,cllctGridSize*sizeof(double),cudaMemcpyDeviceToHost);
	double accumulated_avgs = 0.0;
	for(int i=0;i<cllctGridSize;i++)	accumulated_avgs += h_accAvgs[i];
	printf( "Statistical Kernel - Collected averages: %f  \n",accumulated_avgs/statGridSize );

	// Collect records:
	shrMem = block.x*sizeof(unsigned int);
	unsigned int	*h_accRecs = (unsigned int*) malloc (cllctGridSize*h_numBins*sizeof(unsigned int));
	unsigned int	*d_accRecs; cudaMalloc(&d_accRecs,cllctGridSize*h_numBins*sizeof(unsigned int));
	for(int j=0;j<h_numBins;j++)
		reduceStatData_Int<<<grid,block,shrMem>>>( j,h_numBins, d_records, d_accRecs, statGridSize);
	cudaMemcpy(h_accRecs,d_accRecs,cllctGridSize*h_numBins*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	unsigned long long int accumulated_Recs = 0;
	for(int i=0;i<cllctGridSize;i++)
		for(int j=0;j<h_numBins;j++)
			accumulated_Recs += h_accRecs[i*h_numBins+j];
	printf( "Statistical Kernel - Collected counts in bins: %lld  \n",accumulated_Recs );

	float	*accRecs_normalized = (float*) malloc (h_numBins*sizeof(float));
	float dR = sampleRange/h_numBins;
	float RI = 0;
	float	volI = 0;
	for(int j=0;j<h_numBins;j++) {
			RI +=dR;
			volI = 4.f/3.f*Pi*RI*RI*RI-volI;
			accRecs_normalized[j] = ((float)h_accRecs[j])/accumulated_Recs/volI;
			
			//printf(" Normilised bin[%i] = %f \n",j,accRecs_normalized[j]);
	}
	int	distr_size[2];
		distr_size[0] = 1;
		distr_size[1] = h_numBins;
	write_gnuplotMatrix(fname, distr_size, accRecs_normalized);

	
	cudaError	cuErr = cudaGetLastError();
	if (cuErr==cudaSuccess) return 0;
	else {
		printf("ERROR WHILE COLLECTING STATISTICS: %s !!!\n",cudaGetErrorString(cudaGetLastError()));
		return -1;
	}

}
//--------------------------------------------------------------------------------------------------------------------

int writeResults(char*	fname,int arrSize,int nbins,float *averages,float *dispertions,unsigned int *counts,unsigned int *records){

	//For testing purposes:
	int blockId = 2000;
	printf("!!! FOR TESTING PURPOSES !!!\n");
	printf("printing out %i block statistics\n",blockId);
	printf("COUNT: %i \n",counts[blockId]);
	printf("AVERAGE: Accumulated value: %f - averaged: %f \n",averages[blockId],averages[blockId]/counts[blockId]);
	printf("DISPERSION: Accumulated value: %f - averaged: %f \n",dispertions[blockId],dispertions[blockId]/counts[blockId]);
	printf("Histogram:\n");
	int sum = 0;
	for(int i=0;i<nbins;i++)	{
		printf("bin[%2i] = %f\n",i,records[blockId*nbins+i]/(i+.5)/(i+.5));
		sum += records[blockId*nbins+i];
	}
	printf("total:  %i\n",sum);
	return 0;
	
}
// AUXILARY:==============================================================================================================
float sphericBelt_area(float theta0, float theta1) {
	return 2*acosf(-1.0f)*(cosf(theta0)-cosf(theta1));
}


