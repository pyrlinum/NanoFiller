#include "statKernel_CENT.h"
#include "statKernel_ANGL.h"
#include "statKernel_MUT.h"
#include "statKernel_VDW.h"

#ifndef BLOCK
#define BLOCK	1024
#endif

__global__ void reduceStatData_Int( int displace, int stride, unsigned int	*d_Stat,unsigned int *d_accStat, unsigned int arraySize) 
{
	extern __shared__ unsigned int shr_arr_int[];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_int[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_int[threadIdx.x] = d_Stat[displace+tid*stride];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s) shr_arr_int[threadIdx.x] += shr_arr_int[threadIdx.x+s];
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x*stride+displace] = shr_arr_int[0];
}

__global__ void reduceStatData_Float( float	*d_StatFlt, unsigned int	*d_StatCnts, double	*d_accStat, int arraySize) 
{
	extern __shared__ double shr_arr_float[];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_float[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_float[threadIdx.x] = d_StatFlt[tid]/d_StatCnts[tid];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s) shr_arr_float[threadIdx.x] += shr_arr_float[threadIdx.x+s];
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x] = shr_arr_float[0];
}
__global__ void reduceStatData_Long( int displace, int stride, unsigned int	*d_Stat,unsigned long long int *d_accStat, unsigned int arraySize) 
{
	extern __shared__ unsigned long long int shr_arr_lng[];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_lng[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_lng[threadIdx.x] = d_Stat[displace+tid*stride];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s) shr_arr_lng[threadIdx.x] += shr_arr_lng[threadIdx.x+s];
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x*stride+displace] = shr_arr_lng[0];
}

__global__ void reduceStatData_Dbl( int displace, int stride, float	*d_Stat, double *d_accStat, unsigned int arraySize) 
{
	extern __shared__ double shr_arr_dbl[];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_dbl[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_dbl[threadIdx.x] = d_Stat[displace+tid*stride];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s) shr_arr_dbl[threadIdx.x] += shr_arr_dbl[threadIdx.x+s];
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x*stride+displace] = shr_arr_dbl[0];
}

// MAX:
__global__ void maxStatData_Int( int displace, int stride, unsigned int	*d_Stat, unsigned int *d_accStat, unsigned int arraySize) 
{
	__shared__ int shr_arr_int[BLOCK];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_int[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_int[threadIdx.x] = d_Stat[displace+tid*stride];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s)
			shr_arr_int[threadIdx.x] = ( shr_arr_int[threadIdx.x] >= shr_arr_int[threadIdx.x+s] ? shr_arr_int[threadIdx.x] : shr_arr_int[threadIdx.x+s]);
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x*stride+displace] = shr_arr_int[0];
}
// MIN:
__global__ void minStatData_Int( int displace, int stride, unsigned int	*d_Stat, unsigned int *d_accStat, unsigned int arraySize) 
{
	__shared__ int shr_arr_int[BLOCK];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// initialize:
	shr_arr_int[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// read in data:
	if (tid < arraySize)	shr_arr_int[threadIdx.x] = d_Stat[displace+tid*stride];
	__threadfence_block();
	__syncthreads();
	// reduce data:
	for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
		if (threadIdx.x<s)
			shr_arr_int[threadIdx.x] = ( shr_arr_int[threadIdx.x] <= shr_arr_int[threadIdx.x+s] ? shr_arr_int[threadIdx.x] : shr_arr_int[threadIdx.x+s]);
		__threadfence_block();
		__syncthreads();
	}
	
	// write results:
	if (threadIdx.x==0) d_accStat[blockIdx.x*stride+displace] = shr_arr_int[0];
}
