//====================================================================================================================
//										 <<< Mutual position based statistic kernel >>>
//====================================================================================================================
#pragma once
#include "CNT.h"
#include "intersection_dev.h"
#include "memoryIO.h"
#include "ellipse_fit.h"

#define A 15.2
#define B 24100
#define v 0.393
#define um2A 10000

__device__ float exclArea(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float exclArea_naive(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float exclArea_par(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float exclArea_skew(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float exclArea_skew1(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float exclArea_skew2(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon);
__device__ float curve2surface_limit(float d, float dlim, float R1, float R2, float C6, float C12, float thetaM );
//----------------------------------------------------------------------------------------------------------------------
// Mutual Angle collection: 
//----------------------------------------------------------------------------------------------------------------------
extern "C"
__global__ void statKernel_MUT_ANGLE_weightedFloat(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										float *d_statResult,	// array to store results
										unsigned  int *d_counts)				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];
//results:
extern __shared__ unsigned int dynShr[];
unsigned int	*shrDstrCnts	= (unsigned int*) &(dynShr[0]);
unsigned int	*shrRecord_MA	= (unsigned int*) &(dynShr[warpSize]);
//if(blockIdx.x ==0) {
	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {

	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	// displace neighbouring particles to apply periodic conditions
		// nullify shared arrays
		if (threadIdx.x < warpSize) {
			shrDstrCnts[threadIdx.x] = 0;
			for (int i=0;i<numBins;i++)
				shrRecord_MA[threadIdx.x+i*warpSize] = 0;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		int angleBin;
		int colomn = threadIdx.x%warpSize;
		int lim = 2<<20;
		int loI;
		// check distance:
		if (threadIdx.x < selfCNT) {
			if (neigID==selfID)
				loI = threadIdx.x+1;
			else			
				loI = 0;
				for(int i=loI;i<neigCNT;i++) {
					float3 Dtt;
					float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
					if ( (d == Dtt.x)&&( Dtt.x-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
						float3 vec2 = make_float3(	shrArr[i+3*blockDim.x],
													shrArr[i+4*blockDim.x],
													shrArr[i+5*blockDim.x]	);
						angleBin = ((int) floor( acosf(abs(cosVec2Vec(probe.c,vec2)))/binInterval ))%numBins;
						atomicInc(&shrRecord_MA[angleBin*warpSize+colomn],lim);
						atomicInc(&shrDstrCnts[colomn],lim);
					}
				}
		}
		
		__syncthreads();
		__threadfence_block();
		
		reductOverWarp(numBins+1,dynShr);

	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x == 0)
			d_counts[blockIdx.x] += shrDstrCnts[0];
		if (threadIdx.x < numBins)
			d_statResult[blockIdx.x*numBins+threadIdx.x] += shrRecord_MA[threadIdx.x*warpSize];

	}

}

//----------------------------------------------------------------------------------------------------------------------
//Excluded surface arrea collection: 
//----------------------------------------------------------------------------------------------------------------------
extern "C"
__global__ void statKernel_exclSurf(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										float *d_statResult)	// array to store results				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];
//results:
extern __shared__ float shrArea[];
float	*shrTotArea	= &(shrArea[0]);
float	*shrExcArea	= &(shrArea[warpSize]);


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {

	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	

		if (threadIdx.x < warpSize) {
				shrTotArea[threadIdx.x]	= 0;
				shrExcArea[threadIdx.x]	= 0;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		int colomn = threadIdx.x%warpSize;
		int loI;
		// check distance:
		if (threadIdx.x < selfCNT) {
			if (neigID==selfID)	{
				loI = threadIdx.x+1;
				atomicAdd(&shrTotArea[colomn],totInclArea(probe));
			} else	loI = 0;
			
			for(int i=loI;i<neigCNT;i++) {
				float3 Dtt;
				float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
				if ( (d == Dtt.x)&&( Dtt.x-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
					//float excA = exclArea(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					float excA = exclArea_naive(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					atomicAdd(&shrExcArea[colomn],excA );
				}
			}
		}
		__threadfence_block();
		__syncthreads();
		
		reductOverWarp(2,shrArea);

	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x == 0)	{
			d_statResult[2*selfID+0] += shrArea[0*warpSize];
			d_statResult[2*selfID+1] += shrArea[1*warpSize];
		}

	}

}
// separate calculation for parallel and skew inlusions:
extern "C"
__global__ void statKernel_exclSurf2(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										float *d_statResult)	// array to store results				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];
//results:
extern __shared__ float shrArea[];
float	*shrTotArea	= &(shrArea[0*warpSize]);
float	*shrParArea	= &(shrArea[1*warpSize]);
float	*shrSkewArea= &(shrArea[2*warpSize]);


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {

	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	

		if (threadIdx.x < warpSize) {
				shrTotArea[threadIdx.x]	= 0;
				shrParArea[threadIdx.x]	= 0;
				shrSkewArea[threadIdx.x]= 0;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		int colomn = threadIdx.x%warpSize;
		int loI;
		// check distance:
		if (threadIdx.x < selfCNT) {
			if (neigID==selfID)	{
				loI = threadIdx.x+1;
				atomicAdd(&shrTotArea[colomn],totInclArea(probe));
			} else	loI = 0;
			
			for(int i=loI;i<neigCNT;i++) {
				float3 Dtt;
				float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
				//if ( (d == Dtt.x)&&( Dtt.x-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
				if ( ( d-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
					//float excA = exclArea(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					float excA1 = exclArea_par(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					float excA2 = exclArea_skew(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					atomicAdd(&shrParArea[colomn],excA1 );
					atomicAdd(&shrSkewArea[colomn],excA2 );
				}
			}
		}
		__threadfence_block();
		__syncthreads();
		
		reductOverWarp(3,shrArea);

	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x == 0)	{
			d_statResult[3*selfID+0] += shrArea[0*warpSize];
			d_statResult[3*selfID+1] += shrArea[1*warpSize];
			d_statResult[3*selfID+2] += shrArea[2*warpSize];
		}

	}

}

extern "C"
__global__ void statKernel_Contacts(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										float *d_statResult)	// array to store results				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];
//results:
extern __shared__ float shrArea[];
float	*shrTotArea	= &(shrArea[0]);
float	*shrExcArea	= &(shrArea[warpSize]);


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {
	//if (selfID == 598 ){

	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	

		if (threadIdx.x < warpSize) {
				shrTotArea[threadIdx.x]	= 0;
				shrExcArea[threadIdx.x]	= 0;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		int colomn = threadIdx.x%warpSize;
		int loI;
		// check distance:
		if (threadIdx.x < selfCNT) {
			if (neigID==selfID)	{
				loI = threadIdx.x+1;
				atomicAdd(&shrTotArea[colomn],totInclArea(probe));
			} else	loI = 0;
			
			for(int i=loI;i<neigCNT;i++) {
				float3 Dtt;
				float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
				if ( (d == Dtt.x)&&( Dtt.x-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
					//float excA = exclArea(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					float excA = exclArea_naive(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					atomicAdd(&shrExcArea[colomn],1 );
					//atomicExch(&shrExcArea[0],excA );
				}
			}
		}
		__threadfence_block();
		__syncthreads();
		
		reductOverWarp(2,shrArea);

	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x == 0)	{
			d_statResult[2*selfID+0] += shrTotArea[0*warpSize];
			d_statResult[2*selfID+1] += shrTotArea[1*warpSize];
		}

	}

}

extern "C"
// Accumulate the number of contacts per inclusion:
__global__ void statKernel_CntctPerInc(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										unsigned int *d_statResult)	// array to store results				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {
	
	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();
		shrClean(shrArr);
		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	

	// MAIN CODE:----------------------------------------------------------------- 
		unsigned int cntct = 0; // used to accumulate contacts
		int	loI = 0;
		// check distance:
		if (threadIdx.x < selfCNT) {
			for(int i=loI;i<neigCNT;i++) {
				if (!((selfID==neigID)&&(threadIdx.x==i))) {
					if (cnt_intersec_shr(probe,shrArr,blockDim.x, i,float_precision,dist_limit)) cntct++;
				}
			}
		}
		__threadfence_block();
		__syncthreads();
	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x < selfCNT)	{
			d_statResult[dmnAddr_startSelf+threadIdx.x] += cntct;
		}
	}
}

//
//----------------------------------------------------------------------------------------------------------------------
//Van der Waals collection: 
//----------------------------------------------------------------------------------------------------------------------
extern "C"
__global__ void statKernel_VdW_naive(	float dist_limit,		// shortest distance limit
										int count,				// N of kernell lanch
										char3 stride,			// displacement in dmn grid
										int *d_dmnAddr,			// offset of particles' coordinates in d_partCoords
										short *d_dmnOcc,		// number of created cnts in each domain
										float *d_partCoords,	// array with input coordinates
										float *d_statResult)	// array to store results				
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int neigID;		// the global ID of  neighbouring cell
__shared__ int dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];
//results:
extern __shared__ float shrVdW[];


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if (selfID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2] ) {
	//if (selfID == 598 ){

	// BASIC SETUP:-----------------------------------------------------------------
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_partCoords,glbPtchSz_stat);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID(stride, selfID, grdExt_stat);
			char3 shift  = stride2dspl(stride,selfID, grdExt_stat); 
			dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
			neigCNT =  d_dmnOcc[neigID];
			neigDspl = make_float3(	shift.x*phsDim_stat[0],
									shift.y*phsDim_stat[1],
									shift.z*phsDim_stat[2]);
		}
		__threadfence_block();
		__syncthreads();
		GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_partCoords,glbPtchSz_stat);		
		tarnsCNT(neigCNT, neigDspl, shrArr);	

		if (threadIdx.x < warpSize) {
				shrVdW[threadIdx.x]	= 0;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		int colomn = threadIdx.x%warpSize;
		int loI;
		// check distance:
		if (threadIdx.x < selfCNT) {
			if (neigID==selfID)	{
				loI = threadIdx.x+1;
			} else	loI = 0;
			
			for(int i=loI;i<neigCNT;i++) {
				float3 Dtt;
				float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
				if ( (d == Dtt.x)&&( Dtt.x-probe.a-shrArr[i+blockDim.x*7]  < dist_limit ) )	{
					float excA = exclArea_naive(Dtt,probe,&shrArr[i],blockDim.x,dist_limit,float_precision);
					float theta = atanf(probe.a/(d-shrArr[i+blockDim.x*7]-dist_limit));
					float vdw = excA/theta/probe.a*curve2surface_limit( (Dtt.x-probe.a-shrArr[i+blockDim.x*7])*um2A, dist_limit*um2A, probe.a*um2A, shrArr[i+blockDim.x*7]*um2A, A, B, theta );
					atomicAdd(&shrVdW[colomn],vdw );
					//atomicExch(&shrExcArea[0],excA );
				}
			}
		}
		__threadfence_block();
		__syncthreads();
		
		reductOverWarp(1,shrVdW);

	// CLEAN UP AND WRITE RESULTS:------------------------------------------------------------------
		if (threadIdx.x == 0)	{
			d_statResult[2*selfID+0] += shrVdW[0*warpSize];
		}

	}

}


// auxilary:
__device__ float exclArea(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon) {

	float exclArea = 0;

	float R1 = probe.a+dist;
	float R2 = shrcnt[7*stride]+dist;
	float y2 = Dtt.x;

	float3 vec2 = make_float3(	shrcnt[3*stride],
								shrcnt[4*stride],
								shrcnt[5*stride]	);
	float gamma = acosf(cosVec2Vec(probe.c,vec2));
	if (abs(sinf(gamma))>epsilon) {
		float3 P[3];
		P[0] = make_float3(0,R1,sqrtf(R2*R2-powf(R1-y2,2))/sinf(gamma));

		P[1] = make_float3(sqrtf(R1*R1-powf(R2-y2,2)),y2-R2,0);
		P[1].z = P[1].x*cosf(gamma);

		P[2] = make_float3(0, ( y2-sqrtf( powf(y2*cosf(gamma),2) + powf(sinf(gamma),2)*(R2*R2-powf(R1*cosf(gamma),2)) ) )/powf(sinf(gamma),2),0);
		P[2].x = sqrtf(R1*R1-P[2].y*P[2].y);

		float scale = (1.0/P[2].x);

		//float3 U[3];
		P[0] = unwrapZ(P[0])*scale;
		P[1] = unwrapZ(P[1])*scale;
		P[2] = unwrapZ(P[2])*scale;

		float3 coeff = ellipse_fit(make_float2(P[0].x,P[0].z),make_float2(P[1].x,P[1].z),make_float2(P[2].x,P[2].z));
		//float3 coeff = ellipse_fit(make_double2(U[0].x,U[0].z),make_double2(U[1].x,U[1].z),make_double2(U[2].x,U[2].z));
		exclArea = elli_area(coeff.x,coeff.y,coeff.z);//*powf(P[2].x,2);

		//exclArea =  P[0].y*P[0].y+P[0].x*P[0].x;

		/*

		double A0[3][3] = {	{U[0].x*U[0].x,U[0].x*U[0].z,U[0].z*U[0].z},
							{U[1].x*U[1].x,U[1].x*U[1].z,U[1].z*U[1].z},
							{U[2].x*U[2].x,U[2].x*U[2].z,U[2].z*U[2].z}	};
		double det0 = MatrDet3x3(A0);
		double A1[3][3] = {	{		1 ,U[0].x*U[0].z,U[0].z*U[0].z},
							{		1 ,U[1].x*U[1].z,U[1].z*U[1].z},
							{		1 ,U[2].x*U[2].z,U[2].z*U[2].z}	};
		double det1 = MatrDet3x3(A1);
		double A2[3][3] = {	{U[0].x*U[0].x,		1 ,U[0].z*U[0].z},
							{U[1].x*U[1].x,		1 ,U[1].z*U[1].z},
							{U[2].x*U[2].x,		1 ,U[2].z*U[2].z}	};
		double det2 = MatrDet3x3(A2);
		double A3[3][3] = {	{U[0].x*U[0].x,U[0].x*U[0].z,		1 },
							{U[1].x*U[1].x,U[1].x*U[1].z,		1 },
							{U[2].x*U[2].x,U[2].x*U[2].z,		1 }	};
		double det3 = MatrDet3x3(A3); */

		//exclArea = det3/det0;
		
		//check for possible reduction:
		float2 axii = elli_half_axii(coeff.x,coeff.y,coeff.z);
		float2 c2ld = make_float2(probe.l/2-abs(Dtt.y),shrcnt[6*stride]/2-abs(Dtt.z));
		// full area:
		if ((c2ld.x >= axii.x-epsilon) && (c2ld.y >= axii.x-epsilon))
			exclArea = elli_area(coeff.x,coeff.y,coeff.z);
			//exclArea = 10;
		// partial contact:
		if (abs(c2ld.x) < axii.x-epsilon)
			exclArea = elli_area_symseg(axii.x, axii.y, c2ld.x);
			//exclArea = 20;
		if (abs(c2ld.y) < axii.x-epsilon)
			exclArea = elli_area_symseg(axii.x, axii.y, c2ld.y);
			//exclArea = 30;
		// no contact:
		if ((c2ld.x <= -axii.x+epsilon) || (c2ld.y <= -axii.x+epsilon))
			exclArea = 0;
			//exclArea = 40;
		exclArea *= powf(P[2].x,2);
		exclArea /= powf(R1/probe.a,2);
		
	} else { // parallel:
		float2 P = make_float2(0,0.5f*(y2-(R2*R2-R1*R1)/y2/powf(cos(gamma),2)));
		P.x = sqrtf(R1*R1-P.y*P.y);
		// second tube tip relatively to the first tube's center:
		float tip2[2];
		tip2[0] =	abs(	(shrcnt[0*stride]-probe.r.x)*probe.c.x + 
							(shrcnt[1*stride]-probe.r.y)*probe.c.y + 
							(shrcnt[2*stride]-probe.r.z)*probe.c.z	)
					- shrcnt[6*stride]/2;
		tip2[1] =	tip2[0] + shrcnt[6*stride];
		//full contact:
		if( (tip2[0] <= -probe.l/2+epsilon)&&(tip2[1] >= +probe.l/2-epsilon))
			exclArea = 2*R1*atanf(P.x/P.y)*probe.l;
		if( (tip2[0] > probe.l/2-epsilon) )
			exclArea = 0;
		if( (tip2[0] > -probe.l/2+epsilon) && (tip2[1] < +probe.l/2-epsilon) )
			exclArea = 2*R1*atanf(P.x/P.y)*( (tip2[1]<probe.l/2 ? tip2[1] : probe.l/2) - tip2[0]);	
		//exclArea = 20;
		exclArea /= powf(R1/probe.a,1);
	}
	return exclArea;	
}

__device__ float exclArea_Sph2Sph(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon) {

	float exclArea = 0;

	float R1 = probe.a;
	float R2 = shrcnt[7*stride]+dist;
	float y2 = Dtt.x;

	float3 vec2 = make_float3(	shrcnt[3*stride],
								shrcnt[4*stride],
								shrcnt[5*stride]	);
	float gamma = acosf(cosVec2Vec(probe.c,vec2));
	if (abs(sinf(gamma))>epsilon) {
		float3 P[3];
		P[0] = make_float3(0,0,sqrtf(R2*R2-powf(R1-y2,2))/sinf(gamma));

		P[1] = make_float3(sqrtf(R1*R1-powf(R2-y2,2)),0,0);
		P[1].z = P[1].x*cosf(gamma);

		P[2] = make_float3(0, y2-sqrtf( powf(y2*cosf(gamma),2) + (R2*R2-powf(R1*cosf(gamma),2)) )/powf(sinf(gamma),2),0);
		P[2].x = sqrtf(R1*R1-P[2].y*P[2].y);

		P[0] = unwrapZ(P[0]);
		P[1] = unwrapZ(P[1]);
		P[2] = unwrapZ(P[2]);

		float3 coeff = ellipse_fit(make_float2(P[0].x,P[0].z),make_float2(P[1].x,P[1].z),make_float2(P[2].x,P[2].z));
		exclArea = elli_area(coeff.x,coeff.y,coeff.z);

		exclArea = y2-sqrtf( powf(y2*cosf(gamma),2) + (R2*R2-powf(R1*cosf(gamma),2)) );
		
		//check for possible reduction:
		float2 axii = elli_half_axii(coeff.x,coeff.y,coeff.z);
		float2 c2ld = make_float2(probe.l/2-abs(Dtt.y),shrcnt[6*stride]/2-abs(Dtt.z));
		// full area:
		if ((c2ld.x >= axii.x-epsilon) && (c2ld.y >= axii.x-epsilon))
			exclArea = 10;//exclArea = elli_area(coeff.x,coeff.y,coeff.z);
		// partial contact:
		if (abs(c2ld.x) < axii.x-epsilon)
			exclArea = 20;//exclArea = elli_area_symseg(axii.x, axii.y, c2ld.x);
		if (abs(c2ld.y) < axii.x-epsilon)
			exclArea = 30;//exclArea = elli_area_symseg(axii.x, axii.y, c2ld.y);
		// no contact:
		if ((c2ld.x <= -axii.x+epsilon) || (c2ld.y <= -axii.x+epsilon))
			exclArea = 40;//exclArea = 0;
		//exclArea = axii.x;
	} else { // parallel:
		float2 P = make_float2(0,0.5f*(y2-(R2*R2-R1*R1)/y2/powf(cos(gamma),2)));
		P.x = sqrtf(R1*R1-P.y*P.y);
		// second tube tip relatively to the first tube's center:
		float tip2[2];
		tip2[0] =	abs(	(shrcnt[0*stride]-probe.r.x)*probe.c.x + 
							(shrcnt[1*stride]-probe.r.y)*probe.c.y + 
							(shrcnt[2*stride]-probe.r.z)*probe.c.z	)
					- shrcnt[6*stride]/2;
		tip2[1] =	tip2[0] + shrcnt[6*stride];
		//full contact:
		if( (tip2[0] <= -probe.l/2+epsilon)&&(tip2[1] >= +probe.l/2-epsilon))
			exclArea = 2*R1*atanf(P.x/P.y)*probe.l;
		if( (tip2[0] > probe.l/2-epsilon) )
			exclArea = 0;
		if( (tip2[0] > -probe.l/2+epsilon) && (tip2[1] < +probe.l/2-epsilon) )
			exclArea = 2*R1*atanf(P.x/P.y)*( (tip2[1]<probe.l/2 ? tip2[1] : probe.l/2) - tip2[0]);	
		exclArea = 20;
	}
	return exclArea;	
};

__device__ float exclArea_naive(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon) {

	float exclArea = 0;

	//float y2 = Dtt.x;

	float3 vec2 = make_float3(	shrcnt[3*stride],
								shrcnt[4*stride],
								shrcnt[5*stride]	);
	float gamma = acosf(cosVec2Vec(probe.c,vec2));
	if (abs(sinf(gamma))>epsilon) {
		float a = 2*probe.a/sinf(gamma);
		float b = 2*shrcnt[6*stride]/sinf(gamma);
		exclArea = a*b*sinf(gamma);
		
		//check for possible reduction:
		float2 c2ld = make_float2(probe.l/2-abs(Dtt.y),shrcnt[6*stride]/2-abs(Dtt.z));
		// partial contact:
		if ((abs(c2ld.x) < a)||(abs(c2ld.y) < b))
			exclArea /= 2;
			//exclArea = 20;
		// no contact:
		if ((c2ld.x <= -a) || (c2ld.y <= -b))
			exclArea = 0;
			//exclArea = 40;
		
	} else { // parallel:
		// second tube tip relatively to the first tube's center:
		float tip2[2];
		tip2[0] =	abs(	(shrcnt[0*stride]-probe.r.x)*probe.c.x + 
							(shrcnt[1*stride]-probe.r.y)*probe.c.y + 
							(shrcnt[2*stride]-probe.r.z)*probe.c.z	)
					- shrcnt[6*stride]/2;
		tip2[1] =	tip2[0] + shrcnt[6*stride];
		//full contact:
		if( (tip2[0] <= -probe.l/2+epsilon)&&(tip2[1] >= +probe.l/2-epsilon))
			exclArea = 2*probe.a*probe.l;
		if( (tip2[0] > probe.l/2-epsilon) )
			exclArea = 0;
		if( (tip2[0] > -probe.l/2+epsilon) && (tip2[1] < +probe.l/2-epsilon) )
			exclArea = 2*2*probe.a*( (tip2[1]<probe.l/2 ? tip2[1] : probe.l/2) - tip2[0]);	
	}
	return exclArea;	
}
__device__ float exclArea_par(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon) {

	float exclArea = 0;
	float3 vec2 = make_float3(	shrcnt[3*stride],
								shrcnt[4*stride],
								shrcnt[5*stride]	);
	float gamma = acosf(abs(cosVec2Vec(probe.c,vec2))); //<--- abs()!
	if (abs(sinf(gamma))<=epsilon) {
	 // parallel:
		// second tube tips relatively to the first tube's center:
		float tip2[2];
		tip2[0] =	abs(	(shrcnt[0*stride]-probe.r.x)*probe.c.x + 
							(shrcnt[1*stride]-probe.r.y)*probe.c.y + 
							(shrcnt[2*stride]-probe.r.z)*probe.c.z	)
					- shrcnt[6*stride]/2;
		tip2[1] =	tip2[0] + shrcnt[6*stride];
		exclArea = 2*(probe.a<=shrcnt[7*stride] ? probe.a : shrcnt[7*stride] )*
				(( tip2[1] <  probe.l/2 ? tip2[1] :  probe.l/2 )
			  	-( tip2[0] > -probe.l/2 ? tip2[0] : -probe.l/2 ));

	}
	return (exclArea > 0 ? exclArea : 0);	
}
__device__ float exclArea_skew(float3 Dtt, CNT_t probe, float* shrcnt, int stride, float dist,float epsilon) {

	float exclArea = 0;
	float3 vec2 = make_float3(	shrcnt[3*stride],
								shrcnt[4*stride],
								shrcnt[5*stride]	);
	float gamma = acosf(abs(cosVec2Vec(probe.c,vec2))); 
	if (abs(sinf(gamma))>epsilon) {
		float a = 2*probe.a/sinf(gamma);
		float b = 2*shrcnt[7*stride]/sinf(gamma);
		exclArea = a*b*sinf(gamma);
		// partial contact:
		if ((probe.l/2>=abs(Dtt.y))&&(probe.l/2-abs(Dtt.y)<b/2+b/2*cosf(gamma)))
			exclArea /= 2;
		if ((shrcnt[6*stride]/2>=abs(Dtt.z))&&(shrcnt[6*stride]/2-abs(Dtt.z)<a/2+a/2*cosf(gamma)))
			exclArea /= 2;
		// no contact:
		if (((probe.l/2<abs(Dtt.y))&&(abs(Dtt.y)-probe.l/2>b/2+b/2*cosf(gamma)))
			|| ((shrcnt[6*stride]/2<abs(Dtt.z))&&(abs(Dtt.z)-shrcnt[6*stride]/2>a/2+a/2*cosf(gamma))))
			exclArea = 0;	
	} 
	return (exclArea > 0 ? exclArea : 0);	
}
//-----------------------------------------------------------------------------------------------------
__device__ float curve2surface_limit(float d, float dlim, float R1, float R2, float C6, float C12, float thetaM ) {
	return	-C6*PI*R1/2*thetaM*(1/powf(d,4)-1/powf(dlim,4)*1/(1-thetaM*d/dlim/dlim/(R1/R2+1)))
			+C12*2*PI*R1/5*thetaM*(1/powf(d,10)-1/powf(dlim,10)*1/powf(1-thetaM*d/dlim/dlim/(R1/R2+1),4));
}
