#pragma once
#include "CNT.h"
#include "intersection_dev.h"
#include "memoryIO.h"

__constant__ float C6;	// constant for attractive part of Van der Waals potential
__constant__ float C12;	// constant for attractive part of Van der Waals potential 
__constant__ float vdw_step; // step for integration of van der Waals forces

__device__ float infLine2infLine(float d, float gamma);
__device__ float infLine2Line_0L(float d, float gamma, float L);
__device__ float infLine2infTube(float d, float gamma, float R);

//----------------------------------------------------------------------------------------------------------------------
// Van der Waals interaction:
//----------------------------------------------------------------------------------------------------------------------
/*
extern "C"
__global__ void statKernel_VDW(	float dist_limit,		// shortest distance limit
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
extern __shared__ float dynShr_WdV[];

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
			dynShr_WdV[threadIdx.x] = 0.0f;
		}
		__syncthreads();

	// MAIN CODE:----------------------------------------------------------------- 
		float gamma;
		float3 Dtt = make_float3(0,0,0);
		float d; 
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
					d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,float_precision);
					if ( d < dist_limit ) {
						float3 vec2 = make_float3(	shrArr[i+3*blockDim.x],
													shrArr[i+4*blockDim.x],
													shrArr[i+5*blockDim.x]	);
						gamma = acosf(cosVec2Vec(probe.c,vec2));
						// analytical approximation: 
						if ( (Dtt.y<=probe.l/2-dist_limit)&&(t1<=shrArr[i+5*blockDim.x]/2-dist_limit) ) {
							energy = infLine2Line_0L(Dtt.x,gamma,L)

						
						
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

}	*/
//========================================================================================================
// AUXILARY:

__device__ float infLine2infLine(float d, float gamma) {	// between two infinite lines;
			// attraction:
	return	- C6*PI/(2*powf(d, 4)*sqrtf(sinf(gamma)))
			// repulsion:
			+C12*PI/(5*powf(d,10)*sqrtf(sinf(gamma)));
}

__device__ float infLine2Line_0L(float d, float gamma, float L) {	// between infinite line and line segment (0,L);
			// attraction:
	return	- C6*L*PI*(3*d*d+L*L*(1-cosf(2*gamma)))
			/(2*sqrtf(2)*powf(d, 4)*powf(2*d*d+L*L*(1-cosf(2*gamma)),1.5f))
			// repulsion:
			+C12*L*PI*(315*powf(d,8)+420*powf(d,6)*powf(L,2)+378*powf(d,4)*powf(L,4)+180*powf(d,2)*powf(L,6)+35*powf(L,8)
			-2*L*L*cosf(2*gamma)*(210*powf(d,6)+252*powf(d,4)*powf(L,2)+135*powf(d,2)*powf(L,4)+28*powf(L,6))
			+2*powf(L,4)*cosf(4*gamma)*(63*powf(d,4)+54*powf(d,2)*powf(L,2)+14*powf(L,4)) 
			-18*powf(d,2)*powf(L,6)*cosf(6*gamma)-8*powf(L,8)*cosf(6*gamma)+powf(L,8)*cosf(8*gamma) )
			/(40*sqrtf(2)*powf(d,10)*powf(2*d*d+L*L*(1-cosf(2*gamma)),4.5f));
}

__device__ float infLine2infTube(float d, float gamma, float R) {	// between infinite line and infinite tube
			// attraction:
	return	- C6*2*PI
				*(d*R*(d*d-R*R)+(d*d+R*R)*(d*d+R*R)*atanf((d+R)/(d-R)))
			/powf(d*d-R*R,3)/(d*d+R*R)/sinf(gamma)
			// repulsion:
			+ C12*4*PI
			*(2*(6*powf(d,15)*R+28*powf(d,13)*powf(R,3)+71*powf(d,11)*powf(R,5)+25*powf(d,9)*powf(R,7)
			-25*powf(d,7)*powf(R,9)-71*powf(d,5)*powf(R,11)-28*powf(d,3)*powf(R,13)-6*d*powf(R,15))
			+3*powf(d*d+R*R,4)*(powf(d,8)+16*powf(d,6)*powf(R,2)+36*powf(d,4)*powf(R,4)+16*powf(d,2)*powf(R,6)+powf(R,8))
			*atanf((d+R)/(d-R)) )
			/( 15*powf(d*d-R*R,9)*powf(d*d+R*R,4)*sinf(gamma) );
}