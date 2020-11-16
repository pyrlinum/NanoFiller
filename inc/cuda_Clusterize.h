//====================================================================================================================
//										 <<< Clusterization KERNEL & DEVICE FUNCTIONS >>>
//====================================================================================================================
// This CUDA kernel & device functions
#pragma once
#include "CNT.h"
#include "intersection_dev.h"
#include "memoryIO.h"
#include "stat_distributions.h"


// (Load balanced)  main kernel:
extern "C"
__global__ void cuda_Clusterize2Old(int		count,
									char3	stride,
									float	gamma_lim,
									float	dist_lim,
									int		*d_dmnAddr,
									short	*d_dmnOcc,
									float	*d_result,
									short	*d_Crtd)
{
__shared__ unsigned int blockId;	// Block Idx + count*GridDim 
__shared__ unsigned int selfCNT;	// the number of cnts to be proccessed by this Block
__shared__ unsigned int neigCNT;	// the number of cnts with which intersection will be checked
__shared__ unsigned int neighID;	// the global ID of  neighbouring cell
__shared__ float shrArr[9*BLOCK];
__shared__ int devAddr[2];
__shared__ int neiAddr[2];
__shared__ float3 neigDspl;

	if (threadIdx.x == 0)
		blockId = blockIdx.x + count*gridDim.x;
	__syncthreads();
		
		// obtain neighbouring cell number and prepare dynamic array to store cell occupancy and border addresses
		if (threadIdx.x == 0) {
			selfCNT = d_Crtd[blockId];													// Newly generated
			devAddr[0] = (blockId>0?d_dmnAddr[blockId-1]:0) + d_dmnOcc[blockId];		// Start posision (Addr + Occ)
			devAddr[1] = d_dmnAddr[blockId];											// End posision
		} 
		__syncthreads();
	
		// Load CNTs in register file:
		CNT_t probe;
		GlbRead(selfCNT,shrArr,devAddr[0],d_result,numCNT);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );
		__syncthreads();
		shrClean(shrArr);

		// get cnts from neighbouring domains:
		if (threadIdx.x == 0 ) {
			neighID = stride2glbID(stride, blockId, dmnExt);
			neigCNT =  d_dmnOcc[neighID];
			neiAddr[0] = (neighID>0?d_dmnAddr[neighID-1]:0);
			char3 shift  = stride2dspl(stride,blockId, dmnExt); 
			neigDspl = make_float3(	shift.x*dmnExt[0]*phsScl[0],
									shift.y*dmnExt[1]*phsScl[1],
									shift.z*dmnExt[2]*phsScl[2]);
		}
		__syncthreads();
		GlbRead(neigCNT,shrArr,neiAddr[0],d_result,numCNT);
		tarnsCNT(neigCNT, neigDspl, shrArr);
	
		// check intersection inside domain cells
		if (threadIdx.x < selfCNT) {
			//float gamma_lim = asinf(dist_lim/probe.l/2);
			for(int i=0;i<neigCNT;i++) {	
				float3 r2 = make_float3(	shrArr[i+1*blockDim.x],
											shrArr[i+2*blockDim.x],
											shrArr[i+3*blockDim.x]	);
				if (norm3(probe.r-r2)< (probe.l>=shrArr[i+6*blockDim.x]?probe.l:shrArr[i+6*blockDim.x])/2 ) {
					float3 c2 = make_float3(	shrArr[i+3*blockDim.x],
												shrArr[i+4*blockDim.x],
												shrArr[i+5*blockDim.x]	);
					float gamma = acosf(abs(cosVec2Vec(probe.c,c2)));
					if ( gamma <= gamma_lim) {
						float3 Dtt;
						float d = cnt_minDist(&Dtt,probe,shrArr,blockDim.x,i,epsilon);
						if (d <= dist_lim) {
							probe.r = probe.r + probe.c*Dtt.y;
								 r2 = r2 + c2*Dtt.z;
							probe.r = (probe.r-r2);
							probe.r = probe.r - c2*dotProd(probe.r,c2);
							probe.r = probe.r*(1.0f/norm3(probe.r));
							probe.r = r2 + probe.r*(probe.a+shrArr[i+7*blockDim.x]+separation);
							probe.r = probe.r - c2*(dotProd(probe.c,c2)*Dtt.y);
							probe.c = c2;
						}
					}
				}
			}
		}
		
		// CLEAN & WRITE RESULTS TO GLOBAL MEMORY:
		shrClean(shrArr);
		if (threadIdx.x < selfCNT) reg2shrCNT(probe, shrArr, threadIdx.x);
		GlbWrite(selfCNT,shrArr,devAddr[0],d_result,numCNT);

}
