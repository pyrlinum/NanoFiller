//====================================================================================================================
//										 <<< INTERSECTION KERNELS >>>
//====================================================================================================================
#pragma once
#include "CNT.h"
#include "intersection_dev.h"
#include "memoryIO.h"

//--------------------------------------------------------------------------------------------------------------------
// delete intersecting CNTs inside domains:
extern "C"
__global__ void cuda_InternalIsect_noLck(	int *d_dmnAddr,
											short *d_dmnOcc,
											int *d_masks,
											float *d_result,
											short *d_Crtd)
{


	__shared__ int mask[6];
	__shared__ int vol;
	__shared__ unsigned int selfCNT;	// the number of cnts to be proccessed by this Block
	__shared__ float shrArr[9*BLOCK];
	extern __shared__ int dynShr[];

	// obtain segment boundaries:
	if (threadIdx.x == 0) {
		selfCNT = 0;
		vol = getMask(mask,d_masks);
	}
	__syncthreads();

	// set shared arrays:
	int	*devAddr = (int*)	&(dynShr[0*vol]);
	int	*shrAddr = (int*)	&(dynShr[2*vol]);
	int	*shrCrtd = (int*)	&(dynShr[4*vol]);

	// prepare dynamic array to store cell occupancy and border addresses
	prepareDynamicShr(vol, mask, devAddr, d_dmnAddr, d_dmnOcc,  shrCrtd, d_Crtd, dmnExt); 

	if (threadIdx.x < vol) atomicAdd(&selfCNT,shrCrtd[threadIdx.x]);
	__syncthreads();

	// Load CNTs in register file:
	CNT_t probe;
	int cntGlbAddr = -1;		// the address of current cnt in global memory
	short cell	= -1;			// number of current cell
	short pos	= threadIdx.x;	// position of the current thread in cell
	if (threadIdx.x < selfCNT) {
		bool flag = true;
		while (flag) {
			cell++;
			flag = (pos+1 > shrCrtd[cell]);
			if (flag)
				pos -= shrCrtd[cell];
		}
		cntGlbAddr = devAddr[cell+0*vol]+pos;
	}

	#pragma unroll
	for (int i=0;i<9;i++)
		shrArr[threadIdx.x + i*blockDim.x] = (cntGlbAddr>=0?d_result[cntGlbAddr + i*numCNT]:-1);
	__threadfence_block();
	__syncthreads(); 

	// check intersection inside domain cells
	if (threadIdx.x < selfCNT)  {
		probe = shr2regCNT(shrArr);
		for(int i=threadIdx.x+1;i<threadIdx.x+shrCrtd[cell]-pos;i++)
			if (cnt_intersec_shr(probe,shrArr,blockDim.x, i,epsilon,separation,sc_frac)) {
				probe = make_emptyCNT();
			} 
	}
	__syncthreads();

	if (threadIdx.x < selfCNT) reg2shrCNT( probe, shrArr); 
	
	// clean memory before write:
	if (threadIdx.x < selfCNT) 
		#pragma unroll
		for (int i=0;i<9;i++)
			d_result[cntGlbAddr + i*numCNT] = 0;
	__threadfence_block();
	
	// WRITE RESULTS TO GLOBAL MEMORY:
	writeGlbMem(vol, mask, devAddr, shrAddr, d_Crtd, shrArr, d_result, dmnExt, numCNT);
		
}

//--------------------------------------------------------------------------------------------------------------------
// (Load balanced)  delete intersecting CNTs within neibouring domains:
extern "C"
__global__ void cuda_NewExtIsectLB_parted(	char part,
											int count,
											char3 stride,
											int *d_dmnAddr,
											short *d_dmnOcc,
											float *d_result,
											short *d_Crtd,
											int *d_dmnLck)
{
__shared__ unsigned int selfID;	// Block Idx + count*GridDim 
__shared__ unsigned int selfCNT;	// the number of cnts to be proccessed by this Block
__shared__ unsigned int neigCNT;	// the number of cnts with which intersection will be checked
__shared__ unsigned int neigID;	// the global ID of  neighbouring cell
__shared__ unsigned int actvTrd;	// the number of active threads
__shared__ unsigned int avgLoad;	// the number of pairs to be evaluated by each active thread
__shared__ float shrArr[9*BLOCK];
__shared__ int devAddr[2];
__shared__ int neiAddr[2];
__shared__ int shrCrtd[1];
__shared__ float3 neigDspl;
__shared__ bool exec_flag;

	if (threadIdx.x == 0)	{
		selfID = blockIdx.x + count*gridDim.x;
		neigID = stride2glbID(stride, selfID, dmnExt);
		int lin_displ = stride.x+stride.y*dmnExt[0]+stride.z*dmnExt[0]*dmnExt[1];
		if (lin_displ < 0) {	// revert order
			selfID = dmnExt[0]*dmnExt[1]*dmnExt[2]-1 - selfID;
			neigID = dmnExt[0]*dmnExt[1]*dmnExt[2]-1 - neigID;
			lin_displ = -lin_displ;
			stride.x = -stride.x; stride.y = -stride.y; stride.z = -stride.z;
		}
		exec_flag = ( ((selfID/lin_displ)%2 == part)&&((neigID/lin_displ)%2 != part) )||((part == 2)&&(d_dmnLck[selfID]==0));
		__threadfence();
	}
	__syncthreads();
	
	if(exec_flag)	{
	// CODE_BEGIN:-------
		// obtain neighbouring cell number and prepare dynamic array to store cell occupancy and border addresses
		if (threadIdx.x == 0) {
			shrCrtd[0] = d_Crtd[selfID];												// Newly generated
			devAddr[0] = (selfID>0?d_dmnAddr[selfID-1]:0) + d_dmnOcc[selfID];			// Start posision (Addr + Occ)
			devAddr[1] = d_dmnAddr[selfID];												// End posision
			selfCNT = shrCrtd[0];
			actvTrd = (selfCNT > 0 ? (blockDim.x/selfCNT)*selfCNT : 0);
			avgLoad = 0;
		} 
		__threadfence_block();
		__syncthreads();
	
		// Load CNTs in register file:
		CNT_t probe;

		GlbRead(selfCNT,shrArr,devAddr[0],d_result,numCNT);

		// changing the line for better load ballance:
		probe = ( threadIdx.x < actvTrd ? shrID2regCNT(threadIdx.x%selfCNT,shrArr) : make_emptyCNT() );
		__threadfence_block();
		__syncthreads();
		// clean shared memory:
		shrClean(shrArr);

		// get cnts from neighbouring domains:
		if (threadIdx.x == 0 ) {
			neigCNT =  d_Crtd[neigID];
			neiAddr[0] = (neigID>0?d_dmnAddr[neigID-1]:0)+d_dmnOcc[neigID];
			char3 shift  = stride2dspl(stride,selfID, dmnExt); 
			neigDspl = make_float3(	shift.x*dmnExt[0]*phsScl[0],
									shift.y*dmnExt[1]*phsScl[1],
									shift.z*dmnExt[2]*phsScl[2]);
		}
		__syncthreads();


		GlbRead(neigCNT,shrArr,neiAddr[0],d_result,numCNT);
		tarnsCNT(neigCNT, neigDspl, shrArr);
		__threadfence_block();
		__syncthreads();

		// analyse loads:
		if ((threadIdx.x ==0)&&(actvTrd > 0)) {
			avgLoad = ceil( ((float) selfCNT*neigCNT)/actvTrd );
			actvTrd = selfCNT*ceil( ((float) neigCNT)/avgLoad );
		}
		__syncthreads();
	
		// check intersection inside domain cells
		if (threadIdx.x < actvTrd) {
			short lineB = avgLoad*(threadIdx.x/selfCNT);
			short lineE = avgLoad*(threadIdx.x/selfCNT+1);
			lineE = (lineE<neigCNT?lineE:neigCNT);
			for(int i=lineB;i<lineE;i++)
				if (cnt_intersec_shr(probe,shrArr,blockDim.x, i,epsilon,separation,sc_frac)) {
					probe = make_emptyCNT();
				}
		}
		__syncthreads();

		// gather results:
		__threadfence_block();
		shrClean(shrArr);

		if (threadIdx.x < selfCNT) {
			reg2shrCNT(probe, shrArr, threadIdx.x);
		}

		short counter = (selfCNT > 0 ? actvTrd/selfCNT : 0 );
		for (int i = 1; i < counter; i++) {
			__threadfence_block();
			__syncthreads();									// do you need this???
			if (threadIdx.x/selfCNT == i) {
				int pos = threadIdx.x%selfCNT;
				//if ((probe.k < shrArr[pos+8*blockDim.x] )) {
				if ( (probe.l<0)&&(shrArr[pos+6*blockDim.x]>0) ) {
					reg2shrCNT(probe, shrArr, pos);
				}
			}
		}
		__threadfence_block();
		__syncthreads(); 

		probe = make_emptyCNT();

		if ( threadIdx.x < selfCNT ) {
			probe = shr2regCNT(shrArr);
		}
		__syncthreads();

		// clean memory before write:
		shrClean(shrArr);

		GlbClean(selfCNT, shrArr, devAddr[0], d_result, numCNT);

		
		// WRITE RESULTS TO GLOBAL MEMORY:
		if (selfCNT>0)
			reWriteGlbMem(selfCNT, probe, shrCrtd, devAddr, shrArr, d_result, numCNT);
	
		// write created CNTs per domain cell number:
		if ((threadIdx.x == 0)&&(selfCNT > 0))
			d_Crtd[selfID] = shrCrtd[0];
	
	// CODE_END:------- 
		if (threadIdx.x == 0 ) d_dmnLck[selfID] = 1;
	} else if (threadIdx.x == 0 ) d_dmnLck[selfID] = ( d_dmnLck[selfID]<1 ? 0 : 1 );

	if (part == 2) d_dmnLck[selfID] = -1;	// free all locks
}
//--------------------------------------------------------------------------------------------------------------------
// (Load balanced)  check intersecting CNTs with old:
extern "C"
__global__ void cuda_OldExtIsectLB(	int count,
									char3 stride,
									int *d_dmnAddr,
									short *d_dmnOcc,
									float *d_result,
									short *d_Crtd)
{
__shared__ unsigned int blockId;	// Block Idx + count*GridDim 
__shared__ unsigned int selfCNT;	// the number of cnts to be proccessed by this Block
__shared__ unsigned int neigCNT;	// the number of cnts with which intersection will be checked
__shared__ unsigned int neighID;	// the global ID of  neighbouring cell
__shared__ unsigned int actvTrd;	// the number of active threads
__shared__ unsigned int avgLoad;	// the number of pairs to be evaluated by each active thread
__shared__ float shrArr[9*BLOCK];
__shared__ int devAddr[2];
__shared__ int neiAddr[2];
__shared__ int shrCrtd[1];
__shared__ float3 neigDspl;

	if (threadIdx.x == 0)
		blockId = blockIdx.x + count*gridDim.x;
	__syncthreads();
		
		// obtain neighbouring cell number and prepare dynamic array to store cell occupancy and border addresses
		if (threadIdx.x == 0) {
			selfCNT = d_Crtd[blockId];													// Newly generated
			devAddr[0] = (blockId>0?d_dmnAddr[blockId-1]:0) + d_dmnOcc[blockId];		// Start posision (Addr + Occ)
			devAddr[1] = d_dmnAddr[blockId];											// End posision
			actvTrd = (selfCNT > 0 ? (blockDim.x/selfCNT)*selfCNT : 0);
			avgLoad = 0;
		} 
		__syncthreads();
	
		// Load CNTs in register file:
		CNT_t probe;
		GlbRead(selfCNT,shrArr,devAddr[0],d_result,numCNT);

		// changing the line for better load ballance:
		probe = ( threadIdx.x < actvTrd ? shrID2regCNT(threadIdx.x%selfCNT,shrArr) : make_emptyCNT() );
		__syncthreads();
		// clean shared memory:
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

		// analyse loads:
		if ((threadIdx.x ==0)&&(actvTrd > 0)) {
			avgLoad = ceil( ((float) selfCNT*neigCNT)/actvTrd );
			actvTrd = selfCNT*ceil( ((float) neigCNT)/avgLoad );
		}
		__syncthreads();
	
		// check intersection inside domain cells
		if (threadIdx.x < actvTrd) {
			short lineB = avgLoad*(threadIdx.x/selfCNT);
			short lineE = avgLoad*(threadIdx.x/selfCNT+1);
			lineE = (lineE<neigCNT?lineE:neigCNT);
			for(int i=lineB;i<lineE;i++)
				if (cnt_intersec_shr(probe,shrArr,blockDim.x, i,epsilon,separation,sc_frac)) {
					probe = make_emptyCNT();
				}
		}
		__syncthreads();

		// gather results:
		__threadfence();
		shrClean(shrArr);

		if (threadIdx.x < selfCNT) 
			reg2shrCNT(probe, shrArr, threadIdx.x);
		short counter = (selfCNT > 0 ? actvTrd/selfCNT : 0 );
		for (int i = 1; i < counter; i++) {
			__threadfence();
			__syncthreads();
			if (threadIdx.x/selfCNT == i) {
				int pos = threadIdx.x%selfCNT;
				//if ((probe.k < shrArr[pos+8*blockDim.x] )) {
				if ( (probe.l<0)&&(shrArr[pos+6*blockDim.x]>0) ) {
					reg2shrCNT(probe, shrArr, pos);
				}
			}
		}
		
		__syncthreads(); 

		probe = make_emptyCNT();

		if ( threadIdx.x < selfCNT ) {
			probe = shr2regCNT(shrArr);
		}
		__syncthreads();

		// clean memory before write:
		shrClean(shrArr);

		GlbClean(selfCNT, shrArr, devAddr[0], d_result,numCNT);
		
		// WRITE RESULTS TO GLOBAL MEMORY:
		if (selfCNT>0)
			reWriteGlbMem(selfCNT, probe, shrCrtd, devAddr, shrArr, d_result, numCNT);
	
		// write created CNTs per domain cell number:
		if ((threadIdx.x == 0)&&(selfCNT > 0))
			d_Crtd[blockId] = shrCrtd[0];

}
//==============================================================================================================
