//====================================================================================================================
//										 <<< AUXILARY KERNELS >>>
//--------------------------------------------------------------------------------------------------------------------
#pragma once
#define RENEW_NONE		0
#define RENEW_ADD		1
#define RENEW_REPLACE	2

// add new CNT numbers to occupancies:
extern "C"
__global__ void cuda_renewOcc( char op, 	short *d_dmnOcc,
				short *d_dmnCrt)

{
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	int dmnGridSize = dmnExt[0]*dmnExt[1]*dmnExt[2];

	if ( tid < dmnGridSize ) {
		if (op == RENEW_ADD)
			d_dmnOcc[tid] += d_dmnCrt[tid];
		if (op == RENEW_REPLACE)
			d_dmnOcc[tid] = d_dmnCrt[tid];
		d_dmnCrt[tid] = 0;
	}

}

// calculate current CNT center density:
extern "C"
__device__ int shrAddr(int pSize[3], int x, int y, int z) {
	return x+y*pSize[0]+z*pSize[0]*pSize[1];
}
extern "C"
__global__ void cuda_plotMesh(	int		count,
					int		*d_dmnAddr,
					short		*d_dmnOcc,
					short		*d_Crtd,
					float		*d_result,
					float		*d_curMesh)
{
__shared__ unsigned int blockId;		// Block Idx + count*GridDim 
__shared__ unsigned int selfCNT;		// the number of cnts to be proccessed by this Block
__shared__ float	delta[3];		// step of mesh grid
__shared__ unsigned int patchCRD[3];		// coordinate of the texture patch beginning
__shared__          int patchExt[3];		// extents of the patch
__shared__ unsigned int patchSize;		// size of the patch
__shared__ unsigned int pLoad;			// load per thread while patch writeback
  extern __shared__ float patch[];		// patch of texture mesh 


	if (threadIdx.x == 0) {
		blockId = blockIdx.x + count*gridDim.x;
		selfCNT = d_Crtd[blockId]+d_dmnOcc[blockId];
		
		short x0 = (blockId%dmnExt[0]);			
		short y0 = (blockId/dmnExt[0])%dmnExt[1];	
		short z0 = (blockId/dmnExt[0])/dmnExt[1];	

		patchCRD[0] = floor(((float) x0)/dmnExt[0]*texDim.x);	patchExt[0] = ceil((((float) x0+1))/dmnExt[0]*texDim.x)-patchCRD[0]+1;
		patchCRD[1] = floor(((float) y0)/dmnExt[1]*texDim.y);	patchExt[1] = ceil((((float) y0+1))/dmnExt[1]*texDim.y)-patchCRD[1]+1;
		patchCRD[2] = floor(((float) z0)/dmnExt[2]*texDim.z);	patchExt[2] = ceil((((float) z0+1))/dmnExt[2]*texDim.z)-patchCRD[2]+1;

		patchSize = patchExt[0]*patchExt[1]*patchExt[2];
		pLoad = ceil(((float) patchSize)/blockDim.x);

		delta[0] = (phsScl[0]*dmnExt[0])/texDim.x;
		delta[1] = (phsScl[1]*dmnExt[1])/texDim.y;
		delta[2] = (phsScl[2]*dmnExt[2])/texDim.z;
		
	}
	__syncthreads();

	// patch initialisation:
	for(int i = 0; i<pLoad; i++) {
		int shrKey = threadIdx.x+i*blockDim.x;
		if (shrKey < patchSize) patch[shrKey] = 0.0f;
	}

	__syncthreads();
	//*
	if (threadIdx.x < selfCNT) {
		int	addr = threadIdx.x + (blockId>0?d_dmnAddr[blockId-1]:0);
		
		float x = d_result[addr+0*numCNT];
		float y = d_result[addr+1*numCNT];
		float z = d_result[addr+2*numCNT];

		int	pX = floor(x/delta[0])-patchCRD[0];	x -= (pX+patchCRD[0])*delta[0];	x /= delta[0];
		int	pY = floor(y/delta[1])-patchCRD[1];	y -= (pY+patchCRD[1])*delta[1];	y /= delta[1];
		int	pZ = floor(z/delta[2])-patchCRD[2];	z -= (pZ+patchCRD[2])*delta[2];	z /= delta[2];

		atomicAdd(&patch[shrAddr(patchExt,pX  ,pY  ,pZ  )],(1-x)*(1-y)*(1-z));
		atomicAdd(&patch[shrAddr(patchExt,pX  ,pY  ,pZ+1)],(1-x)*(1-y)*(  z));
		atomicAdd(&patch[shrAddr(patchExt,pX  ,pY+1,pZ  )],(1-x)*(  y)*(1-z));
		atomicAdd(&patch[shrAddr(patchExt,pX  ,pY+1,pZ+1)],(1-x)*(  y)*(  z));
		atomicAdd(&patch[shrAddr(patchExt,pX+1,pY  ,pZ  )],(  x)*(1-y)*(1-z));
		atomicAdd(&patch[shrAddr(patchExt,pX+1,pY  ,pZ+1)],(  x)*(1-y)*(  z));
		atomicAdd(&patch[shrAddr(patchExt,pX+1,pY+1,pZ  )],(  x)*(  y)*(1-z));
		atomicAdd(&patch[shrAddr(patchExt,pX+1,pY+1,pZ+1)],(  x)*(  y)*(  z));
	}
	__syncthreads();

	
		for(int i = 0; i<pLoad; i++) {
		int shrKey = threadIdx.x+i*blockDim.x;
		if (shrKey < patchSize) {
			int pX = ( patchCRD[0]+(shrKey%patchExt[0])		)%texDim.x;
			int pY = ( patchCRD[1]+(shrKey/patchExt[0])%patchExt[1] )%texDim.y;
			int pZ = ( patchCRD[2]+(shrKey/patchExt[0])/patchExt[1] )%texDim.z;
			int glbKey = pX+pY*texDim.x+pZ*texDim.x*texDim.y; 
			atomicAdd(&d_curMesh[glbKey],patch[shrKey]);
		}
	}
	//*/
}


