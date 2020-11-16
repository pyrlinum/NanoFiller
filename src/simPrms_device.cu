#include <cuda_runtime.h>
#include <stdio.h>
#include "simPrms.h"

// set k field to domain index:
extern "C"
__global__ void		set_k2dmnIdx(	int		count,
									int		gridSize,
									int		offset,
									int*	d_dmnAddr,
									float*	d_result	)
{
// current cell:
__shared__ int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell

	// get starting address of this cell inclusions
	// also mark cell as electrode or internal (selfFLG):
	if (threadIdx.x == 0)	{
		// default values for a cell not to be processed:
		dmnAddr_startSelf = -1;
		selfCNT = 0;

		selfID = blockIdx.x + count*gridDim.x;
		if ( ( selfID < gridSize ) ) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnAddr[selfID]-dmnAddr_startSelf;
		}
	}
	__threadfence_block();
	__syncthreads();

	// mark inclusion as belonging to electrode or conducting net
	if (threadIdx.x < selfCNT )	{
		int pidx = dmnAddr_startSelf + threadIdx.x;
		d_result[pidx+8*offset] = selfID;
	}
}

int			cuda_SetK2dmnIdx( simPrms*	Simul ) 	{
	printf("Setting k-value to cell Idx:\n");

	int	dmnGridSize = Simul->Domains.ext[0]*Simul->Domains.ext[1]*Simul->Domains.ext[2];
	//printf("dmnGridSize:\t%i\n",dmnGridSize);
	//printf("kernelSplit:\t%i\n",Simul->kernelSplit);
	//printf("kernelSize:\t%i\n",Simul->kernelSize);
	//printf("Domains.ttlCNT\t%i\n",Simul->Domains.ttlCNT);

	for(int count=0; count<Simul->kernelSplit; count++)	{
		int lnchGSize = ( (dmnGridSize - count*Simul->kernelSize) < Simul->kernelSize ? (dmnGridSize - count*Simul->kernelSize) : Simul->kernelSize );
		set_k2dmnIdx<<<lnchGSize,Simul->Block>>>(	count,
													dmnGridSize,
													Simul->Domains.ttlCNT,
													Simul->Domains.d_dmnAddr,
													Simul->d_result	);
	}

	return cudaGetLastError();
}
