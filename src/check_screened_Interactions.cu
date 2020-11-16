#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include "definitions.h"
#include "simPrms.h"
#include "core_cell_interaction_kernel.h"

// mark screened pair contacts:
void	check_screened_Interactions(	thrust::device_vector<char>*			isnt_screened,
										thrust::device_vector<unsigned int>&	d_virtAddr,
										thrust::device_vector<unsigned int>&	d_virtOcc,
										thrust::device_vector<float>&			d_virt_inc,
										simPrms* const Simul,
										int dir ) {

	printf("Eliminating screened interactions...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// set up constant memory:
	core_const_Prms_t	constPrms = setup_real2virt(Simul, dir);
	setup_core(constPrms);

	// Setup arrays: (TODO: later replace initial arrays in generation cycle wit arrays of unsigned ints)
	thrust::device_vector<unsigned int>	d_realOff( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realOcc( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realAddr(Simul->Domains.grdSize,0);

	thrust::device_vector<unsigned int>	d_virtOff(Simul->Domains.grdSize,0);

	// setup 1nd input data structure:
	device_data_struct_t<float> dataSelf = get_selfDataStruct(Simul, d_realAddr, d_realOff, d_realOcc);
	// setup 2nd input data structure:
	device_data_struct_t<float>	dataNeig = get_DataStruct(d_virtAddr, d_virtOff, d_virtOcc, d_virt_inc);
	// setup results data structure:
	device_data_struct_t<char>	dataRes  = get_DataStruct(d_virtAddr, d_virtOff, d_virtOcc, *isnt_screened);

	// call kernel:
	for (int count = 0; count < Simul->kernelSplit; count++)	{
		int lnchGSize = ( (Simul->Domains.grdSize - count*Simul->kernelSize) < Simul->kernelSize ? (Simul->Domains.grdSize - count*Simul->kernelSize) : Simul->kernelSize );
		set_core_splitIdx(count);

		core_cell_interact_kernel<char><<<lnchGSize,Simul->Block>>>( dataSelf, dataNeig, dataRes );

		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR: screening out contacts: %s\n",cudaGetErrorString(cuErr));
	}

	cuErr = cudaGetLastError();
	if ( cuErr != cudaSuccess ) printf("ERROR: CUDA kernel launch fail: %s\n",cudaGetErrorString(cuErr));

	// check data:
	unsigned int num_screened0 = thrust::count( isnt_screened->begin(), isnt_screened->end(), VIRT_INC_EMPTY);
	unsigned int num_screened1 = thrust::count( isnt_screened->begin(), isnt_screened->end(), VIRT_INC_PRESENT);
	printf("Final contacts deleted/left: %i/%i \n",num_screened0,num_screened1);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA Cell Interaction time: %f ms \n",time );

}
