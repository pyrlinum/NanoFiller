#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include "definitions.h"
#include "simPrms.h"
#include "core_cell_interaction_kernel.h"

// calculate the number of contacts per inclusion:
unsigned int get_contact_counts(	simPrms* const 							Simul,
									int										dir,
									thrust::device_vector<unsigned int>*	d_contact_counts	)
{

	printf("Collecting number of contacts per inclusion...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// set up constant memory:
	core_const_Prms_t	constPrms = setup_collect_Ncnct(Simul, dir);
	setup_core(constPrms);

	// Setup arrays:
	thrust::device_vector<unsigned int>	d_realOff( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realOcc( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realAddr(Simul->Domains.grdSize,0);

	// setup 1nd input data structure:
	device_data_struct_t<float> dataSelf = get_selfDataStruct(Simul, d_realAddr, d_realOff, d_realOcc);
	// setup 2nd input data structure:
	device_data_struct_t<float>	dataNeig = dataSelf;
	// setup results data structure:
	device_data_struct_t<unsigned int>	dataRes;
										copy_structure(dataSelf,&dataRes);
										dataRes.num_Fields = 1;
										dataRes.cell_Data = thrust::raw_pointer_cast(d_contact_counts->data());

	// call kernel:
	for (int count = 0; count < Simul->kernelSplit; count++)	{
		int lnchGSize = ( (Simul->Domains.grdSize - count*Simul->kernelSize) < Simul->kernelSize ? (Simul->Domains.grdSize - count*Simul->kernelSize) : Simul->kernelSize );
		set_core_splitIdx(count);

		core_cell_interact_kernel<unsigned int><<<lnchGSize,Simul->Block>>>( dataSelf, dataNeig, dataRes );

		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR: screening out contacts: %s\n",cudaGetErrorString(cuErr));
	}
	cuErr = cudaGetLastError();
	if ( cuErr != cudaSuccess ) printf("ERROR: CUDA kernel launch fail: %s\n",cudaGetErrorString(cuErr));

	// check data: (if any expected contact was missed):
	unsigned int	num_contacts = thrust::reduce(d_contact_counts->begin(),d_contact_counts->end(),(unsigned int) 0);
	printf( "Found contacts: %d\n",num_contacts );

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA Cell Interaction time: %f ms \n",time );

	return num_contacts;
}

// get inclusion pairs of indices and relative positions:
void get_contact_arrays(	simPrms* const 							Simul,
							int										dir,
							thrust::device_vector<unsigned int>&	d_contact_counts,
							thrust::device_vector<unsigned int>*	d_contact_I,
							thrust::device_vector<unsigned int>* 	d_contact_J,
							thrust::device_vector<float>* 			d_contact_tI,
							thrust::device_vector<float>* 			d_contact_tJ	)
{

	printf("Collecting contact data arrays...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// set up constant memory:
	core_const_Prms_t	constPrms = setup_collect_IJt(Simul, dir);
	setup_core(constPrms);

	// Setup arrays:
	thrust::device_vector<unsigned int>	d_realOff( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realOcc( Simul->Domains.grdSize,0);
	thrust::device_vector<unsigned int>	d_realAddr(Simul->Domains.grdSize,0);

	thrust::device_vector<unsigned int> d_cnctAddr(d_contact_counts.size(),0);
	thrust::device_vector<unsigned int> d_cnctOff( d_contact_counts.size(),0);
	thrust::device_vector<unsigned int> d_cnctOcc( d_contact_counts.size(),0);
	thrust::inclusive_scan(d_contact_counts.begin(),d_contact_counts.end(),d_cnctAddr.begin());
	unsigned int dataLen = d_cnctAddr[d_contact_counts.size()-1];

	// results vector:
	thrust::device_vector<float> d_resData(4*dataLen,-1.0);

	// setup 1nd input data structure:
	device_data_struct_t<float> dataSelf = get_selfDataStruct(Simul, d_realAddr, d_realOff, d_realOcc);
	// setup 2nd input data structure:
	device_data_struct_t<float>	dataNeig = dataSelf;
	// setup results data structure:
	device_data_struct_t<float>	dataRes  = get_DataStruct(d_cnctAddr, d_cnctOff, d_cnctOcc, d_resData);

	// call kernel:
	for (int count = 0; count < Simul->kernelSplit; count++)	{
		int lnchGSize = ( (Simul->Domains.grdSize - count*Simul->kernelSize) < Simul->kernelSize ? (Simul->Domains.grdSize - count*Simul->kernelSize) : Simul->kernelSize );
		set_core_splitIdx(count);

		core_cell_interact_kernel<float><<<lnchGSize,Simul->Block>>>( dataSelf, dataNeig, dataRes );

		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR: screening out contacts: %s\n",cudaGetErrorString(cuErr));
	}
	cuErr = cudaGetLastError();
	if ( cuErr != cudaSuccess ) printf("ERROR: CUDA kernel launch fail: %s\n",cudaGetErrorString(cuErr));

	thrust::copy(d_resData.begin()+0*dataLen,d_resData.begin()+1*dataLen,d_contact_I->begin() );
	thrust::copy(d_resData.begin()+1*dataLen,d_resData.begin()+2*dataLen,d_contact_J->begin() );
	thrust::copy(d_resData.begin()+2*dataLen,d_resData.begin()+3*dataLen,d_contact_tI->begin() );
	thrust::copy(d_resData.begin()+3*dataLen,d_resData.begin()+4*dataLen,d_contact_tJ->begin() );

	// check data: (if any expected contact was missed):
	unsigned int	num_unfilled = thrust::count(d_contact_I->begin(),d_contact_I->end(),-1.0);
	if ( num_unfilled>0 ) {	printf("WARNING: Unfilled vacancies left: %i\n",num_unfilled); }

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA Cell Interaction time: %f ms \n",time );
}

