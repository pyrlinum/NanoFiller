// This file contains functions to calculate characteristics of contacts between inclusions
// provided array of coordinates, simulation parameters, arrays of indices of inclusions in
// contact and position of contact along inclusion axii

#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include "intersection.h"
#include "definitions.h"
#include "simPrms.h"
#include "vectorMath.h"
#include "get_contact_data.h"


// get contact separation distances:----------------------------------------------------------
// kernel:
__global__ void get_distance_cyl2cyl(	float3			physDim,
										int				kern_offset,
										unsigned int	num_contacts,
										unsigned int	data_offset,
										unsigned int*	d_contact_I,
										unsigned int*	d_contact_J,
										float*			d_contact_tI,
										float*			d_contact_tJ,
										float*			d_coords,
										float*			d_dist			)
{
	unsigned int tid = kern_offset + blockIdx.x*blockDim.x + threadIdx.x;
	float	probeA[MAX_INC_FIELDS];
	float	probeB[MAX_INC_FIELDS];
	unsigned int	idxA,idxB;
	float			tI,tJ;
	float3			pntA,pntB;

	if (tid < num_contacts) {
		// get inclusion indices:
		idxA = d_contact_I[tid];
		idxB = d_contact_J[tid];
		tI	 = d_contact_tI[tid];
		tJ	 = d_contact_tJ[tid];


		// load coordinates into registry:
		for(int i=0;i<MAX_INC_FIELDS; i++) {
			probeA[i] = d_coords[idxA + i*data_offset];
			probeB[i] = d_coords[idxB + i*data_offset];
		}

		// type-specific part (later to be replaced by class method):
		pntA = get_cyl_point(probeA,tI);
		pntB = get_cyl_point(probeB,tJ);
		pntB = get_relative_crd(pntA,pntB,physDim); // no need to account for direction, if the pair was selected - it does not crosses non-periodic boundary

		//d_dist[tid] =  max(MIN_SEPARATION,dist2pnt(pntA,pntB)-get_cyl_radius(probeA)-get_cyl_radius(probeB));
		d_dist[tid] =  dist2pnt(pntA,pntB);
	}
}

// wrapper:
void get_contact_distance(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>& 	d_contact_J,
							thrust::device_vector<float>& 			d_contact_tI,
							thrust::device_vector<float>& 			d_contact_tJ,
							thrust::device_vector<float>* 			d_contact_D	)
{

	printf("Calculating contact distance separation...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;
	int currentDev = -1;		cudaGetDevice(&currentDev);
	cudaDeviceProp curDevProp;	cudaGetDeviceProperties(&curDevProp,currentDev);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// call kernel:
	dim3 block = Simul->Block;
	int num_contacts = d_contact_I.size();
	int grid_extd = (int) ceil((float)num_contacts/block.x)*block.x;
	int num_lnch = (int) ceil((float)grid_extd/curDevProp.maxGridSize[0] );
	int	kern_offset = 0;
	float3 physDim = make_float3(Simul->physDim[0],Simul->physDim[1],Simul->physDim[2]);


	for (int count = 0; count < num_lnch; count++)	{
		int remainSize = grid_extd - count*curDevProp.maxGridSize[0];
		int lnchGSize = ( remainSize < curDevProp.maxGridSize[0] ?  remainSize : curDevProp.maxGridSize[0]);
		dim3 grid(lnchGSize);

		get_distance_cyl2cyl<<<grid,block>>>(	physDim,
												kern_offset,
												num_contacts,
												Simul->Domains.ttlCNT,
												thrust::raw_pointer_cast(d_contact_I.data()),
												thrust::raw_pointer_cast(d_contact_J.data()),
												thrust::raw_pointer_cast(d_contact_tI.data()),
												thrust::raw_pointer_cast(d_contact_tJ.data()),
												Simul->d_result,
												thrust::raw_pointer_cast(d_contact_D->data()));

		kern_offset += lnchGSize;
		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR: estimating separation distances: %s\n",cudaGetErrorString(cuErr));
	}
	cuErr = cudaGetLastError();
	if ( cuErr != cudaSuccess ) printf("ERROR: CUDA kernel launch fail: %s\n",cudaGetErrorString(cuErr));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA time for estimating separation distances: %f ms \n",time );
}

// get contact surface areas:----------------------------------------------------------
// kernel:
__global__ void get_surface_cyl2cyl(	float			rlim,
										float3			physDim,
										int				kern_offset,
										unsigned int	num_contacts,
										unsigned int	data_offset,
										unsigned int*	d_contact_I,
										unsigned int*	d_contact_J,
										float*			d_contact_tI,
										float*			d_contact_tJ,
										float*			d_coords,
										float*			d_surf			)
{
	unsigned int tid = kern_offset + blockIdx.x*blockDim.x + threadIdx.x;
	float	probeA[MAX_INC_FIELDS];
	float	probeB[MAX_INC_FIELDS];
	unsigned int	idxA,idxB;
	float			tI,tJ;
	float3			pntA,pntB,pntB0;

	if (tid < num_contacts) {
		// get inclusion indices:
		idxA = d_contact_I[tid];
		idxB = d_contact_J[tid];
		tI	 = d_contact_tI[tid];
		tJ	 = d_contact_tJ[tid];

		// load coordinates into registry:
		for(int i=0;i<MAX_INC_FIELDS; i++) {
			probeA[i] = d_coords[idxA + i*data_offset];
			probeB[i] = d_coords[idxB + i*data_offset];
		}

		// type-specific part (later to be replaced by class method):
		pntA = get_cyl_point(probeA,tI);
		pntB0 = get_cyl_point(probeB,tJ);
		pntB = get_relative_crd(pntA,pntB0,physDim); // no need to account for non-periodic direction, if the pair was selected - it does not crosses non-periodic boundary
		// translate probeB:
		displace_cyl(probeB,pntB-pntB0);

		float3 Dtt, dtt =  distance_cylinder_capped(probeA,probeB, &Dtt);
		d_surf[tid] =  surf_to_cylinder_capped( probeA, probeB, Dtt, rlim);
	}
}

// wrapper:
void get_contact_surface(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>& 	d_contact_J,
							thrust::device_vector<float>& 			d_contact_tI,
							thrust::device_vector<float>& 			d_contact_tJ,
							thrust::device_vector<float>* 			d_contact_S	)
{

	printf("Calculating contact surface areas...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;
	int currentDev = -1;		cudaGetDevice(&currentDev);
	cudaDeviceProp curDevProp;	cudaGetDeviceProperties(&curDevProp,currentDev);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// call kernel:
	dim3 block = Simul->Block;
	int num_contacts = d_contact_I.size();
	int grid_extd = (int) ceil((float)num_contacts/block.x)*block.x;
	int num_lnch = (int) ceil((float)grid_extd/curDevProp.maxGridSize[0] );
	int	kern_offset = 0;
	float3 physDim = make_float3(Simul->physDim[0],Simul->physDim[1],Simul->physDim[2]);

	for (int count = 0; count < num_lnch; count++)	{
		int remainSize = grid_extd - count*curDevProp.maxGridSize[0];
		int lnchGSize = ( remainSize < curDevProp.maxGridSize[0] ?  remainSize : curDevProp.maxGridSize[0]);
		dim3 grid(lnchGSize);

		get_surface_cyl2cyl<<<grid,block>>>(	Simul->stat_dist_lim,
												physDim,
												kern_offset,
												num_contacts,
												Simul->Domains.ttlCNT,
												thrust::raw_pointer_cast(d_contact_I.data()),
												thrust::raw_pointer_cast(d_contact_J.data()),
												thrust::raw_pointer_cast(d_contact_tI.data()),
												thrust::raw_pointer_cast(d_contact_tJ.data()),
												Simul->d_result,
												thrust::raw_pointer_cast(d_contact_S->data()));

		kern_offset += lnchGSize;
		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR: estimating surface areas: %s\n",cudaGetErrorString(cuErr));
	}
	cuErr = cudaGetLastError();
	if ( cuErr != cudaSuccess ) printf("ERROR: CUDA kernel launch fail: %s\n",cudaGetErrorString(cuErr));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA time for estimating surface areas: %f ms \n",time );
}

// get segments conductance:----------------------------------------------------------

// wrapper:
void get_segment_length(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_segment_I,
							thrust::device_vector<float>& 			d_segment_T,
							thrust::device_vector<unsigned int>*	d_diff_pos,
							thrust::device_vector<unsigned int>*	d_diff_cnts,
							thrust::device_vector<float>* 			d_segment_L,
							thrust::device_vector<float>* 			d_segment_A		) {

	// basic version - for cylinders of constant diameter only (scaled conductance is s/l):

	unsigned int num_contacts_full = d_segment_T.size();

	// get differences in scaled length:
	thrust::device_vector<unsigned int>	d_segment_id(d_segment_I);
	thrust::device_vector<float>		d_segment_dt(num_contacts_full,0);
	thrust::adjacent_difference(	d_segment_T.begin(),
									d_segment_T.end(),
									d_segment_dt.begin()	);

	// remove cross-inclusion elements by writing empty_scale and partitioning vector of differences:
	thrust::device_vector<unsigned int>	d_voids_I(d_diff_pos->size(),	VOID_ELEC);
	thrust::device_vector<float>		d_voids_F(d_diff_pos->size(),	EMPTY_SCALE);
	thrust::scatter(	d_voids_I.begin(),
						d_voids_I.end(),
						d_diff_pos->begin(),
						d_segment_id.begin()	);
	thrust::scatter(	d_voids_F.begin(),
						d_voids_F.end(),
						d_diff_pos->begin(),
						d_segment_dt.begin()	);
	thrust::device_vector<unsigned int>::iterator	new_id_end =
				thrust::stable_partition(	d_segment_id.begin(),
											d_segment_id.end(),
											unary_not_equal_to<unsigned int>(VOID_ELEC)	);
	thrust::device_vector<float>::iterator	new_dt_end =
				thrust::stable_partition(	d_segment_dt.begin(),
											d_segment_dt.end(),
											unary_not_equal_to<float>(EMPTY_SCALE)	);
	unsigned int	partitioned_size = thrust::distance(d_segment_id.begin(),	new_id_end);
	d_segment_id.resize(partitioned_size);
	d_segment_dt.resize(partitioned_size);

	// get difference counts (uniq_counts-1) and new positions:
	thrust::transform(	d_diff_cnts->begin(),
						d_diff_cnts->end(),
						d_diff_cnts->begin(),
						decrement<unsigned int>()	);
	thrust::exclusive_scan(	d_diff_cnts->begin(), d_diff_cnts->end(), d_diff_pos->begin() );

	// Scale dt by 1/a (t has units of l):
	thrust::device_ptr<float> rad_start_ptr = thrust::device_pointer_cast(Simul->d_result+7*Simul->Domains.ttlCNT);
	thrust::gather( d_segment_id.begin(), d_segment_id.end(), rad_start_ptr, d_segment_A->begin());
	//*
	thrust::transform(	d_segment_dt.begin(),
						d_segment_dt.end(),
						d_segment_A->begin(),
						d_segment_L->begin(),
						thrust::divides<float>()	);/*/
	thrust::copy(d_segment_dt.begin(),d_segment_dt.end(),d_segment_L->begin()); //*/
	d_segment_L->resize(d_segment_dt.size());
	d_segment_A->resize(d_segment_dt.size());

}
