// This kernel provides basic wrapper for cell-by-cell processing of inclusions interactions
// The role of the kernel is to load inclusions data of the appropriate cells, apply periodic
// translation, perform processing (estimate distance / detect collisions) and write back the
// global memory. Interaction parameters should be hidden in constant memory
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "definitions.h"
#include "simPrms.h"
#include "memoryIO.h"
#include "intersection_dev.h"
#include "intersection.h"
#include "core_cell_interaction_kernel.h"

#include "CNT.h"
//#include "conductKernels.h"
// Constant memory variables used here
 __constant__	interact_prms_t	inter_prms;				// parameters of interaction calculation
 __constant__	int				core_splitIdx;				// Id of share of grid, split if too large
 __constant__	int				core_splitSize;				// Size of share of grid, split if too large
 __constant__	int 			core_peri_dir;				// periodic flags (7=0b0111 - periodic in XYZ)
 __constant__	int				core_grdExt[3];				// extents of domain cells' grid
 __constant__	float			core_phsDim[3];				// physical dimensions of sample
 __constant__	int				core_bufLayer;				// Buffer layer thickness;
 __constant__	int				core_order_lim;				// maximum order of neighbour cell to check;
 __constant__	bool			core_upper_tri;				// whether the processing is upper triangular order or full
 __constant__	unsigned int	core_numIter;				// number of iteration to be used to process neighbour cell interactions
 __constant__	bool			core_reWrt_self;			// flag whether current cell needs to be rewritten

// Self -	related to current cell, will reside in register memory
//			must always relate to real cell with the number of inclusions <= BLOCK
// Neig -	related to neighbouring cell, will reside in shared memory
//			can be same as Self or other cell with number of inclusions > BLOCK, read in blocks

// auxiliary functions:
inline __device__	char3	get_stride(unsigned int iter) {
	char3	str = make_char3(0,0,0);
	int	offset = 2*core_order_lim+1;
	str.x = (iter%offset)		 - core_order_lim;
	str.y = (iter/offset)%offset - core_order_lim;
	str.z = (iter/offset)/offset - core_order_lim;
	return	str;
}

inline  __device__ int get_dmnIndex(	unsigned int	selfID,
                                        		 int	bufLayer ) {

	int pos = 0;

	if ( (core_peri_dir&1)==0 ) { pos = (selfID%core_grdExt[0])					- bufLayer; }
	if ( (core_peri_dir&2)==0 ) { pos = (selfID/core_grdExt[0])%core_grdExt[1]	- bufLayer; }
	if ( (core_peri_dir&4)==0 ) { pos = (selfID/core_grdExt[0])/core_grdExt[1]	- bufLayer; }

	int dmnID = selfID;
	if (pos < 0) { dmnID = -2; }
	if (pos >= (core_grdExt[((7-core_peri_dir)>>1)]-2*bufLayer) ) { dmnID = -1; }

	return dmnID;
}

// interaction processing function
template<class Tr>
__device__ void cell_interaction(	bool			sameFLG,					// whether self and neighbour cells are the same
									interact_prms_t* params,						// interaction parameters
									unsigned int	selfID,						// ID of the processing cells
									unsigned int	selfINC,					// number of objects in current cell
									float*			probe_raw,					// pointer to register object of current thread
									unsigned int	neigID,						// ID of the processing cells
									unsigned int	neigOFF,					// offset of the current chunk
									unsigned int	neigLOAD,					// number of neighbour objects in current chunk
									device_data_struct_t<float>*	shrdCell,	// pointer to shared array of neighbour objects
									device_data_struct_t<float>*	dataSelf,
									device_data_struct_t<float>*	dataNeig,
									device_data_struct_t<Tr>*	dataRes			// pointer of global memory to store results
)  {
	// initial preparation:------------------------------------
	unsigned int	num_contacts = 0;
	unsigned int	partA_addr,partB_addr;
	unsigned int	addrStart;

	if (threadIdx.x<selfINC) {
		switch(params->process_f) {
		case INTERSECT_ELIM:
			break;
		case CONTACT_COUNT:
			// start of the current cell in results memory
			addrStart = dataRes->get_cell_start(selfID) + dataRes->cell_Off[selfID] + threadIdx.x;
			break;
		case CONTACT_RECORD:
			partA_addr = dataSelf->get_cell_start(selfID) + dataSelf->cell_Off[selfID] + threadIdx.x;
			partB_addr = dataNeig->get_cell_start(neigID) + neigOFF;
			// address of results array segment reserved for the contacts of current inclusion
			addrStart = dataRes->get_cell_start(partA_addr) + dataRes->cell_Occ[partA_addr];
			break;
		case VIRTUAL_CHECK:
			// start of the neighbour cell (of virtual inclusions) in results memory
			addrStart = dataRes->get_cell_start(neigID)+neigOFF;
			break;
		default:
			printf("WARNING: Unrecognised cell interaction processing specified %i\n",params->process_f);
			break;
		}
	}
	__threadfence_block();
	__syncthreads();

	// Swipe over shared memory:------------------------------------
	unsigned int startJ = (sameFLG ? threadIdx.x+1 : 0);
	for(unsigned int J = 0; J<neigLOAD; J++ ) {
		if (threadIdx.x<selfINC && J>=startJ) {
			float3 Dtt,dtt = distance_cylinder_capped(probe_raw,shrdCell->cell_Data, J, &Dtt);
			//float3 Dtt;
			//float d = distance_cylinder_capped(probe_raw,shrdCell->cell_Data, J, &Dtt);
			//if ( if_intersects_cylinder_capped(d,probe_raw,shrdCell->cell_Data, J, params->separation, params->soft_frac) ) {
			if ( if_intersects_cylinder_capped(dtt.x,probe_raw,shrdCell->cell_Data, J, params->separation, params->soft_frac) ) {
				switch ( params->process_f ) {
				case INTERSECT_ELIM:
					// void the current inclusion
					//regClean(probe_raw, dataSelf->num_Fields);
					memset(probe_raw,DEF_EMPTY_VAL,dataSelf->num_Fields);
					if (sameFLG)	shrdCell->put_item(0,threadIdx.x,probe_raw);
					break;
				case CONTACT_COUNT:
					// simply use contact counter
					//printf("Contact detected: %i - %i d=%f {xN=%f} \n",threadIdx.x,J,d,shrdCell->cell_Data[J]);
					break;
				case CONTACT_RECORD:
					// save I,J pair
					dataRes->cell_Data[addrStart+num_contacts+0*dataRes->field_Off] = partA_addr;
					dataRes->cell_Data[addrStart+num_contacts+1*dataRes->field_Off] = partB_addr + J;
					dataRes->cell_Data[addrStart+num_contacts+2*dataRes->field_Off] = dtt.y; //dtt.x; //dtt.y; //dtt.x; //dtt.y; // t[0]
					dataRes->cell_Data[addrStart+num_contacts+3*dataRes->field_Off] = dtt.z; //surf_to_cylinder_capped( probe_raw, shrdCell->cell_Data, blockDim.x, J, Dtt, params->separation);// ;Dtt.z; // t[1]
					break;
				case VIRTUAL_CHECK:
					// mark the contact pair as empty if intersection occurs at within the length of virtual inclusion:
					if ( (body_intersect_cylinder_capped(Dtt.z, shrdCell->cell_Data, J) )) {
						dataRes->cell_Data[addrStart+J] = VIRT_INC_EMPTY;
					}
					break;
				default:
					break;
				}
				num_contacts+=1;
			}// end of interaction processing if
		} // end of thread condition
		__threadfence_block();
		__syncthreads();
	} // end of for-loop iteration

	// WATCHOUT: may need correction
	// Post-swipe operations:------------------------------------
	if (threadIdx.x<selfINC) {
		switch(params->process_f) {
		case INTERSECT_ELIM:
			break;
		case CONTACT_COUNT:
			dataRes->cell_Data[addrStart] += num_contacts;
			//atomicAdd((unsigned int*) &(dataRes->cell_Data[addrStart]), num_contacts );
			break;
		case CONTACT_RECORD:
			dataRes->cell_Occ[partA_addr] += num_contacts;
			break;
		case VIRTUAL_CHECK:
			break;
		default:
			break;
		}
	}
}

//-----------------------------------------------------------------------------------------------
// CUDA core kernel:
template<class Tr>
__global__ void core_cell_interact_kernel(	device_data_struct_t<float>	dataSelf,
											device_data_struct_t<float>	dataNeig,
											device_data_struct_t<Tr>	dataRes		)
{
// current cell:
__shared__ unsigned int	selfID;		// the global ID of current cell
__shared__ int			selfFLG;	// flag if the current cell needs processing
__shared__ unsigned int selfINC;	// the number of inclusions in current cell
// neighbour cell:
__shared__ char3		stride;		// stride for current order of cell
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ unsigned int	neigID;		// the global ID of neighbouring cell
__shared__ int			neigFLG;	// flag if the neighbour cell needs processing
__shared__ unsigned int neigINC;	// the number of inclusions in a neighbouring cell - total
__shared__ unsigned int neigLOAD;	// the number of inclusions in a neighbouring cell - currently processed
__shared__ unsigned int neigOFF;	// offset of the fraction of neighbouring cell load to be processed on current iteration
__shared__ unsigned int neigSTPs;	// number of iterations to process the neighbouring cell

__shared__ bool			procFLG;	// whether to process current interaction
__shared__ bool			sameFLG;	// whether to the current and neighbour are the same cell

__shared__ unsigned int	shrdAddr[1];	// shared cell's address
__shared__ unsigned int shrdOff[1];	// shared cell's offsets
__shared__ unsigned int shrdOcc[1];	// shared cell's occupancy
__shared__ float 		shrArr[MAX_INC_FIELDS*BLOCK];	// inclusion coordinates
__shared__ device_data_struct_t<float>	shrdCell;	// shared memory structure to load cell

// initial setup:
	float	probe_raw[MAX_INC_FIELDS];	// inclusion object stored in register memory
	if (threadIdx.x == 0)	{
			selfID		= blockIdx.x + core_splitIdx * core_splitSize;
			selfFLG		= get_dmnIndex(selfID,core_bufLayer);

			shrdAddr[0]	= blockDim.x;
			shrdOff[0]	= 0;
			shrdOcc[0]	= 0;

			shrdCell.num_Fields	= MAX_INC_FIELDS;
			shrdCell.num_Cells	= 1;
			shrdCell.field_Off	= blockDim.x;
			shrdCell.cell_Addr	= &shrdAddr[0];
			shrdCell.cell_Off	= &shrdOff[0];
			shrdCell.cell_Occ	= &shrdOcc[0];
			shrdCell.cell_Data	= &shrArr[0];

	}
	__threadfence_block();
	__syncthreads();

	if ( selfID < dataSelf.num_Cells ) {
		// get inclusions from current cell:
		shrdCell.copy_cell( 0, selfID, &dataSelf );

		if (threadIdx.x == 0) selfINC = shrdCell.cell_Occ[0];
		shrdCell.get_item(0,threadIdx.x,probe_raw);
		__threadfence_block();
		__syncthreads();

		// swipe over neighbouring cells:
		for(int iter = 0; iter<core_numIter; iter++) {
			if (threadIdx.x ==0 ) {	// define neighbouring cell id, periodic displacement and need to process interactions:
				stride  = get_stride(iter);
				neigID  = stride2glbID_directed_uint(core_peri_dir,stride, selfID, core_grdExt);
				neigFLG = get_dmnIndex(neigID,core_bufLayer);

				procFLG = (neigID<dataNeig.num_Cells);
				if (procFLG) {
					procFLG = procFLG && (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (core_order_lim+1)*(core_order_lim+1));
					procFLG = procFLG && ( ( (selfFLG>=0)||(neigFLG>=0) )||( (selfFLG<0)&&(selfFLG!=neigFLG) ) );	// to disregard contacts within electrode (FLG<0, core_peri_dir!=7, core_bufLayer>0)
					procFLG = procFLG && ( (!core_upper_tri)||(neigID>=selfID) );
				}

				sameFLG = ( (dataSelf.cell_Addr == dataNeig.cell_Addr) && (selfID == neigID) );

				if (procFLG) {
					char3 shift	= stride2dspl(stride,selfID, core_grdExt);
					neigDspl = make_float3(	shift.x*core_phsDim[0],
											shift.y*core_phsDim[1],
											shift.z*core_phsDim[2]);

					neigINC = dataNeig.cell_Occ[neigID] - dataNeig.cell_Off[neigID];
					neigSTPs = (short) ceil((float)neigINC/blockDim.x);
				}
			}
			__threadfence_block();
			__syncthreads();

			if (procFLG) { // process interactions:
				// load other inclusions data into shared memory in chunks - if cell or array is different:
				for(int frac = 0; frac<neigSTPs; frac++) {
					if (threadIdx.x==0) {
						neigLOAD = ( (frac+1)*blockDim.x <= neigINC ? blockDim.x : neigINC-frac*blockDim.x );
						neigOFF = dataNeig.cell_Off[neigID] + frac*blockDim.x;
					}
					__threadfence_block();
					__syncthreads();

					// load neighbour cell inclusions into shared memory:
					shrdCell.clean_cell(0);
					shrdCell.copy_cell( 0, neigID, neigOFF, neigLOAD, &dataNeig);
					tarnsCNT(neigLOAD, neigDspl, shrArr);

					//*
					cell_interaction(	sameFLG,
										&inter_prms,
										selfID,
										selfINC,
										probe_raw,
										neigID,
										neigOFF,
										neigLOAD,
										&shrdCell,
										&dataSelf,
										&dataNeig,
										&dataRes	);
										//*/
					__threadfence_block();
					__syncthreads();
				} // end of chunk processing for
				shrdCell.clean_cell(0);

				if ((core_reWrt_self)&&(procFLG)) {// sort and rewrite current cell in global memory:
				//if ((true)&&(procFLG)) {// sort and rewrite current cell in global memory:
					// Option A - atomic operations:
					// WATCHOUT - does not preserves order - may be dangerous for multi-segment inclusions!!!
					shrdCell.add_item(0,selfINC,probe_raw);
					dataSelf.clean_cell(selfID);
					dataSelf.copy_cell(selfID,0,&shrdCell);
					__syncthreads();
					// TODO: Option B - shared memory sort
				}
			} // end of processing if
		} // end of for-loop */
	} // end of cell is in the grid if
}
// Core kernel wrappers:---------------------------------------------------------------------------
// Set up device constant memory:
bool		setup_core(const core_const_Prms_t& constPrms) {
	cudaError_t cuErr = cudaSuccess;
	cuErr = cudaMemcpyToSymbol(inter_prms,		&constPrms.inter_prms,		sizeof(struct interact_prms_t),	0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: inter_prms\n");
	cudaMemcpyToSymbol(core_splitSize,	&constPrms.core_splitSize,	sizeof(int),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_splitSize\n");
	cudaMemcpyToSymbol(core_peri_dir,	&constPrms.core_peri_dir,	sizeof(int),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_peri_dir\n");
	cudaMemcpyToSymbol(core_grdExt,	    &constPrms.core_grdExt,		sizeof(int)*3,					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_grdExt\n");
	cudaMemcpyToSymbol(core_phsDim,	    &constPrms.core_phsDim,		sizeof(float)*3,				0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_phsDim\n");
	cudaMemcpyToSymbol(core_bufLayer,	&constPrms.core_bufLayer, 	sizeof(int),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_bufLayer\n");
	cudaMemcpyToSymbol(core_order_lim,	&constPrms.core_order_lim,	sizeof(int),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_order_lim\n");
	cudaMemcpyToSymbol(core_upper_tri,	&constPrms.core_upper_tri,	sizeof(bool),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_upper_tri\n");
	cudaMemcpyToSymbol(core_numIter,	&constPrms.core_numIter,	sizeof(unsigned int),			0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_numIter\n");
	cudaMemcpyToSymbol(core_reWrt_self,	&constPrms.core_reWrt_self,	sizeof(bool),					0,cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess) printf("Error while setting const parameter: core_reWrt_self\n");
	return (cuErr==cudaSuccess);
}

void set_core_splitIdx(int count) {
	cudaMemcpyToSymbol(core_splitIdx,&count, sizeof(int), 0,cudaMemcpyHostToDevice);
}

device_data_struct_t<float> get_selfDataStruct(	simPrms* Simul,
												thrust::device_vector<unsigned int>&	d_realAddr,
												thrust::device_vector<unsigned int>&	d_realOff,
												thrust::device_vector<unsigned int>&	d_realOcc	) {
	// convert arrays: (TODO: later replace initial arrays in generation cycle wit arrays of unsigned ints)
	thrust::copy(	thrust::device_pointer_cast(Simul->Domains.d_dmnOcc),
					thrust::device_pointer_cast(Simul->Domains.d_dmnOcc)+Simul->Domains.grdSize,
					d_realOcc.begin() );
	thrust::copy(	thrust::device_pointer_cast(Simul->Domains.d_dmnAddr),
					thrust::device_pointer_cast(Simul->Domains.d_dmnAddr)+Simul->Domains.grdSize,
					d_realAddr.begin() );

	device_data_struct_t<float>	dataSelf;
								dataSelf.num_Cells	= Simul->Domains.grdSize;
								dataSelf.num_Fields = MAX_INC_FIELDS;
								dataSelf.field_Off	= Simul->Domains.ttlCNT;
								dataSelf.cell_Addr	= thrust::raw_pointer_cast(d_realAddr.data());
								dataSelf.cell_Occ	= thrust::raw_pointer_cast(d_realOcc.data());
								dataSelf.cell_Off	= thrust::raw_pointer_cast(d_realOff.data());
								dataSelf.cell_Data	= Simul->d_result;

	return dataSelf;
}

template<class T>
device_data_struct_t<T> get_DataStruct(	thrust::device_vector<unsigned int>&	d_Addr,
										thrust::device_vector<unsigned int>&	d_Off,
										thrust::device_vector<unsigned int>&	d_Occ,
										thrust::device_vector<T>&	d_data	)
{
	device_data_struct_t<T>		dataStruct;
								dataStruct.num_Cells	= d_Addr.size();
								dataStruct.field_Off	= d_Addr[d_Addr.size()-1];
								dataStruct.num_Fields 	= d_data.size()/dataStruct.field_Off;
								dataStruct.cell_Addr	= thrust::raw_pointer_cast(d_Addr.data());
								dataStruct.cell_Off		= thrust::raw_pointer_cast(d_Off.data());
								dataStruct.cell_Occ		= thrust::raw_pointer_cast(d_Occ.data());
								dataStruct.cell_Data	= thrust::raw_pointer_cast(d_data.data());

	return dataStruct;
}

// Template instantiation:
template __global__ void core_cell_interact_kernel<char>(	device_data_struct_t<float>	dataSelf,
															device_data_struct_t<float>	dataNeig,
															device_data_struct_t<char>	dataRes		);

template __global__ void core_cell_interact_kernel<float>(	device_data_struct_t<float>	dataSelf,
															device_data_struct_t<float>	dataNeig,
															device_data_struct_t<float>	dataRes		);

template __global__ void core_cell_interact_kernel<unsigned int>(	device_data_struct_t<float>	dataSelf,
															device_data_struct_t<float>	dataNeig,
															device_data_struct_t<unsigned int>	dataRes		);

template device_data_struct_t<float> get_DataStruct(	thrust::device_vector<unsigned int>&	d_Addr,
														thrust::device_vector<unsigned int>&	d_Off,
														thrust::device_vector<unsigned int>&	d_Occ,
														thrust::device_vector<float>&	d_data	);

template device_data_struct_t<char> get_DataStruct(		thrust::device_vector<unsigned int>&	d_Addr,
														thrust::device_vector<unsigned int>&	d_Off,
														thrust::device_vector<unsigned int>&	d_Occ,
														thrust::device_vector<char>&	d_data	);

