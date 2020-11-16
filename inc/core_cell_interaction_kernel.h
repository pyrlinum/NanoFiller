#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include "simPrms.h"
#include "definitions.h"
#include "vectorMath.h"

struct interact_prms_t {
	char	process_f;
	float	separation;
	float	soft_frac;
};

template<class T>
class device_data_struct_t {

public:

	unsigned int	num_Fields;
	unsigned int	num_Cells;
	unsigned int	field_Off;
	unsigned int*	cell_Addr;
	unsigned int*	cell_Off;
	unsigned int*	cell_Occ;
	T*		cell_Data;

	__host__ __device__ inline unsigned int	get_cell_start(unsigned int I) {
			return ( I>0 ? this->cell_Addr[I-1] : 0);
	}


	__host__ __device__ inline unsigned int	get_cell_end(unsigned int I) {
			return this->cell_Addr[I];
	}


	__host__ __device__ inline unsigned int	get_cell_size(unsigned int I) {
			return this->get_cell_end(I) - this->get_cell_start(I);
	}

	__device__ inline void	copy_cell(	unsigned int	I,
										unsigned int	J,
										unsigned int	patchStart,
										unsigned int	patchSize,
										device_data_struct_t* srcArr)	{
			__shared__ unsigned int begDST, avlSize, begSRC, cpySize;

			if (threadIdx.x==0) { // set up write addresses:
				begDST	=   this->get_cell_start(I) +   this->cell_Occ[I];
				avlSize =	this->get_cell_size(I)	-	this->cell_Occ[I];

				begSRC 	= srcArr->get_cell_start(J) + patchStart;
				cpySize = ( patchStart + patchSize <= srcArr->cell_Occ[J] ?
									patchSize : srcArr->cell_Occ[J] - patchStart );
			}
			__threadfence_block();
			__syncthreads();

			if ( cpySize <= avlSize ) { // protects from over writing next cell - in theory should not occur
				if (threadIdx.x < cpySize)
					for(unsigned int i = 0; i < srcArr->num_Fields; i++)	{
						  this->cell_Data[begDST + threadIdx.x + i*  this->field_Off] =
						srcArr->cell_Data[begSRC + threadIdx.x + i*srcArr->field_Off];
					}
				//__threadfence_block();
				//__syncthreads();
				if (threadIdx.x == 0) // update occupancy:
					this->cell_Occ[I] += cpySize;
			}
			__threadfence_block();
			__syncthreads();
	}

	__device__ inline void	copy_cell(	unsigned int	I,
										unsigned int	J,
										device_data_struct_t* srcArr)	{
			this->copy_cell(I,J,srcArr->cell_Off[J],srcArr->cell_Occ[J]-srcArr->cell_Off[J],srcArr);
	}

	__host__ __device__ inline void	get_item(	unsigned int	I, unsigned int itemID, T* item_raw) {
			bool legalF = (itemID < this->cell_Occ[I]-this->cell_Off[I]);
			for(unsigned int f = 0; f< this->num_Fields; f++)
				item_raw[f] = (legalF ? this->cell_Data[this->get_cell_start(I)+this->cell_Off[I]+itemID+f*this->field_Off] : DEF_EMPTY_VAL);
	}

	__host__ __device__ inline void	put_item(	unsigned int	I, unsigned int itemID, T* item_raw) {
			if (itemID < this->cell_Occ[I]-this->cell_Off[I])	{
				for(unsigned int f = 0; f< this->num_Fields; f++)
					this->cell_Data[this->get_cell_start(I)+this->cell_Off[I]+itemID+f*this->field_Off] = item_raw[f];
			}
	}

	__device__ inline void	add_item(	unsigned int I, unsigned int numINC, T* item_raw ) {
			__shared__ unsigned int nItems;
			if (threadIdx.x==0) nItems = this->cell_Occ[I];
			__threadfence_block();
			__syncthreads();
			if (threadIdx.x<numINC)	{
				bool write_flg = (abs(item_raw[this->num_Fields-1]-DEF_EMPTY_VAL)>DEV_PRECISION);
				if (write_flg) {
					unsigned int itemID0 = atomicAdd(&nItems,1);
					if (itemID0 < this->get_cell_size(I))	{
						for(unsigned int f = 0; f< this->num_Fields; f++)
							this->cell_Data[this->get_cell_start(I)+this->cell_Off[I] + itemID0 + f*this->field_Off] = item_raw[f];
					} else printf("ERROR: attempting to put item %i beyond cell %i size %i!!!\n",itemID0,I,this->get_cell_size(I));
				}
			}
			__threadfence_block();
			__syncthreads();
			if (threadIdx.x==0) this->cell_Occ[I] = nItems;
			__threadfence_block();
			__syncthreads();
	}

	__device__ inline void	clean_cell(	unsigned int	I) {
			unsigned int begDST =   this->get_cell_start(I) +   this->cell_Off[I];
			unsigned int occSize = this->cell_Occ[I]-this->cell_Off[I];
			if (threadIdx.x < occSize)
				for(unsigned int i = 0; i < this->num_Fields; i++)
					  this->cell_Data[begDST + threadIdx.x + i*this->field_Off] = DEF_EMPTY_VAL;
			__threadfence_block();
			__syncthreads();
			if (threadIdx.x == 0) this->cell_Occ[I] = this->cell_Off[I];
			__threadfence_block();
			__syncthreads();
	}



};

template<class T1,class T2>
void copy_structure(const device_data_struct_t<T1>& A, device_data_struct_t<T2>* B) {
	B->num_Fields	=	A.num_Fields;
	B->num_Cells	=	A.num_Cells;
	B->field_Off	=	A.field_Off;
	B->cell_Addr	=	A.cell_Addr;
	B->cell_Off		=	A.cell_Off;
	B->cell_Occ		=	A.cell_Occ;
}




device_data_struct_t<float> get_selfDataStruct(	simPrms* Simul,
												thrust::device_vector<unsigned int>&	d_realAddr,
												thrust::device_vector<unsigned int>&	d_realOff,
												thrust::device_vector<unsigned int>&	d_realOcc	);

template<class T>
device_data_struct_t<T> get_DataStruct(	thrust::device_vector<unsigned int>&	d_Addr,
										thrust::device_vector<unsigned int>&	d_Off,
										thrust::device_vector<unsigned int>&	d_Occ,
										thrust::device_vector<T>&	d_data	);

// Constant memory setup:------------------------------------------------------------------
struct core_const_Prms_t {
		interact_prms_t	inter_prms;					// parameters of interaction calculation
	 	int				core_splitIdx;				// Id of share of grid, split if too large
	 	int				core_splitSize;				// Size of share of grid, split if too large
	 	int 			core_peri_dir;				// periodic flags (7=0b0111 - periodic in XYZ)
	 	int				core_grdExt[3];				// extents of domain cells' grid
	 	float			core_phsDim[3];				// physical dimensions of sample
	 	int				core_bufLayer;				// Buffer layer thickness;
	 	int				core_order_lim;				// maximum order of neighbour cell to check;
	 	bool			core_upper_tri;				// whether the processing is upper triangular order or full
	 	unsigned int	core_numIter;				// number of iteration to be used to process neighbour cell interactions
	 	bool			core_reWrt_self;			// flag whether current cell needs to be rewritten
};

bool		setup_core(const core_const_Prms_t& constPrms);
void 		set_core_splitIdx(int count);

inline core_const_Prms_t setup_collect_Ncnct(simPrms* Simul, int dir) {
	core_const_Prms_t constPrms;
	constPrms.inter_prms.process_f	= CONTACT_COUNT;
	constPrms.inter_prms.separation = Simul->stat_dist_lim;
	constPrms.inter_prms.soft_frac	= 1.0;
	memcpy(&constPrms.core_grdExt,&Simul->Domains.ext,	sizeof(int)*3	);
	memcpy(&constPrms.core_phsDim,&Simul->physDim,		sizeof(float)*3	);
	constPrms.core_splitSize		= Simul->kernelSize;
	constPrms.core_peri_dir			= 7-dir;
	constPrms.core_bufLayer			= Simul->NeiOrder;
	constPrms.core_order_lim		= Simul->NeiOrder+1;
	constPrms.core_numIter			= (2*constPrms.core_order_lim+1)*(2*constPrms.core_order_lim+1)*(2*constPrms.core_order_lim+1);
	constPrms.core_reWrt_self		= false;
	//constPrms.core_upper_tri		= false;
	constPrms.core_upper_tri		= true; // for testing purposes - in reality need to be replaced by full
	return constPrms;
}

inline core_const_Prms_t setup_collect_IJt(simPrms* Simul, int dir) {
	core_const_Prms_t constPrms;
	constPrms.inter_prms.process_f	= CONTACT_RECORD;
	constPrms.inter_prms.separation = Simul->stat_dist_lim;
	constPrms.inter_prms.soft_frac	= 1.0;
	memcpy(&constPrms.core_grdExt,&Simul->Domains.ext,	sizeof(int)*3	);
	memcpy(&constPrms.core_phsDim,&Simul->physDim,		sizeof(float)*3	);
	constPrms.core_splitSize		= Simul->kernelSize;
	constPrms.core_peri_dir			= 7-dir;
	constPrms.core_bufLayer			= Simul->NeiOrder;
	constPrms.core_order_lim		= Simul->NeiOrder+1;
	constPrms.core_numIter			= (2*constPrms.core_order_lim+1)*(2*constPrms.core_order_lim+1)*(2*constPrms.core_order_lim+1);
	constPrms.core_reWrt_self		= false;
	//constPrms.core_upper_tri		= false;
	constPrms.core_upper_tri		= true; // for testing purposes - in reality need to be replaced by full
	return constPrms;
}

inline core_const_Prms_t setup_real2virt(simPrms* Simul, int dir) {
	core_const_Prms_t constPrms;
	constPrms.inter_prms.process_f	= VIRTUAL_CHECK;
	constPrms.inter_prms.separation = 0.0f;
	constPrms.inter_prms.soft_frac	= 1.0f;
	memcpy(&constPrms.core_grdExt,&Simul->Domains.ext,	sizeof(int)*3	);
	memcpy(&constPrms.core_phsDim,&Simul->physDim,		sizeof(float)*3	);
	constPrms.core_splitSize		= Simul->kernelSize;
	constPrms.core_peri_dir			= 7-dir;
	constPrms.core_bufLayer			= Simul->NeiOrder;
	constPrms.core_order_lim		= Simul->NeiOrder;
	constPrms.core_numIter			= (2*Simul->NeiOrder+1)*(2*Simul->NeiOrder+1)*(2*Simul->NeiOrder+1);
	constPrms.core_reWrt_self		= false;
	constPrms.core_upper_tri		= false;
	return constPrms;
}

// Template instantiation:--------------------------------------------------------------------
template<class Tr>
__global__ void core_cell_interact_kernel(	device_data_struct_t<float>	dataSelf,
											device_data_struct_t<float>	dataNeig,
											device_data_struct_t<Tr>	dataRes		);
