//====================================================================================================================
//										 <<< Kernels to collect statistical data: >>>
//====================================================================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include "simPrms.h"
#include "IO.h"
#include "array_kern.h"
#include "core_interfaces.h"
#include "get_contact_data.h"
#include "adjacency_graph.h"

//CUDA constants:
__constant__ char	peri_dir;					// field direction
__constant__ int	glbPtchSz_cond;				// offset of CNT data alignment
__constant__ int	grdExt_size;				// extents of domain cells' grid
__constant__ int	grdExt_cond[3];				// extents of domain cells' grid
__constant__ float	phsDim_cond[3];				// physical dimensions of sample
__constant__ float	range_lim;					// maximum distance interaction distance;
__constant__ float	electrode_crd[2];			// coordinates of electrode plates along the preferred axis
__constant__ float	cond_precision;				// precision for float operations

#include "conductKernels.h"
#include "conductivity.h"


// Conductance calculation:--------------------------------------------------------------------
int2 init_contacts_map(simPrms *Simul, char efield_dir) {

	// define kernel grid dimensions:
	int		currentDev;		 cudaGetDevice(&currentDev);
	cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
	int block = devProp.maxThreadsPerBlock;
	int grid  = Simul->Domains.ext[0]*Simul->Domains.ext[1]*Simul->Domains.ext[2];
	char dir = 7 - efield_dir;
	float	elec_lim[2];
			elec_lim[0] = 0;
			elec_lim[1] = Simul->physDim[efield_dir>>1]*(1.0-2*Simul->NeiOrder*1.0/Simul->Domains.ext[efield_dir>>1]);
	printf("Electrode positions used in calculation: %f - %f\n",elec_lim[0],elec_lim[1]);

	cudaMemcpyToSymbol(peri_dir,	    &dir,	  				  sizeof(char), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz_cond,&(Simul->Domains.ttlCNT),	  sizeof(int),  0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt_size,	   &grid,					  sizeof(int),	0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt_cond,		Simul->Domains.ext,		3*sizeof(int),	0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(phsDim_cond,		Simul->physDim,			3*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(electrode_crd,	elec_lim,				2*sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(range_lim,	  &(Simul->stat_dist_lim),	  sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cond_precision,&(Simul->TOL),			  sizeof(float),0,cudaMemcpyHostToDevice);

	if (cudaGetLastError() != 0) {printf("ERROR: setting constants failed");}

	return make_int2(grid,block);
}

int reduce(	simPrms	*Simul, int2 cond_grid, int min_cont )
{
	//printf("Current number of CNTs %d: \n",Simul->get_numCNT());
	cudaError cuErr;
	dim3	 grid(cond_grid.x);
	dim3	block(cond_grid.y);
	int	statData_size = (int) ceil((float)Simul->Domains.ttlCNT/block.x);

	// Initialise arrays
	unsigned int	*d_contact_counts = cuda_init<unsigned int>(0, statData_size*block.x);

	cuErr = cudaGetLastError();
	if (cuErr == cudaSuccess)	{

		cudaEvent_t start,stop;
		float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		// Counting inter-inclusion contacts - full version:
		int rangeOrd = Simul->NeiOrder;
		int zl = -rangeOrd; int zh = rangeOrd;
		int yl = -rangeOrd; int yh = rangeOrd;
		int xl = -rangeOrd; int xh = rangeOrd;
		char3 stride = make_char3(0,0,0);

		for(stride.z=zl;stride.z<=zh;stride.z++)
			for(stride.y=yl;stride.y<=yh;stride.y++)
				for(stride.x=xl;stride.x<=xh;stride.x++)
					if (stride.x*stride.x+stride.y*stride.y+stride.z*stride.z <= (rangeOrd+1)*(rangeOrd+1))
						contact_count_kernel_full<<<grid,block>>>(	0,
																	stride,
																	Simul->Domains.d_dmnAddr,
																	Simul->d_result,
																	d_contact_counts	);

		reductKernel<<<grid,block>>>(	0,
										min_cont,
										Simul->NeiOrder,
										Simul->Domains.d_dmnAddr,
										Simul->Domains.d_dmnOcc,
										Simul->d_result,
										d_contact_counts	);

		cuErr = cudaGetLastError();
		if (cuErr!=cudaSuccess) printf("ERROR WHILE REDUCING CNTs: %s !!!\n",cudaGetErrorString(cuErr));
		printf("Inclusions remained: %d \n",Simul->get_numCNT());

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );

	}

	cudaFree(d_contact_counts);
	return (cuErr==cudaSuccess);
}

int collect_cunduct_data(int dir, int internal_flag, simPrms *Simul) {
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// Step 0: Set memory
	int2 cond_grid = init_contacts_map(Simul, dir);
	dim3 grid(cond_grid.x);
	dim3 block(cond_grid.y);
	int	statData_size = (int) ceil((float)Simul->Domains.ttlCNT/block.x);
	Simul->set_k2dmnIdx(); // make sure that k-field of inclusions stores its cell index

	// Step 1: Count number of contacts:
	thrust::device_vector<unsigned int> d_contact_counts(Simul->Domains.ttlCNT,0);
	unsigned int	num_contacts = get_contact_counts(	Simul,
														dir,
														&d_contact_counts	);

	// Step 2: Collect indices of sparse adjacency matrix {A(i,j)} and contact characteristics (D,S):
	thrust::device_vector<unsigned int> d_contact_I(num_contacts,2*Simul->numCNT);
	thrust::device_vector<unsigned int> d_contact_J(num_contacts,2*Simul->numCNT);
	thrust::device_vector<float> 		d_contact_tI(num_contacts,-1.0);
	thrust::device_vector<float> 		d_contact_tJ(num_contacts,-1.0);

	get_contact_arrays(	Simul,
						dir,
						d_contact_counts,
						&d_contact_I,
						&d_contact_J,
						&d_contact_tI,
						&d_contact_tJ	);

	// Step 3: eliminate screened contacts:
	thrust::device_vector<char>	isnt_screened(num_contacts,VIRT_INC_PRESENT);
	check_screened_contacts(	num_contacts,
								Simul,
								dir,
								grid,
								block,
								d_contact_I,
								d_contact_J,
								&isnt_screened);

	// Step 4: clean screened elements and arrange array into internal, sink and source parts:
	unsigned int num_int_conts = 0;
	unsigned int num_snk_conts = 0;
	unsigned int num_src_conts = 0;
	sort_contacts_by_role(	Simul,
							dir,
							num_contacts,
							grid,
							block,
							isnt_screened,
							&d_contact_I,
							&d_contact_J,
							&d_contact_tI,
							&d_contact_tJ,
							&num_int_conts,
							&num_snk_conts,
							&num_src_conts );

	// Step 5A: calculate distances for non-screened contacts:
	thrust::device_vector<float> 		d_contact_D(num_contacts,-1.0);
	get_contact_distance(	Simul,
							d_contact_I,
							d_contact_J,
							d_contact_tI,
							d_contact_tJ,
							&d_contact_D	);

	// Step 5B: calculate surfaces for non-screened contacts:
	thrust::device_vector<float> 		d_contact_S(num_contacts,-1.0);
	get_contact_surface(	Simul,
							d_contact_I,
							d_contact_J,
							d_contact_tI,
							d_contact_tJ,
							&d_contact_S	);

	if ( !internal_flag )	{
	// Save matrix of contacts:
		cout << "Ignoring internal resistances...\n";
		save_coo( d_contact_I, d_contact_J, d_contact_D, d_contact_S, num_int_conts,num_snk_conts,num_src_conts);
	} else {
		// If internal resistance should be included:
		cout << "Estimating internal resistances...\n";

		// Step 6: Create a table for indexing of nodes - nID: (I,tI) and node - internal/electrode flag relation:
		unsigned int	num_contacts_full = 2*num_int_conts + num_snk_conts + num_src_conts;
		thrust::device_vector<unsigned int>	d_node_ID(num_contacts_full,0);
		thrust::device_vector<unsigned int>	d_node_I(num_contacts_full,0);
		thrust::device_vector<float> 		d_node_t(num_contacts_full,0.0);
		thrust::device_vector<unsigned int>	d_node_edgeID(num_contacts_full,0);
		thrust::device_vector<unsigned short>		d_node_edgeLR(num_contacts_full,0);
		thrust::device_vector<unsigned int>	d_contact_nI(d_contact_I.size(),0);
		thrust::device_vector<unsigned int>	d_contact_nJ(d_contact_I.size(),0);

		unsigned int num_nodes = get_node_relations(	num_int_conts,
														num_snk_conts,
														num_src_conts,
														d_contact_I,
														d_contact_J,
														d_contact_tI,
														d_contact_tJ,
														&d_node_ID,
														&d_node_I,
														&d_node_t,
														&d_contact_nI,
														&d_contact_nJ,
														&d_node_edgeID,
														&d_node_edgeLR	);


		// Step 7: calculate segment lengths along the inclusion and get rid of segments with 0 length:
		thrust::device_vector<unsigned int>	d_segment_nI(d_node_ID);
		thrust::device_vector<unsigned int>	d_segment_nJ(d_node_ID);
		thrust::device_vector<float>		d_segment_L(d_node_ID.size(),-1.0);
		thrust::device_vector<float>		d_segment_A(d_node_ID.size(),-1.0);
		get_internal_conduct(	Simul,
								d_node_ID,
								d_node_I,
								d_node_t,
								&d_segment_nI,
								&d_segment_nJ,
								&d_segment_L,
								&d_segment_A	);

		// Step 8: sort by node_ID
		uint2ZipIter	nIJ_begin	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_contact_nI.begin(),
																d_contact_nJ.begin()	));
		uint2ZipIter	nIJ_end		= thrust::make_zip_iterator(
											thrust::make_tuple(	d_contact_nI.end(),
																d_contact_nJ.end()		));
		flt2ZipIter		nDS_begin	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_contact_D.begin(),
																d_contact_S.begin()	));
		thrust::sort_by_key(nIJ_begin,nIJ_end,nDS_begin);

		// Step 9: optimize vertices:
		save_coo(	"Nodes.tbl.ini",
					d_node_ID.begin(),
					d_node_ID.end(),
					d_node_I.begin(),
					d_node_t.begin()	);

		save_coo(	"segment.LA.ini",
					d_segment_nI.begin(),
					d_segment_nI.end(),
					d_segment_nJ.begin(),
					d_segment_L.begin(),
					d_segment_A.begin()	);

		save_coo(	"internal.DS.ini",
					d_contact_nI.begin(),
					d_contact_nI.begin()+num_int_conts,
					d_contact_nJ.begin(),
					d_contact_D.begin(),
					d_contact_S.begin()	);

		save_coo(	"sink.DS.ini",
					d_contact_nI.begin()+num_int_conts,
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nJ.begin()+num_int_conts,
					d_contact_D.begin() +num_int_conts,
					d_contact_S.begin() +num_int_conts	);

		save_coo(	"source.DS.ini",
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nI.end(),
					d_contact_nJ.begin()+num_int_conts+num_snk_conts,
					d_contact_D.begin() +num_int_conts+num_snk_conts,
					d_contact_S.begin() +num_int_conts+num_snk_conts	);

		cout << "Eliminating redundant nodes...\n";
		unsigned int redundant_node_count = remove_redundant_nodes(	1,
																	&d_segment_nJ,
																	&d_segment_L,
																	&d_node_ID	);

		cout << "Redundant nodes count: " << redundant_node_count << "\n";

		// remap nodes to a new indices:
		remap_nodes(d_contact_nI.begin(), d_contact_nI.begin()+num_int_conts, d_node_ID.begin());
		remap_nodes(d_contact_nJ.begin(), d_contact_nJ.end(), d_node_ID.begin());
		remap_nodes(d_segment_nI.begin(), d_segment_nI.end(), d_node_ID.begin());
		remap_nodes(d_segment_nJ.begin(), d_segment_nJ.end(), d_node_ID.begin());

		save_coo(	"Nodes.tbl.rmp",
					d_node_ID.begin(),
					d_node_ID.end(),
					d_node_I.begin(),
					d_node_t.begin()	);

		save_coo(	"segment.LA.rmp",
					d_segment_nI.begin(),
					d_segment_nI.end(),
					d_segment_nJ.begin(),
					d_segment_L.begin(),
					d_segment_A.begin()	);

		save_coo(	"internal.DS.rmp",
					d_contact_nI.begin(),
					d_contact_nI.begin()+num_int_conts,
					d_contact_nJ.begin(),
					d_contact_D.begin(),
					d_contact_S.begin()	);

		save_coo(	"sink.DS.rmp",
					d_contact_nI.begin()+num_int_conts,
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nJ.begin()+num_int_conts,
					d_contact_D.begin() +num_int_conts,
					d_contact_S.begin() +num_int_conts	);

		save_coo(	"source.DS.rmp",
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nI.end(),
					d_contact_nJ.begin()+num_int_conts+num_snk_conts,
					d_contact_D.begin() +num_int_conts+num_snk_conts,
					d_contact_S.begin() +num_int_conts+num_snk_conts	);

		// reduce node map:
		unsigned int	reduced_node_count = reduce_node_tbl(&d_node_ID,	&d_node_I, &d_node_t	);
		cout << "Reduced nodes count: " << reduced_node_count << "\n";

		// recalculate length of segments and erase redundant ones:
		unsigned int	reduced_segment_count = recalculate_segments(	d_node_ID,
																		d_node_t,
																		&d_segment_nI,
																		&d_segment_nJ,
																		&d_segment_L,
																		&d_segment_A);
		cout << "Reduced segments count: " << reduced_segment_count << "\n";

		// Step 10:save matrices save connectivity data in COO sparse-matrix format:
		//*
		// TEST:
		save_coo(	"Nodes.tbl.dat",
					d_node_ID.begin(),
					d_node_ID.end(),
					d_node_I.begin(),
					d_node_t.begin()	);
		//*/

		save_coo(	"segment.LA.coo",
					d_segment_nI.begin(),
					d_segment_nI.end(),
					d_segment_nJ.begin(),
					d_segment_L.begin(),
					d_segment_A.begin()	);

		save_coo(	"internal.DS.coo",
					d_contact_nI.begin(),
					d_contact_nI.begin()+num_int_conts,
					d_contact_nJ.begin(),
					d_contact_D.begin(),
					d_contact_S.begin()	);

		save_coo(	"sink.DS.coo",
					d_contact_nI.begin()+num_int_conts,
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nJ.begin()+num_int_conts,
					d_contact_D.begin() +num_int_conts,
					d_contact_S.begin() +num_int_conts	);

		save_coo(	"source.DS.coo",
					d_contact_nI.begin()+num_int_conts+num_snk_conts,
					d_contact_nI.end(),
					d_contact_nJ.begin()+num_int_conts+num_snk_conts,
					d_contact_D.begin() +num_int_conts+num_snk_conts,
					d_contact_S.begin() +num_int_conts+num_snk_conts	);
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "Total Time: %f ms \n",time );
	return (cuErr==cudaSuccess);
}

// Auxiliary functions:--------------------------------------------------------------------
// Check whether contacts are not obstructed by other inclusions:
// for each internal contact create a corresponding line segment and store it as a virtual inclusion
// calculate intersections between segments representing contact and real inclusions
// and mark virtual inclusions to delete:
void check_screened_contacts(	unsigned int	num_contacts,
								simPrms* const 	Simul,
								int				dir,
								dim3			grid,
								dim3			block,
								thrust::device_vector<unsigned int>&	d_contact_I,
								thrust::device_vector<unsigned int>&	d_contact_J,
								thrust::device_vector<char>*			isnt_screened	)
{

	thrust::device_vector<float>		d_virt_inc(num_contacts*MAX_INC_FIELDS,-1.0);
	thrust::device_vector<unsigned int>	d_virtAddr(grid.x,0);
	thrust::device_vector<unsigned int>	d_virtOcc(grid.x,0);
	thrust::device_ptr<float>	d_real_inc = thrust::device_pointer_cast(Simul->d_result);
	generate_virt_Incs(	num_contacts,
						d_real_inc,
						grid,
						block,
						d_contact_I,
						d_contact_J,
						&d_virtAddr,
						&d_virtOcc,
						&d_virt_inc	);

	check_screened_Interactions(	isnt_screened,
									d_virtAddr,
									d_virtOcc,
									d_virt_inc,
									Simul,
									dir );
}

// generate array of virtual inclusions, representing inter-inclusion's contacts:
void	generate_virt_Incs(	unsigned int							num_contacts,
							const thrust::device_ptr<float>&		d_real_inc,
							dim3									grid,
							dim3 									block,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>&	d_contact_J,
							thrust::device_vector<unsigned int>*	d_virtAddr,
							thrust::device_vector<unsigned int>*	d_virtOcc,
							thrust::device_vector<float>*			d_virt_inc	)
{

	cudaError_t cuErr;
	int	aux_grid_size = (int) ceil((float)num_contacts/block.x);
	dim3 grid_aux(aux_grid_size);

	generate_virt_inclusions<<<grid_aux,block>>>(	num_contacts,
													thrust::raw_pointer_cast(d_contact_I.data()),
													thrust::raw_pointer_cast(d_contact_J.data()),
													thrust::raw_pointer_cast(d_virtOcc->data()),
													thrust::raw_pointer_cast(d_virt_inc->data()),
													thrust::raw_pointer_cast(d_real_inc) );
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess) printf("Virtual inclusions generation: %s\n",cudaGetErrorString(cuErr));

	thrust::device_vector<unsigned int>		d_cnctIJ_idx(num_contacts,0);
	thrust::sequence(d_cnctIJ_idx.begin(),d_cnctIJ_idx.end(),0);

	thrust::device_vector<float>::iterator	d_kIter_beg(d_virt_inc->begin()+8*num_contacts);
	thrust::device_vector<float>::iterator	d_kIter_end(d_virt_inc->end());
	thrust::device_vector<float>			d_kVec_cpy(d_kIter_beg,d_kIter_end);

	int missed_count = thrust::count_if(d_kIter_beg,d_kIter_end,is_negative<float>());
	thrust::inclusive_scan(d_virtOcc->begin(),d_virtOcc->end(),d_virtAddr->begin());
	unsigned int num_occ = d_virtAddr->data()[grid.x-1];
	if (missed_count>0) printf("Number of unassigned virtual inclusions: %i/%i/%i\n",missed_count,num_occ,num_contacts);

	// sort d_virt_inc by k-field (grid cell index) and compare partial sums with expected occupancies:
	flt8ZipIter	virtInc_Iter = thrust::make_zip_iterator(thrust::make_tuple(	d_virt_inc->begin()+0*num_contacts,
																				d_virt_inc->begin()+1*num_contacts,
																				d_virt_inc->begin()+2*num_contacts,
																				d_virt_inc->begin()+3*num_contacts,
																				d_virt_inc->begin()+4*num_contacts,
																				d_virt_inc->begin()+5*num_contacts,
																				d_virt_inc->begin()+6*num_contacts,
																				d_virt_inc->begin()+7*num_contacts	));

	flt8uInt_zIter virtInc_tIter =	thrust::make_zip_iterator(thrust::make_tuple(	virtInc_Iter, d_cnctIJ_idx.begin()) );
	thrust::sort_by_key(d_kIter_beg,d_kIter_end,virtInc_tIter);

	// verify sorted structure:
	thrust::device_vector<unsigned int>		d_testOcc(grid.x,0);
	thrust::device_vector<unsigned int>		d_testErr(grid.x,0);
	count_incInCell<<<grid,block>>>(	0,
										num_contacts,
										thrust::raw_pointer_cast(d_virt_inc->data()+8*num_contacts),
										thrust::raw_pointer_cast(d_virtAddr->data()),
										thrust::raw_pointer_cast(d_testOcc.data()),
										thrust::raw_pointer_cast(d_testErr.data())	);

	int sum_virt_occ = thrust::reduce(d_testOcc.begin(),d_testOcc.end(),(int) 0);
	int sum_virt_err = thrust::reduce(d_testErr.begin(),d_testErr.end(),(int) 0);
	if (sum_virt_occ != num_contacts )
		printf("WARNING: Number of virtual inclusions correctly assigned to cells differs from total number of contacts: %i/%i\n",sum_virt_occ,num_contacts);
	if (sum_virt_err != 0 )
		printf("WARNING: Cell addresses of %i virtual inclusions differ from expected!!!\n",sum_virt_err);
}

unsigned int sort_contacts_by_role(	simPrms* const 							Simul,
									int										dir,
									unsigned int							num_contacts,
									dim3									grid,
									dim3									block,
									const thrust::device_vector<char>&		isnt_screened,
									thrust::device_vector<unsigned int>*	d_contact_I,
									thrust::device_vector<unsigned int>*	d_contact_J,
									thrust::device_vector<float>*			d_contact_tI,
									thrust::device_vector<float>*			d_contact_tJ,
									unsigned int* num_int_conts,
									unsigned int* num_snk_conts,
									unsigned int* num_src_conts	)
{

	thrust::device_vector<unsigned int>	d_incFlg_I(d_contact_I->size(),0);
	thrust::device_vector<unsigned int>	d_incFlg_J(d_contact_J->size(),0);

	// get inclusions cell index:
	thrust::transform(	d_contact_I->begin(),
						d_contact_I->end(),
						d_incFlg_I.begin(),
						get_inc_dmnID<unsigned int,float>(Simul->d_result,8*Simul->Domains.ttlCNT)
						);

	thrust::transform(	d_contact_J->begin(),
						d_contact_J->end(),
						d_incFlg_J.begin(),
						get_inc_dmnID<unsigned int,float>(Simul->d_result,8*Simul->Domains.ttlCNT)
						);
	// mark electrode cells:
	thrust::transform(	d_incFlg_I.begin(),
						d_incFlg_I.end(),
						d_incFlg_I.begin(),
						get_elec_dmnID<unsigned int>( Simul->NeiOrder)
						);

	thrust::transform(	d_incFlg_J.begin(),
						d_incFlg_J.end(),
						d_incFlg_J.begin(),
						get_elec_dmnID<unsigned int>( Simul->NeiOrder)
						);

	// set both I&J of contacts, marked to delete, to ELEC value to be disregarded later while partitioning:
	thrust::replace_if(d_incFlg_I.begin(),d_incFlg_I.end(),isnt_screened.begin(),equal_to_val<char>(VIRT_INC_EMPTY),VOID_ELEC);
	thrust::replace_if(d_incFlg_J.begin(),d_incFlg_J.end(),isnt_screened.begin(),equal_to_val<char>(VIRT_INC_EMPTY),VOID_ELEC);

	// create a masked array with inclusion indices only for valid contact points:
	thrust::device_vector<unsigned int> d_masked_I(*d_contact_I);
	thrust::device_vector<unsigned int> d_masked_J(*d_contact_J);

	uint2ZipIter	maskI_first	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_masked_I.begin(),
																d_incFlg_I.begin()	));
	uint2ZipIter	maskJ_first	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_masked_J.begin(),
																d_incFlg_J.begin()	));
	uint2ZipIter	maskI_last	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_masked_I.end(),
																d_incFlg_I.end()	));
	uint2ZipIter	maskJ_last	= thrust::make_zip_iterator(
											thrust::make_tuple(	d_masked_J.end(),
																d_incFlg_J.end()	));
	thrust::transform_if(maskI_first, maskI_last, maskI_first, copy_2nd<unsigned int>(), isnot_valid<unsigned int>()	);
	thrust::transform_if(maskJ_first, maskJ_last, maskJ_first, copy_2nd<unsigned int>(), isnot_valid<unsigned int>()	);

	// sort contact arrays by masked arrays:
	ui4f2ZipIter	first_ijIJ	= thrust::make_zip_iterator(
										thrust::make_tuple(	d_masked_I.begin(),
															d_masked_J.begin(),
															d_contact_I->begin(),
															d_contact_J->begin(),
															d_contact_tI->begin(),
															d_contact_tJ->begin()	));
	ui4f2ZipIter	last_ijIJ	= thrust::make_zip_iterator(
										thrust::make_tuple(	d_masked_I.end(),
															d_masked_J.end(),
															d_contact_I->end(),
															d_contact_J->end(),
															d_contact_tI->end(),
															d_contact_tJ->end()	));

	// swap tuple order for "electrode" inclusions
	thrust::transform_if(	first_ijIJ, last_ijIJ, first_ijIJ, switch_tuple<unsigned int,float>(), is_electrode_2nd<unsigned int,float>() );

	// sort to bring to conductor - sink - source order
	thrust::sort( first_ijIJ, last_ijIJ );

	// move invalid and electrode-electrode contacts to the end
	// and get the position of the last valid element
	ui4f2ZipIter	new_last_ijIJ = thrust::stable_partition(	first_ijIJ,	last_ijIJ,	isnot_elec_int<unsigned int,float>()	);
	unsigned int	new_size = thrust::distance(first_ijIJ,new_last_ijIJ);

	// Final step: save data in CSR3 format:
	*num_int_conts = thrust::count_if(first_ijIJ,first_ijIJ+new_size,is_conductor<unsigned int,float>());
	*num_src_conts = thrust::count(d_masked_I.begin(),d_masked_I.begin()+new_size,SRC_ELEC);
	*num_snk_conts = thrust::count(d_masked_I.begin(),d_masked_I.begin()+new_size,SNK_ELEC);
	printf("Internal contacts estimated:\t%i / %i \n",*num_int_conts,new_size);
	printf("Source contacts estimated:\t%i / %i \n",*num_src_conts,new_size);
	printf("Sink contacts estimated:\t%i / %i \n",*num_snk_conts,new_size);

	d_contact_I->resize(new_size);
	d_contact_J->resize(new_size);
	d_contact_tI->resize(new_size);
	d_contact_tJ->resize(new_size);

	return d_contact_I->size();
}

unsigned int get_node_relations(	unsigned int							num_int2int,
									unsigned int							num_snk2int,
									unsigned int							num_src2int,
									thrust::device_vector<unsigned int>&	d_contact_I,
									thrust::device_vector<unsigned int>& 	d_contact_J,
									thrust::device_vector<float>& 			d_contact_tI,
									thrust::device_vector<float>& 			d_contact_tJ,
									thrust::device_vector<unsigned int>*	d_node_ID,
									thrust::device_vector<unsigned int>*	d_node_I,
									thrust::device_vector<float>*			d_node_t,
									thrust::device_vector<unsigned int>*	d_contact_nI,
									thrust::device_vector<unsigned int>* 	d_contact_nJ,
									thrust::device_vector<unsigned int>*	d_node_edgeID,
									thrust::device_vector<unsigned short>*	d_node_edgeLR	)
{
	printf("Building node-contact relation...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// Form vector of nodes excluding electrode inclusions in the first column:
	//unsigned int	num_contacts_full = 2*num_int2int + num_snk2int + num_src2int;
	thrust::sequence(d_node_ID->begin(), d_node_ID->end(), 0);

	// copy upper triangular arrays to full matrix:
	thrust::device_vector<unsigned int>::iterator tmpI_iter =
			thrust::copy(d_contact_I.begin(), d_contact_I.begin()+num_int2int, d_node_I->begin());
	thrust::copy(d_contact_J.begin(), d_contact_J.end(), tmpI_iter);

	thrust::device_vector<float>::iterator tmpTI_iter=
					thrust::copy(d_contact_tI.begin(), d_contact_tI.begin()+num_int2int, d_node_t->begin());
	thrust::copy(d_contact_tJ.begin(),d_contact_tJ.end(),tmpTI_iter);


	// vectors describing the relation between the nodes and the corresponding edge
	// and the node's position (left(0) - begin, right(1) - end)
	// in the upper triangular adjacency matrix of contacts:
	thrust::sequence(	d_node_edgeID->begin(),
						d_node_edgeID->begin()+num_int2int, 0);
	thrust::fill(		d_node_edgeLR->begin(),
						d_node_edgeLR->begin()+num_int2int, 0);
	thrust::sequence(	d_node_edgeID->begin()+num_int2int,
						d_node_edgeID->end(), 0);
	thrust::fill(		d_node_edgeLR->begin()+num_int2int,
						d_node_edgeLR->end(), 1);

	// sort by first index and position (I,tI):
	uifZipIter segIT_beg = thrust::make_zip_iterator(
								thrust::make_tuple(	d_node_I->begin(),
													d_node_t->begin()	));
	uifZipIter segIT_end = thrust::make_zip_iterator(
								thrust::make_tuple(	d_node_I->end(),
													d_node_t->end()	));

	// dependent data - edge index and position (L/R):
	uishZipIter edge_data_beg = thrust::make_zip_iterator(
									thrust::make_tuple(	d_node_edgeID->begin(),
														d_node_edgeLR->begin()	));

	// node_edge... - has length of d_contact_I, while node_I is num_int2int larger!!!
	thrust::sort_by_key(segIT_beg, segIT_end, edge_data_beg);

	// Inverse relation - replace pairs (I,tI) in edges; description with corresponding nodes:
	// sorting key - pair of left-right position and edge index:
	thrust::device_vector<unsigned short>d_edge_nLR(d_node_edgeLR->begin(),d_node_edgeLR->end());
	thrust::device_vector<unsigned int> d_edge_idx(d_node_edgeID->begin(),d_node_edgeID->end());

	ushiZipIter edge_key_beg = thrust::make_zip_iterator(
										thrust::make_tuple(	d_edge_nLR.begin(),
															d_edge_idx.begin()	));
	ushiZipIter edge_key_end = thrust::make_zip_iterator(
										thrust::make_tuple(	d_edge_nLR.end(),
															d_edge_idx.end()	));

	// dependent data - node index:
	thrust::device_vector<unsigned int> d_edge_nodeID(d_node_ID->begin(),d_node_ID->end());

	thrust::sort_by_key(edge_key_beg, edge_key_end, d_edge_nodeID.begin());

	// assign left nodes to edges:
	thrust::copy(	d_edge_nodeID.begin(),
					d_edge_nodeID.begin()+num_int2int,
					d_contact_nI->begin()	);

	thrust::fill(	d_contact_nI->begin()+num_int2int,
					d_contact_nI->begin()+num_int2int+num_snk2int,
					SNK_ELEC	);

	thrust::fill(	d_contact_nI->begin()+num_int2int+num_snk2int,
					d_contact_nI->end(),
					SRC_ELEC	);

	// assign right nodes to edges:
	thrust::copy(	d_edge_nodeID.begin()+num_int2int,
					d_edge_nodeID.end(),
					d_contact_nJ->begin()	);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA time for building node-edge relation table: %f ms \n",time );

	return d_node_ID->size();

}

// get distances between contacts:----------------------------------------------------------
void get_internal_conduct(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_node_ID,
							thrust::device_vector<unsigned int>& 	d_node_I,
							thrust::device_vector<float>& 			d_node_t,
							thrust::device_vector<unsigned int>* 	d_segment_nI,
							thrust::device_vector<unsigned int>* 	d_segment_nJ,
							thrust::device_vector<float>* 			d_segment_L,
							thrust::device_vector<float>* 			d_segment_A	)
{

	printf("Calculating segment length...\n");
	cudaError_t cuErr = cudaGetLastError();
	cudaEvent_t start,stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// Calculate the differences between the points along the same inclusion:
	// get the starting positions of subsets of points belonging to the same inclusion:
	thrust::constant_iterator<unsigned int>		ones_itr(1);
	thrust::device_vector<unsigned int>		d_uniq_cnts(d_node_ID.size(),0);
	thrust::device_vector<unsigned int>		d_uniq_keys(d_node_ID.size(),0);
	thrust::pair<thrust::discard_iterator<thrust::use_default>,uintIter>	uniq_end =
			thrust::reduce_by_key(	d_node_I.begin(),
									d_node_I.end(),
									ones_itr,
									thrust::make_discard_iterator(),
									d_uniq_cnts.begin()	);
	unsigned int uniq_size = thrust::distance(d_uniq_cnts.begin(),uniq_end.second);
	d_uniq_cnts.resize(uniq_size);
	thrust::device_vector<unsigned int> d_uniq_pos(uniq_size,0);
	thrust::exclusive_scan(d_uniq_cnts.begin(),d_uniq_cnts.end(),d_uniq_pos.begin());

	// submit vectors to inclusion-type dependent function to determine length:
	thrust::device_vector<unsigned int>	d_diff_pos(d_uniq_pos);
	thrust::device_vector<unsigned int>	d_diff_cnts(d_uniq_cnts);
	get_segment_length(	Simul,
						d_node_I,
						d_node_t,
						&d_diff_pos,
						&d_diff_cnts,
						d_segment_L,
						d_segment_A	);

	// construct start/end node indices for segments:
	unsigned int	num_segments = d_node_ID.size() - uniq_size;
	thrust::device_vector<unsigned int>	d_voids_u(uniq_size,VOID_ELEC);
	thrust::scatter(	d_voids_u.begin(),
						d_voids_u.end(),
						d_uniq_pos.begin(),
						d_segment_nJ->begin()	);
	thrust::stable_partition(	d_segment_nJ->begin(), d_segment_nJ->end(),
								unary_not_equal_to<unsigned int>(VOID_ELEC)	);
	d_segment_nJ->resize(num_segments);

	thrust::transform(	d_uniq_pos.begin(),
						d_uniq_pos.end(),
						d_uniq_cnts.begin(),
						d_uniq_pos.begin(),
						thrust::plus<unsigned int>());
	thrust::transform(	d_uniq_pos.begin(),
						d_uniq_pos.end(),
						d_uniq_pos.begin(),
						decrement<unsigned int>());

	thrust::scatter(	d_voids_u.begin(),
						d_voids_u.end(),
						d_uniq_pos.begin(),
						d_segment_nI->begin()	);
	thrust::stable_partition(	d_segment_nI->begin(), d_segment_nI->end(),
								unary_not_equal_to<unsigned int>(VOID_ELEC)	);
	d_segment_nI->resize(num_segments);

	//C. merge points with dt<a (L(=dt/a)<1):
	// short-cut: replace with 1:
	thrust::replace_if(	d_segment_L->begin(),
						d_segment_L->end(),
						less_than_val<float>(1.0f), 1.0f	);


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	printf( "TIMER: CUDA time for estimating internal lengths: %f ms \n",time );
}

void save_coo(	const thrust::device_vector<unsigned int>&	d_contact_I,
				const thrust::device_vector<unsigned int>&	d_contact_J,
				const thrust::device_vector<float>&			d_contact_D,
				const thrust::device_vector<float>&			d_contact_S,
				unsigned int num_int_conts,
				unsigned int num_snk_conts,
				unsigned int num_src_conts	)
{

	thrust::host_vector<unsigned int>	h_vecI(d_contact_I);
	thrust::host_vector<unsigned int>	h_vecJ(d_contact_J);
	thrust::host_vector<float> 			h_vecD(d_contact_D);
	thrust::host_vector<float>			h_vecS(d_contact_S);

	std::ofstream ofileI("internal.srt.coo");
	ofileI << "#\tI\tJ\tD\tS\n";
	unsigned int startI = 0;
	unsigned int endI	= startI + num_int_conts;
    for(int i=startI; i<endI; i++)   {
    	 ofileI << "\t" << h_vecI[i] << "\t" << h_vecJ[i] << "\t" <<h_vecD[i] << "\t" <<h_vecS[i] << "\n";
    }
    ofileI.close();

    std::ofstream ofileK("sink.elec.coo");
    ofileK << "#\tI\tJ\tD\tS\n";
    unsigned int startK = endI;
    unsigned int endK	= startK + num_snk_conts;
    for(int i=startK; i<endK; i++)   {
      	 ofileK << "\t" << h_vecI[i] << "\t" << h_vecJ[i] << "\t" <<h_vecD[i] << "\t" <<h_vecS[i] << "\n";
    }
    ofileK.close();

    std::ofstream ofileS("source.elec.coo");
    ofileS << "#\tI\tJ\tD\tS\n";
    unsigned int startS = endK;
    unsigned int endS	= startS + num_src_conts;
    for(int i=startS; i<endS; i++)   {
       	 ofileS << "\t" << h_vecI[i] << "\t" << h_vecJ[i] << "\t" <<h_vecD[i] << "\t" <<h_vecS[i] << "\n";
    }
    ofileS.close();

}

void save_coo(	const char*											fname,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_begin,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_end,
				const thrust::device_vector<unsigned int>::iterator	d_node_J_begin,
				const thrust::device_vector<float>::iterator		d_data_IJ_begin	)
{

	unsigned int num_edges = thrust::distance(d_node_I_begin,d_node_I_end);

	thrust::host_vector<unsigned int>	h_node_I(	d_node_I_begin,	d_node_I_end				);
	thrust::host_vector<unsigned int>	h_node_J(	d_node_J_begin,	d_node_J_begin +num_edges	);
	thrust::host_vector<float>	h_data_IJ(	d_data_IJ_begin,d_data_IJ_begin+num_edges);

	std::ofstream ofileI(fname);
	ofileI << "#\tI\tJ\tData\n";
    for(unsigned int i=0; i<num_edges; i++)   {
    	 ofileI << "\t" << h_node_I[i]
    	        << "\t" << h_node_J[i]
    	        << "\t" << h_data_IJ[i] << "\n";
    }
    ofileI.close();

}

void save_coo(	const char*											fname,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_begin,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_end,
				const thrust::device_vector<unsigned int>::iterator	d_node_J_begin,
				const thrust::device_vector<float>::iterator		d_data1_IJ_begin,
				const thrust::device_vector<float>::iterator		d_data2_IJ_begin	)
{

	unsigned int num_edges = thrust::distance(d_node_I_begin,d_node_I_end);

	thrust::host_vector<unsigned int>	h_node_I(	d_node_I_begin,	d_node_I_end				);
	thrust::host_vector<unsigned int>	h_node_J(	d_node_J_begin,	d_node_J_begin +num_edges	);
	thrust::host_vector<float>	h_data1_IJ(	d_data1_IJ_begin,d_data1_IJ_begin+num_edges);
	thrust::host_vector<float>	h_data2_IJ(	d_data2_IJ_begin,d_data2_IJ_begin+num_edges);

	std::ofstream ofileI(fname);
	ofileI << "#\tI\tJ\tData1\tData2\n";
    for(unsigned int i=0; i<num_edges; i++)   {
    	 ofileI << "\t" << h_node_I[i]
    	        << "\t" << h_node_J[i]
    	        << "\t" << h_data1_IJ[i]
    	 	 	<< "\t" << h_data2_IJ[i] << "\n";
    }
    ofileI.close();

}
