//====================================================================================================================
//			 <<< Functions to construct and optimise the adjacency matrix graph:>>>
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
#include <thrust/gather.h>
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
#include "definitions.h"
#include "types.h"
//----------------------------------------------------------------------
// A. Replace indices of the nodes to be eliminated with the indices
// of the preceding remaining nodes in the copy of the nodes list
// returns the number of redundant nodes:
// auxiliary functor:
template<typename T>
struct mark_to_delete {
	T* data;
	mark_to_delete(T* _data) : data(_data) {}
	__host__ __device__ T operator()(const T& idx ) {
		data[idx] = (T) 1;
		return data[idx];
	}
};

template<typename T>
struct le_than_val {
	T a;
	le_than_val(T _a) : a(_a) {}
	__host__ __device__ bool operator()(const T& A ) {
		return (A<=a);
	}
};
// core function:
unsigned int	remove_redundant_nodes(		float L_crit,
											thrust::device_vector<unsigned int>*	d_segment_nJ,
											thrust::device_vector<float>*			d_segment_L,
											thrust::device_vector<unsigned int>*	d_newnode_map	)
{

	// Mark the nodes to be deleted:
	thrust::device_vector<unsigned int>	d_rdd_node_map(d_newnode_map->size(),0);
	// if (d_segment_L[I] < L_crit) then d_rdd_node_map[d_segment_nJ[I]] = 1:
	//thrust::make_discard_iterator(),
	//mark_to_delete<unsigned int>(d_rdd_node_map.data()),
	thrust::transform_if(	d_segment_nJ->begin(), d_segment_nJ->end(),
							d_segment_L->begin(),
							thrust::make_discard_iterator(),
							mark_to_delete<unsigned int>(thrust::raw_pointer_cast(d_rdd_node_map.data())),
							le_than_val<float>(L_crit)	);

	// get the number of nodes to be deleted:
	unsigned int	num_rdd_nodes = thrust::reduce(d_rdd_node_map.begin(), d_rdd_node_map.end());

	// Make the map relating the old nodes' indices to the new:
	// new_ID[I] = old_ID[I]-accumulate(d_rdd_node_map)[I]
	thrust::inclusive_scan(	d_rdd_node_map.begin(), d_rdd_node_map.end(),
							d_rdd_node_map.begin()	);
	thrust::transform(		d_newnode_map->begin(),	d_newnode_map->end(),
							d_rdd_node_map.begin(),
							d_newnode_map->begin(),
							thrust::minus<unsigned int>());

	return num_rdd_nodes;

};
//----------------------------------------------------------------------
// B. Remap indices of the segments/contacts end-nodes:
// core function - actually simple wrapper over thrust::gather
void	remap_nodes(	thrust::device_vector<unsigned int>::iterator	d_nID_begin,
						thrust::device_vector<unsigned int>::iterator	d_nID_end,
						thrust::device_vector<unsigned int>::iterator	d_newnode_map_begin	) {
	thrust::gather(	d_nID_begin, 	d_nID_end,
					d_newnode_map_begin,
					d_nID_begin 	);
};
//----------------------------------------------------------------------
// C. Reduce the table of nodes:
// auxiliary functor
template<typename Ta,typename Tb,typename Tc>
struct average_repetition {
	__host__ __device__ thrust::tuple<Ta,Tb,Tc> operator()( const thrust::tuple<Ta,Tb,Tc>& A, const thrust::tuple<Ta,Tb,Tc>& B ) {
		return	thrust::make_tuple(	thrust::get<0>(A),
									thrust::get<1>(A)+thrust::get<1>(B),
									thrust::get<2>(A)+thrust::get<2>(B)	);

	}

};
// core function:
unsigned int	reduce_node_tbl(	thrust::device_vector<unsigned int>*	d_node_ID,
									thrust::device_vector<unsigned int>*	d_node_I,
									thrust::device_vector<float>*			d_node_t	) {

	thrust::device_vector<float>	repetition_count(d_node_ID->size(),1.0f);
	uif2ZipIter	nodeIrt_iter = thrust::make_zip_iterator(
										thrust::make_tuple(	d_node_I->begin(),
															repetition_count.begin(),
															d_node_t->begin() ));

	thrust::device_vector<unsigned int>	d_reduced_nID(d_node_ID->size(),0);
	thrust::device_vector<unsigned int>	d_reduced_I(d_node_I->size(),0);
	thrust::device_vector<float>		d_reduced_r(repetition_count.size(),0.0f);
	thrust::device_vector<float>		d_reduced_t(d_node_t->size(), EMPTY_SCALE);

	uif2ZipIter	redIrt_iter = thrust::make_zip_iterator(
										thrust::make_tuple(	d_reduced_I.begin(),
															d_reduced_r.begin(),
															d_reduced_t.begin() ));


	thrust::pair<uintIter,uif2ZipIter>
		new_end	= thrust::reduce_by_key(	d_node_ID->begin(), d_node_ID->end(),
											nodeIrt_iter,
											d_reduced_nID.begin(),
											redIrt_iter,
											thrust::equal_to<unsigned int>(),
											average_repetition<unsigned int,float,float>());

	unsigned int	reduced_node_count = thrust::distance(d_reduced_nID.begin(), new_end.first);
	thrust::copy(	d_reduced_nID.begin(),	new_end.first, d_node_ID->begin());
	thrust::copy(	d_reduced_I.begin(),	d_reduced_I.begin()+reduced_node_count,
					d_node_I->begin()	);
	thrust::transform(	d_reduced_t.begin(),d_reduced_t.begin()+reduced_node_count,
						d_reduced_r.begin(),
						d_node_t->begin(),
						thrust::divides<float>());

	d_node_ID->resize(reduced_node_count);
	d_node_I->resize(reduced_node_count);
	d_node_t->resize(reduced_node_count);

	return reduced_node_count;
};
//----------------------------------------------------------------------
// D. Reduce the table of nodes:
// auxiliary functor
template<typename Ti,typename Tf>
struct is_symmetric {
	__host__ __device__ bool operator()( const thrust::tuple<Ti,Ti,Tf,Tf>& A ) {
		return	(thrust::get<0>(A)==thrust::get<1>(A));
	}
};
// core function:
unsigned int	recalculate_segments(	thrust::device_vector<unsigned int>&	d_node_ID,
										thrust::device_vector<float>&			d_node_t,
										thrust::device_vector<unsigned int>*	d_segment_nI,
										thrust::device_vector<unsigned int>*	d_segment_nJ,
										thrust::device_vector<float>*			d_segment_L,
										thrust::device_vector<float>*			d_segment_A	) {

	// clear out redundant segments (for which nI=nJ after re-mapping):
	ui2f2ZipIter	key_bgn =	thrust::make_zip_iterator(
									thrust::make_tuple(	d_segment_nI->begin(),
														d_segment_nJ->begin(),
														d_segment_L->begin(),
														d_segment_A->begin()	));

	ui2f2ZipIter	key_end =	thrust::make_zip_iterator(
									thrust::make_tuple(	d_segment_nI->end(),
														d_segment_nJ->end(),
														d_segment_L->end(),
														d_segment_A->end()	));

	ui2f2ZipIter	new_end =	thrust::remove_if(	key_bgn, key_end,
													is_symmetric<unsigned int,float>() );
	unsigned int reduced_seg_count = thrust::distance(key_bgn,new_end);
	d_segment_nI->resize(reduced_seg_count);
	d_segment_nJ->resize(reduced_seg_count);
	d_segment_L->resize(reduced_seg_count);
	d_segment_A->resize(reduced_seg_count);

	// for the remaining - recalculate the length of the segments:
	thrust::device_vector<float>	d_seg_tI(d_segment_nI->size(),0.0f);
	thrust::gather(	d_segment_nI->begin(),d_segment_nI->end(),
					d_node_t.begin(),
					d_seg_tI.begin()	);

	thrust::device_vector<float>	d_seg_tJ(d_segment_nJ->size(),0.0f);
	thrust::gather(	d_segment_nJ->begin(),d_segment_nJ->end(),
						d_node_t.begin(),
						d_seg_tJ.begin()	);

	thrust::transform(	d_seg_tJ.begin(), d_seg_tJ.end(),
						d_seg_tI.begin(),
						d_segment_L->begin(),
						thrust::minus<float>()	);

	//*
	thrust::transform(	d_segment_L->begin(), d_segment_L->end(),
						d_segment_A->begin(),
						d_segment_L->begin(),
						thrust::divides<float>()	); //*/

	return d_segment_L->size();

};
