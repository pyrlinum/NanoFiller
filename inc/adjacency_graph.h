#pragma once

unsigned int	remove_redundant_nodes(		float L_crit,
											thrust::device_vector<unsigned int>*	d_segment_nJ,
											thrust::device_vector<float>*			d_segment_L,
											thrust::device_vector<unsigned int>*	d_newnode_map	);

void	remap_nodes(	thrust::device_vector<unsigned int>::iterator	d_nID_begin,
						thrust::device_vector<unsigned int>::iterator	d_nID_end,
						thrust::device_vector<unsigned int>::iterator	d_newnode_map	);

unsigned int	reduce_node_tbl(	thrust::device_vector<unsigned int>*	d_node_ID,
									thrust::device_vector<unsigned int>*	d_node_I,
									thrust::device_vector<float>*			d_node_t	);

unsigned int	recalculate_segments(	thrust::device_vector<unsigned int>&	d_node_ID,
										thrust::device_vector<float>&			d_node_t,
										thrust::device_vector<unsigned int>*	d_segment_nI,
										thrust::device_vector<unsigned int>*	d_segment_nJ,
										thrust::device_vector<float>*			d_segment_L,
										thrust::device_vector<float>*			d_segment_A	);
