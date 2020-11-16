// Local declarations:--------------------------------------------------------------------
#include "types.h"

template<typename Ti,typename Tf>
struct is_conductor {
	__host__ __device__ bool operator()(const thrust::tuple<Ti,Ti,Ti,Ti,Tf,Tf>& A) {
		return (thrust::get<0>(A)<SNK_ELEC) && (thrust::get<1>(A)<SNK_ELEC);
	}
};

template<typename Ti,typename Tf>
struct is_electrode_2nd {
	__host__ __device__ bool operator()(const thrust::tuple<Ti,Ti,Ti,Ti,Tf,Tf>& A) {
		return (thrust::get<1>(A)>=SNK_ELEC) && (thrust::get<1>(A)>thrust::get<0>(A));
	}
};

template<typename Ti>
struct isnot_valid {
	__host__ __device__ bool operator()(const thrust::tuple<Ti,Ti>& A) {
		return	( thrust::get<1>(A)==SNK_ELEC ) ||
				( thrust::get<1>(A)==SRC_ELEC ) ||
				( thrust::get<1>(A)==VOID_ELEC );
	}
};

template<typename Ti>
struct copy_2nd {
	__host__ __device__ thrust::tuple<Ti,Ti> operator()(const thrust::tuple<Ti,Ti>& A) {
		return thrust::make_tuple(	thrust::get<1>(A),thrust::get<1>(A));
	}
};

template<typename Ti,typename Tf>
struct switch_tuple {
	__host__ __device__ thrust::tuple<Ti,Ti,Ti,Ti,Tf,Tf> operator()(const thrust::tuple<Ti,Ti,Ti,Ti,Tf,Tf>& A) {
		return thrust::make_tuple(	thrust::get<1>(A),thrust::get<0>(A),
									thrust::get<3>(A),thrust::get<2>(A),
									thrust::get<5>(A),thrust::get<4>(A)	);
	}
};

template<typename Ti,typename Tf>
struct is_elec_int {
	__host__ __device__ bool operator()(const thrust::tuple<Ti,Ti,Tf,Tf>& A) {
		// True if both belong to either Sink or Source electrode: (SRC_ELEC>SNK_ELEC)
		return (thrust::get<0>(A)>=SNK_ELEC) && (thrust::get<1>(A)>=SNK_ELEC);
	}
};

template<typename Ti,typename Tf>
struct isnot_elec_int {
	__host__ __device__ bool operator()(const thrust::tuple<Ti,Ti,Ti,Ti,Tf,Tf>& A) {
		// True if both belong to either Sink or Source electrode: (SRC_ELEC>SNK_ELEC)
		return (thrust::get<0>(A)<SNK_ELEC) || (thrust::get<1>(A)<SNK_ELEC);
	}
};

template<typename Ti,typename Tf>
struct get_inc_dmnID {
	Tf*	f_dmn_ptr;
	Ti	offset;
	get_inc_dmnID(Tf* _ptr,Ti _off):	f_dmn_ptr(_ptr), offset(_off) {}

	__host__ __device__ Ti operator() (const Ti& incID) {
		return (Ti)f_dmn_ptr[incID+offset];
	}

};

template<typename Tu>
struct get_elec_dmnID {
	int						bLay;
	get_elec_dmnID(Tu _bL) : bLay(_bL) {}
	__device__ Tu operator()(const Tu& selfID) {

		int pos = 0;

		if ( (peri_dir&1)==0 ) { pos = (selfID%grdExt_cond[0])	- bLay; }
		if ( (peri_dir&2)==0 ) { pos = (selfID/grdExt_cond[0])%grdExt_cond[1]	- bLay; }
		if ( (peri_dir&4)==0 ) { pos = (selfID/grdExt_cond[0])/grdExt_cond[1]	- bLay; }

		Tu dmnID = selfID;	// internal
		if (pos < 0) { dmnID = SRC_ELEC; } // source
		if (pos >= (grdExt_cond[((7-peri_dir)>>1)]-2*bLay) ) { dmnID = SNK_ELEC; } // sink

		return dmnID;

	}
};


template<typename T>
struct is_negative {
	__host__ __device__ bool operator()(const T& a) {
		return (a<0);
	}
};

template<typename T>
struct equal_to_val {
	T val;
	equal_to_val(T _v) : val(_v) {}
	__host__ __device__ bool operator()(const T& a) {return (a==val);}
};

template<typename T>
struct less_than_val {
	T val;
	less_than_val(T _v) : val(_v) {}
	__host__ __device__ bool operator()(const T& a) {return (a<val);}
};

// Auxiliary declarations:--------------------------------------------------------------------

// count contacts:
unsigned int get_contact_counts(	simPrms* const 							Simul,
									int										dir,
									thrust::device_vector<unsigned int>*	d_contact_counts);
unsigned int get_contact_counts(	simPrms* const 							Simul,
									dim3									grid,
									dim3									block,
									thrust::device_vector<unsigned int>*	d_contact_counts);
// store contact data:
void get_contact_arrays(	simPrms* const 							Simul,
							int				dir,
							thrust::device_vector<unsigned int>&	d_contact_counts,
							thrust::device_vector<unsigned int>*	d_contact_I,
							thrust::device_vector<unsigned int>* 	d_contact_J,
							thrust::device_vector<float>* 			d_contact_D,
							thrust::device_vector<float>* 			d_contact_S	);
// check whether contacts are obstructed:
void check_screened_contacts(	unsigned int	num_contacts,
								simPrms* const 	Simul,
								int				dir,
								dim3			grid,
								dim3			block,
								thrust::device_vector<unsigned int>&	d_contact_I,
								thrust::device_vector<unsigned int>&	d_contact_J,
								thrust::device_vector<char>*			isnt_screened	);
// prepare contact arrays before output:
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
									unsigned int* num_src_conts	);
// arranges a table ( Node ID - internal/electrode flag - inclusion ID - local coordinate t):
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
									thrust::device_vector<unsigned short>*	d_node_edgeLR	);
// calculate internal conductance of segments of inclusions:
void get_internal_conduct(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_node_ID,
							thrust::device_vector<unsigned int>& 	d_node_I,
							thrust::device_vector<float>& 			d_node_t,
							thrust::device_vector<unsigned int>* 	d_segment_nI,
							thrust::device_vector<unsigned int>* 	d_segment_nJ,
							thrust::device_vector<float>* 			d_segment_L,
							thrust::device_vector<float>* 			d_segment_A	);
// write adjacency matrix to files
void save_coo(	const thrust::device_vector<unsigned int>&	d_contact_I,
				const thrust::device_vector<unsigned int>&	d_contact_J,
				const thrust::device_vector<float>&			d_contact_D,
				const thrust::device_vector<float>&			d_contact_S,
				unsigned int num_int_conts,
				unsigned int num_snk_conts,
				unsigned int num_src_conts	);

void save_coo(	const char*											fname,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_begin,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_end,
				const thrust::device_vector<unsigned int>::iterator	d_node_J_begin,
				const thrust::device_vector<float>::iterator		d_data_IJ_begin	);

void save_coo(	const char*											fname,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_begin,
				const thrust::device_vector<unsigned int>::iterator	d_node_I_end,
				const thrust::device_vector<unsigned int>::iterator	d_node_J_begin,
				const thrust::device_vector<float>::iterator		d_data1_IJ_begin,
				const thrust::device_vector<float>::iterator		d_data2_IJ_begin	);

// Virtual inclusions representing contacts:-------------------------------------------------
// Generate virtual inclusions:
void	generate_virt_Incs(	unsigned int							num_contacts,
							const thrust::device_ptr<float>&		d_real_inc,
							dim3									grid,
							dim3 									block,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>&	d_contact_J,
							thrust::device_vector<unsigned int>*	d_virtAddr,
							thrust::device_vector<unsigned int>*	d_virtOcc,
							thrust::device_vector<float>*			d_virt_inc	);

void	check_screened_Interactions(	thrust::device_vector<char>*			isnt_screened,
										thrust::device_vector<unsigned int>&	d_virtAddr,
										thrust::device_vector<unsigned int>&	d_virtOcc,
										thrust::device_vector<float>&			d_virt_inc,
										simPrms* const Simul,
										int dir );
