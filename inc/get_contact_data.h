// This file declares functions to calculate characteristics of contacts between inclusions
// provided array of coordinates, simulation parameters, arrays of indices of inclusions in
// contact and position of contact along inclusion axii

template<typename T>
struct decrement{
	__host__ __device__ T operator()(const T& A) { return (A>0) ? A-1 : 0; }
};
template<typename T>
struct unary_not_equal_to{
	T val;
	unary_not_equal_to(const T& _v) : val(_v) {}
	__host__ __device__ T operator()(const T& A) { return (A!=val); }
};

// get contact separation distances:----------------------------------------------------------
void get_contact_distance(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>& 	d_contact_J,
							thrust::device_vector<float>& 			d_contact_tI,
							thrust::device_vector<float>& 			d_contact_tJ,
							thrust::device_vector<float>* 			d_contact_D	);
// get contact surface areas:----------------------------------------------------------
void get_contact_surface(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_contact_I,
							thrust::device_vector<unsigned int>& 	d_contact_J,
							thrust::device_vector<float>& 			d_contact_tI,
							thrust::device_vector<float>& 			d_contact_tJ,
							thrust::device_vector<float>* 			d_contact_S	);
// get distances between contacts:----------------------------------------------------------
void get_segment_length(	simPrms* const 							Simul,
							thrust::device_vector<unsigned int>&	d_segment_I,
							thrust::device_vector<float>& 			d_segment_T,
							thrust::device_vector<unsigned int>*	d_diff_pos,
							thrust::device_vector<unsigned int>*	d_diff_cnts,
							thrust::device_vector<float>* 			d_segment_L,
							thrust::device_vector<float>* 			d_segment_A		);

