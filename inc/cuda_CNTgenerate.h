//====================================================================================================================
//										 <<< GENERATION KERNEL >>>
//--------------------------------------------------------------------------------------------------------------------
#pragma once
#include <curand_kernel.h>
#include "memoryIO.h"

//__constant__	float Phi_avg;
//__constant__	float Phi_dev;
//__constant__	float Theta_avg;
//__constant__	float Theta_dev;
//__constant__	float prefDir[2];


__device__ float sym_truncNormal(float a, float s, float w, curandState	*state);
__device__ float3 inc_rotation(float3 vec, float Mu, float Xi);
__device__ float3 genCosine(float phi_avg, float phi_dev,float theta_avg, float theta_dev, curandState *state);

// Generate CNT - Switchable kernel:-------------------------------------------------------------------------------
extern "C"
__global__ void cuda_CNTgenerate_flex(	int *d_dmnAddr,
										short *d_dmnOcc,
										int *d_masks,
										int *d_RNGseed,
										float *d_result,
										short *d_Crtd)

{

	__shared__ int mask[6];
	__shared__ int vol;
	__shared__ unsigned int shrCNT;
	__shared__ unsigned int devCNT;
	__shared__ float shrArr[9*BLOCK];
	extern __shared__ int dynShr[];

	//if(blockIdx.x==94)	{
	
	// obtain segment boundaries:
	if (threadIdx.x == 0) {
		shrCNT = 0;
		devCNT = 0;
		vol = getMask(mask,d_masks);
	}
	shrClean(shrArr);
	__syncthreads();

	// set shared arrays:
	int	*devAddr = (int*) &(dynShr[0*vol]);
	int	*shrAddr = (int*) &(dynShr[2*vol]);
	
	// prepare dynamic array to store cell occupancy and border addresses
	prepareDynamicShr(vol, mask, devAddr, d_dmnAddr, d_dmnOcc,dmnExt);

	if (threadIdx.x < vol) {
		int locCNTs = devAddr[threadIdx.x+1*vol]-devAddr[threadIdx.x+0*vol];	// how many CNTs to create in current domain
		atomicAdd(&devCNT,locCNTs); // <-- avoid this for _DEBUGging
	} 
	__syncthreads();
	
	// Initial work:
	int x0 = mask[0]; int y0 = mask[1]; int z0 = mask[2];
	int dx = mask[3]; int dy = mask[4]; int dz = mask[5];
	__syncthreads();
	
	// generation:
	curandState	cuRNGstate;
	curand_init(d_RNGseed[blockIdx.x * blockDim.x + threadIdx.x],0,0,&cuRNGstate);
	
	float sample;
	float den_local;

	float  x, y, z;
	float3 incCos;
	float key;

	bool flag = false;
	while (shrCNT<devCNT) {

		if (!flag) {
		  
			x = dx*(1-curand_uniform(&cuRNGstate));
			y = dy*(1-curand_uniform(&cuRNGstate));
			z = dz*(1-curand_uniform(&cuRNGstate));
			key = (float) (floor(x) + floor(y)*dx + floor(z)*dx*dy);
			
			x += x0;	y += y0;	z += z0;
			
			shrArr[threadIdx.x+0*blockDim.x] = x*phsScl[0];
			shrArr[threadIdx.x+1*blockDim.x] = y*phsScl[1];
			shrArr[threadIdx.x+2*blockDim.x] = z*phsScl[2];
			
			x /= dmnExt[0];	y /= dmnExt[1];	z /= dmnExt[2];

			den_local = tex3D(denTex,x,y,z);
			flag = (key<vol);
			flag = flag && isInCell(	shrArr[threadIdx.x+0*blockDim.x],
										shrArr[threadIdx.x+1*blockDim.x],
										shrArr[threadIdx.x+2*blockDim.x],
										key, mask, dmnExt, phsScl,epsilon) ;
//*
			sample = curand_uniform(&cuRNGstate);
			
			flag = ( sample < den_local ) && flag; //*/
			//flag = true;
			if (flag) {
				atomicInc(&shrCNT,blockDim.x);

				incCos = genCosine( Phi_avg, Phi_dev, Theta_avg, Theta_dev, &cuRNGstate);
			
				// rotate to the required orientation
				incCos = inc_rotation(incCos, prefDir[0], -prefDir[1] );
			} 

			shrArr[threadIdx.x+3*blockDim.x] = (flag?incCos.x:-1);
			shrArr[threadIdx.x+4*blockDim.x] = (flag?incCos.y:-1);
			shrArr[threadIdx.x+5*blockDim.x] = (flag?incCos.z:-1);

			shrArr[threadIdx.x+6*blockDim.x] = (flag?DEF_L:-1);
			shrArr[threadIdx.x+7*blockDim.x] = (flag?DEF_A:-1);
			shrArr[threadIdx.x+8*blockDim.x] = (flag?key:-1);

		}
		__syncthreads(); 
	}
	__syncthreads(); 
	
	// WRITE RESULTS TO GLOBAL MEMORY:
	writeGlbMem(vol, mask, devAddr, shrAddr, d_Crtd, shrArr, d_result, dmnExt, numCNT); 
	
	__syncthreads();
//}	
}
// Generate CNT - orientation array:-------------------------------------------------------------------------------
extern "C"
__global__ void cuda_CNTgenerate_orient(	int		*d_dmnAddr,
											short	*d_dmnOcc,
											short	*d_Crtd,
											int		*d_masks,
											int		*d_RNGseed,
											float	*d_theta_avg,		// avarage theta angle 
											float	*d_theta_dev,		// mean squared deviation of theta
											float	*d_phi_avg,			// avarage phi angle 
											float	*d_phi_dev,			// mean squared deviation of phi
											float	*d_prefMu,			// prefered orientation - in XY plane
											float	*d_prefXi,			// prefered orientation - from Z axis
											float	*d_result)

{

	__shared__ int mask[6];
	__shared__ int vol;
	__shared__ unsigned int shrCNT;
	__shared__ unsigned int devCNT;
	__shared__ float shrArr[9*BLOCK];
	extern __shared__ int dynShr[];
	
	// obtain segment boundaries:
	if (threadIdx.x == 0) {
		shrCNT = 0;
		devCNT = 0;
		vol = getMask(mask,d_masks);
	}
	shrClean(shrArr);
	__syncthreads();

	// set shared arrays:
	int		*devAddr = (int*)	&(dynShr[0*vol]);
	int		*shrAddr = (int*)	&(dynShr[2*vol]);
	float	*shrOrt_avg	 = (float*)	&(dynShr[4*vol]);
	float	*shrOrt_dev	 = (float*)	&(dynShr[6*vol]);
	float	*shrOrt_prf	 = (float*)	&(dynShr[8*vol]);
	
	// prepare dynamic array to store cell occupancy and border addresses
	int glbPos = prepareDynamicShr(vol, mask, devAddr, d_dmnAddr, d_dmnOcc,dmnExt);
	if ((glbPos >= 0)&&(threadIdx.x<vol)) {
		shrOrt_avg[threadIdx.x+0*vol] = d_phi_avg[glbPos];			// average phi
		shrOrt_avg[threadIdx.x+1*vol] = d_theta_avg[glbPos];		// average theta
		shrOrt_dev[threadIdx.x+0*vol] = d_phi_dev[glbPos];			// deviation of phi
		shrOrt_dev[threadIdx.x+1*vol] = d_theta_dev[glbPos];		// deviation of theta
		shrOrt_prf[threadIdx.x+0*vol] = d_prefMu[glbPos];			// average phi
		shrOrt_prf[threadIdx.x+1*vol] = d_prefXi[glbPos];			// average theta
	}
	__syncthreads();
	__threadfence_block();

	if (threadIdx.x < vol) {
		int locCNTs = devAddr[threadIdx.x+1*vol]-devAddr[threadIdx.x+0*vol];	// how many CNTs to create in current domain
		atomicAdd(&devCNT,locCNTs); // <-- avoid this for _DEBUGging
	} 
	__syncthreads();
	
	// Initial work:
	int x0 = mask[0]; int y0 = mask[1]; int z0 = mask[2];
	int dx = mask[3]; int dy = mask[4]; int dz = mask[5];
	__syncthreads();
	
	// generation:
	curandState	cuRNGstate;
	curand_init(d_RNGseed[blockIdx.x * blockDim.x + threadIdx.x],0,0,&cuRNGstate);
	
	float sample;
	float den_local;

	float  x, y, z;
	float3 incCos;
	float key;

	bool flag = false;
	while (shrCNT<devCNT) {

		if (!flag) {
		  
			x = dx*(1-curand_uniform(&cuRNGstate));
			y = dy*(1-curand_uniform(&cuRNGstate));
			z = dz*(1-curand_uniform(&cuRNGstate));
			key = (float) (floor(x) + floor(y)*dx + floor(z)*dx*dy);
			
			x += x0;	y += y0;	z += z0;
			
			shrArr[threadIdx.x+0*blockDim.x] = x*phsScl[0];
			shrArr[threadIdx.x+1*blockDim.x] = y*phsScl[1];
			shrArr[threadIdx.x+2*blockDim.x] = z*phsScl[2];
			
			x /= dmnExt[0];	y /= dmnExt[1];	z /= dmnExt[2];

			sample = curand_uniform(&cuRNGstate);
			den_local = tex3D(denTex,x,y,z);
			flag = ( sample < den_local )&&(key<vol)&&( isInCell(	shrArr[threadIdx.x+0*blockDim.x],
																	shrArr[threadIdx.x+1*blockDim.x],
																	shrArr[threadIdx.x+2*blockDim.x],
																	key, mask, dmnExt, phsScl,epsilon) ) ;
			
			if (flag) {
				atomicInc(&shrCNT,blockDim.x);
				// generate directional cosines:
				incCos = genCosine(shrOrt_avg[(int)key+0*vol],shrOrt_dev[(int)key+0*vol],shrOrt_avg[(int)key+1*vol],shrOrt_dev[(int)key+1*vol],&cuRNGstate);
				// rotate to the required orientation
				incCos = inc_rotation(incCos, shrOrt_prf[(int)key+0*vol], -shrOrt_prf[(int)key+1*vol]);
			} 

			shrArr[threadIdx.x+3*blockDim.x] = (flag?incCos.x:-1);
			shrArr[threadIdx.x+4*blockDim.x] = (flag?incCos.y:-1);
			shrArr[threadIdx.x+5*blockDim.x] = (flag?incCos.z:-1);

			shrArr[threadIdx.x+6*blockDim.x] = (flag?DEF_L:-1);
			shrArr[threadIdx.x+7*blockDim.x] = (flag?DEF_A:-1);
			shrArr[threadIdx.x+8*blockDim.x] = (flag?key:-1);

		}
		__syncthreads(); 
	}
	__syncthreads(); 
	
	// WRITE RESULTS TO GLOBAL MEMORY:
	writeGlbMem(vol, mask, devAddr, shrAddr, d_Crtd, shrArr, d_result, dmnExt, numCNT); 
	
	__syncthreads();

} 
// device functions:-----------------------------------------------------------------------------------------------
inline __device__ float3 genCosine(float phi_avg, float phi_dev,float theta_avg, float theta_dev, curandState *state) {
	float3	cosine;
	cosine.z =	( Theta_dev != -1 ? cosf(sym_truncNormal( theta_avg, theta_dev, Pi/2, state) ) : 2*curand_uniform(state)-1 );
	float phi = (   Phi_dev != -1 ?		 sym_truncNormal( phi_avg,	 phi_dev,   Pi, state)	: 2*Pi*curand_uniform(state) );
	if (cosine.z > 1) cosine.z -= 2;
	cosine.x = cosf(phi)*sqrtf(1-cosine.z*cosine.z);
	cosine.y = sinf(phi)*sqrtf(1-cosine.z*cosine.z);
	return cosine;
}
inline __device__ float sym_truncNormal(float a, float s, float w, curandState	*state) {
	float Fa = 0.5*erf((-w)/s/sqrtf(2.0f));
	float xi = Fa*(1-2*curand_uniform(state));
	return a+2*s*sqrtf(2.0f)*erfinv(xi);
}

inline __device__ float Chatterjee(float b, float m, curandState	*state) {
	// distribution suggested in Chatterjee, A.P. J Chem Phys 140 (2014) 204911
	// g(t) = a+b*abs(cos(theta))^m
	// a+b/(m+1) = 1/2Pi (for theta [0;Pi/2) )
	float a = 1./2./Pi-b/(m+1.);
	float gc = 1.0-curand_uniform(state);
	return powf((gc-a)/b,1./m);
}


// rotation:

inline __device__ float3 inc_rotation(float3 vec, float Mu, float Xi) {
float3	result;
		result.x = ( cosf(Mu)*cosf(Xi))*vec.x + (sinf(Mu))*vec.y + (-sinf(Xi)*cosf(Mu))*vec.z;
		result.y = (-sinf(Mu)*cosf(Xi))*vec.x + (cosf(Mu))*vec.y + ( sinf(Xi)*sinf(Mu))*vec.z;
		result.z =			( sinf(Xi))*vec.x +		   (0)*vec.y +			( cosf(Xi))*vec.z;
return result;
}
