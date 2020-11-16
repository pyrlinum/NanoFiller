#include "vectorMath.h"

// statistical distributions in device implementation:
inline __device__ float stat_gauss1D(float X, float AVG, float DEV) {
	return	1.0f/sqrtf(2*PI)/DEV*expf(-(X-AVG)*(X-AVG)/2/DEV/DEV);
}
inline __device__ float stat_gauss2D(float2 X, float2 AVG, float2 DEV) {
	// independent variables:
	return	stat_gauss1D(X.x,AVG.x,DEV.x)*
			stat_gauss1D(X.y,AVG.y,DEV.y);
}
inline __device__ float stat_gauss3D(float3 X, float3 AVG, float3 DEV) {
	// independent variables:
	return	stat_gauss1D(X.x,AVG.x,DEV.x)*
			stat_gauss1D(X.y,AVG.y,DEV.y)*
			stat_gauss1D(X.z,AVG.z,DEV.z);
}