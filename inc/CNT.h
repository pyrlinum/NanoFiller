// This file describes CNT structure and basic operations with it
#pragma once
#include "vectorMath.h"
#include "definitions.h"

typedef struct {
	float3	r;	// CNT center coordinates
	float3	c;	// CNT directional cosines
	float	l;	// CNT length
	float	a;	// CNT Diameter
	float	k;	// diagnostics only
} CNT_t;

// Create CNT:
inline __device__ CNT_t make_CNT(float3 r,float3 c,float l,float a, float k) {
	CNT_t cnt;	
	cnt.r = r;
	cnt.c = c*(1.0f/norm3(c));
	cnt.l = l;
	cnt.a = a;
	cnt.k = k;
return cnt;
} 

// Create empty CNT:
inline  __device__ CNT_t make_emptyCNT()  {
	CNT_t cnt;	
	cnt.r = make_float3(DEF_EMPTY_VAL,DEF_EMPTY_VAL,DEF_EMPTY_VAL);
	cnt.c = make_float3(DEF_EMPTY_VAL,DEF_EMPTY_VAL,DEF_EMPTY_VAL);
	cnt.l = DEF_EMPTY_VAL;
	cnt.a = DEF_EMPTY_VAL;
	cnt.k = DEF_EMPTY_VAL;

return cnt;
} 

// Create empty CNT - compatibility:
inline  __device__ void make_empty(CNT_t* probe)  {
	*probe = make_emptyCNT();
}

// surface energy of CNT:
inline __device__ float totInclArea(CNT_t probe) {
	return 2*PI*probe.a*probe.l;
};

// Create CNT from raw data :
inline __device__ void make_INC(CNT_t* cnt, float* data) {
	cnt->r = make_float3(data[0],data[1],data[2]);
	cnt->c = make_float3(data[3],data[4],data[5]);
	cnt->l = data[6];
	cnt->a = data[7];
	cnt->k = data[8];
};


// Create CNT from raw data :
inline __device__ void make_RAW(CNT_t* cnt, float* data) {
	data[0] = cnt->r.x;
	data[1] = cnt->r.y;
	data[2] = cnt->r.z;
	data[3] = cnt->c.x;
	data[4] = cnt->c.y;
	data[5] = cnt->c.z;
	data[6] = cnt->l;
	data[7] = cnt->a;
	data[8] = cnt->k;
}
