// Function to calculate the distance between the two sphero-cylindrical inclusions
// TODO: replace with class

#pragma once
#include "definitions.h"
#include "vectorMath.h"
// access fields:
// partA - resides in register file:
#define	partA_rx	regArr[0]
#define	partA_ry	regArr[1]
#define	partA_rz	regArr[2]
#define	partA_cx	regArr[3]
#define	partA_cy	regArr[4]
#define	partA_cz	regArr[5]
#define	partA_l 	regArr[6]
#define	partA_a 	regArr[7]
#define	partA_k 	regArr[8]
// partB - resides in shared memory:
#define	partB_rx	shrArr[i+0*stride]
#define	partB_ry	shrArr[i+1*stride]
#define	partB_rz	shrArr[i+2*stride]
#define	partB_cx	shrArr[i+3*stride]
#define	partB_cy	shrArr[i+4*stride]
#define	partB_cz	shrArr[i+5*stride]
#define	partB_l 	shrArr[i+6*stride]
#define	partB_a 	shrArr[i+7*stride]
#define	partB_k 	shrArr[i+8*stride]

// Auxiliary functions:
// get point along the axis:
inline	__device__ float3 get_cyl_point(float* regArr, float t) {
	float3 r = make_float3(partA_rx,partA_ry,partA_rz);
	float3 c = make_float3(partA_cx,partA_cy,partA_cz);
	return r+c*t;
}

// get inclusion parameters:
inline	__device__ float3 get_cyl_center(float* regArr) {
	return make_float3(partA_rx,partA_ry,partA_rz);
}
inline	__device__ float3 get_cyl_direction(float* regArr) {
	return make_float3(partA_cx,partA_cy,partA_cz);
}
inline	__device__ float get_cyl_length(float* regArr) {
	return partA_l;
}
inline	__device__ float get_cyl_radius(float* regArr) {
	return partA_a;
}
inline	__device__ float get_cyl_cell(float* regArr) {
	return partA_k;
}

// set center:
inline	__device__ void set_cyl_center(float* regArr,float3 pnt) {
	partA_rx = pnt.x;
	partA_ry = pnt.y;
	partA_rz = pnt.z;
}
// displace:
inline	__device__ void displace_cyl(float* regArr,float3 pnt) {
	float3 r = get_cyl_center(regArr);
	set_cyl_center(regArr,pnt+r);
}



// contact characteristics:
inline  __device__ float3 distance_cylinder_capped(float* regArr, float *shrArr, int stride, unsigned int i, float3* Dtt)  {

		float d;
		// find common perpendicular:
		float t[2] = {0,0};
		float3 a,b = make_float3(	partB_rx-partA_rx,
									partB_ry-partA_ry,
									partB_rz-partA_rz);			// r2-r1

		d = partA_cx*partB_cx+
			partA_cy*partB_cy+
			partA_cz*partB_cz;								// (c2*c1)

		if ( 1-abs(d) > DEV_PRECISION) {	// not parallel:
			// solving linear system
			// obtained from eq: (r2-r1)+(c2*t2-c1*t1) = c1xc2*f
			// xc1:	1*t1-c1c2*t2 = (r2-r1)*c1
			// xc2:	c1c2*t1-1*t2 = (r2-r1)*c2
			// det = (c1c2)^2-1
			// det1 = (r2-r1)*((c1c2)*c2-c1); t1=det1/det
			// det2 = (r2-r1)*(c2-(c1c2)*c1); t2=det2/det

			a = make_float3(d*partB_cx-partA_cx,
							d*partB_cy-partA_cy,
							d*partB_cz-partA_cz);			// (c1*c2)c2-c1
			t[0] = a.x*b.x+a.y*b.y+a.z*b.z;								// det1

			a = make_float3(partB_cx-d*partA_cx,
							partB_cy-d*partA_cy,
							partB_cz-d*partA_cz);			// c2-(c1*c2)c1

			t[1] = a.x*b.x+a.y*b.y+a.z*b.z;								// det2
			t[0] /= d*d-1;
			t[1] /= d*d-1;

		} else { // parallel
			// r2+c2*t2 = r2 - c2*(r2-r1) +c1*t1
			t[0] = copysignf(partA_l/2,b.x*partA_cx+b.y*partA_cy+b.z*partA_cz);
			t[1] = -1*(b.x*partB_cx+b.y*partB_cy+b.z*partB_cz) + d*t[0];
		}

		// coordinates for common perpendicular:
		Dtt->y = t[0];
		Dtt->z = t[1];

		bool flag1 = (abs(t[0])<=(partA_l/2 + partA_a + DEV_PRECISION));
		bool flag2 = (abs(t[1])<=(partB_l/2 + partB_a + DEV_PRECISION));

		if  ( !(flag1&&flag2) ) {
			// if at least one of the endpoints is out of limits - change the point that is further from the end
			bool flag3 = (abs(t[0])-partA_l/2)<(abs(t[1])-partB_l/2);
			t[flag3] = copysignf( (flag3 ? partB_l/2 : partA_l/2) , t[flag3] );

			a.x = flag3 ? partA_cx : partB_cx;
			a.y = flag3 ? partA_cy : partB_cy;
			a.z = flag3 ? partA_cz : partB_cz;

			t[!flag3] = (2*flag3-1)*dotProd(b,a) + d*t[flag3];

			float len = flag3 ? partA_l/2 : partB_l/2;
			if ( abs(t[!flag3]) > len )	t[!flag3] = copysignf( len, t[!flag3] );
		}
		// distance between the points:
		a.x = partA_rx + t[0]*partA_cx - partB_rx - t[1]*partB_cx;
		a.y = partA_ry + t[0]*partA_cy - partB_ry - t[1]*partB_cy;
		a.z = partA_rz + t[0]*partA_cz - partB_rz - t[1]*partB_cz;

		d = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);

		// distance along common perpendicular:
		a.x = partA_rx + Dtt->y*partA_cx - partB_rx - Dtt->z*partB_cx;
		a.y = partA_ry + Dtt->y*partA_cy - partB_ry - Dtt->z*partB_cy;
		a.z = partA_rz + Dtt->y*partA_cz - partB_rz - Dtt->z*partB_cz;

		Dtt->x = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);

	//return d;
	return make_float3(d,t[0],t[1]);
}

inline  __device__ float3 distance_cylinder_capped(float* regArr, float *shrArr, unsigned int i, float3* Dtt)  {
	return distance_cylinder_capped(regArr, shrArr, blockDim.x, i, Dtt);
}

inline  __device__ float3 distance_cylinder_capped(float* regArr, float *shrArr, float3* Dtt)  {
	return distance_cylinder_capped(regArr, shrArr, 1, 0, Dtt);
}

inline  __device__ bool if_intersects_cylinder_capped(float d, float* regArr, float *shrArr, int stride, unsigned int i, float separation, float soft_frac)  {
	return (d<((partB_a+partA_a)*soft_frac + separation + DEV_PRECISION));
}

inline  __device__ bool if_intersects_cylinder_capped(float d, float* regArr, float *shrArr, unsigned int i, float separation, float soft_frac)  {
	return if_intersects_cylinder_capped(d, regArr, shrArr, blockDim.x, i, separation, soft_frac);
}

inline  __device__ bool if_intersects_cylinder_capped(float d, float* regArr, float *shrArr, float separation, float soft_frac)  {
	return if_intersects_cylinder_capped(d, regArr, shrArr, 1, 0, separation, soft_frac);
}

inline  __device__ bool end_intersect_cylinder_capped(float t, float *shrArr, int stride, unsigned int i)  {
	return (abs(abs(t)-partB_l/2.0)<(partB_a+DEV_PRECISION));
}

inline  __device__ bool end_intersect_cylinder_capped(float t, float *shrArr, unsigned int i)  {
	return end_intersect_cylinder_capped(t, shrArr, blockDim.x, i);
}

inline  __device__ bool end_intersect_cylinder_capped(float t, float *shrArr)  {
	return end_intersect_cylinder_capped(t, shrArr, 1, 0);
}

inline  __device__ bool body_intersect_cylinder_capped(float t, float *shrArr, int stride, unsigned int i)  {
	return ((partB_l/2.0-abs(t))>(partB_a+DEV_PRECISION));
}

inline  __device__ bool body_intersect_cylinder_capped(float t, float *shrArr, unsigned int i)  {
	return body_intersect_cylinder_capped(t, shrArr, blockDim.x, i);
}

inline  __device__ bool body_intersect_cylinder_capped(float t, float *shrArr)  {
	return body_intersect_cylinder_capped(t, shrArr, 1, 0);
}

inline  __device__ bool intersect_cylinder_capped(float* regArr, float *shrArr, unsigned int i, float separation, float soft_frac)  {
	if (partA_l<0) {
		return false;
	} else {
		float3 Dtt, dtt = distance_cylinder_capped(regArr, shrArr, i, &Dtt);
		return (if_intersects_cylinder_capped(dtt.x, regArr, shrArr, i, separation, soft_frac));
	}
}


// Approximate surface of contact:
inline __device__ float surf_to_cylinder_capped( float *probe, float *shrArr ,int stride, int i, float3 Dtt, float rlim)  {
 	// define part of CNT surface within tunnelling range:
 	float	l1 = probe[6];
 	float	l2 = shrArr[i+6*stride];
 	float	a1 = probe[7];
 	float	a2 = shrArr[i+7*stride];
 	float3 	c1 = make_float3(	probe[3],
 								probe[4],
 								probe[5]	);
 	float3 	c2 = make_float3(	shrArr[i+3*stride],
 								shrArr[i+4*stride],
 								shrArr[i+5*stride]	);
 	float3 c2p = c2*copysign(1.0,dotProd(c1,c2));		// c1 and c2p will have sharp angle
 	float3 	r1 = make_float3(	probe[0],
 								probe[1],
 								probe[2]	);
 	float3 	r2 = make_float3(	shrArr[i+0*stride],
 								shrArr[i+1*stride],
 								shrArr[i+2*stride]	);

 	float dr1 = 0.0;
 	float dr2 = 0.0;
 	float	h1 = max(Dtt.x-a2-rlim,0.0);
 	float	h2 = max(Dtt.x-a1-rlim,0.0);
 	if ( h1 <= a1 )	dr1 = a1*sqrt(1-pow(h1/a1,2));
 	if ( h2 <= a2 )	dr2 = a2*sqrt(1-pow(h2/a2,2));
 	float dr = min(dr1,dr2);
 	float S = PI*dr*dr; // default surface area - point contact
 	float t1 = Dtt.y;
 	float t2 = Dtt.z;

 	//*
 	float d =	dotProd(c1,c2p);
 	float sng = sqrt(1-d*d); // sin gamma
 	if ( abs(t1)<(l1/2-a1) && abs(t2)<(l2/2-a2) ) {
 		// non-parallel case - ellipse approximation
 		// select in each direction closest of other tube edge projection and end-point of this tube:

 		// select axii of ellipse:
 		float3 r12 = (r2+c2*t2-r1-c1*t1)*(1.0/Dtt.x);

 		float3  ax1 = (c1+c2p); 					ax1 = ax1*(1.0/norm3(ax1));
 		float3	ax3 = r12 - ax1*dotProd(r12,ax1);	ax3 = ax3*(1.0/norm3(ax3));
 		float3	ax2 = vecProd(ax3,ax1);				ax2 = ax2*(1.0/norm3(ax2));

 		float csg = sqrt(1-sng*sng);
 		float sngh = sqrt((1-csg)/2.0);	// sin of half angle

 		// projection of intersection point on ax1:
 		float t2p = copysign(t2,dotProd(c1,c2));

 		float dt0p = ( sngh>DEV_PRECISION ? dr2/sngh : l1/2 );
 		float dt0m = dt0p;
 		float dt1p =     dotProd(r1+ c1*( l1/2-t1 ),ax1);
 		float dt1m = abs(dotProd(r1+ c1*(-l1/2-t1 ),ax1));
 		float dt2p =     dotProd(r2+c2p*( l2/2-t2p),ax1);
 		float dt2m = abs(dotProd(r2+c2p*(-l2/2-t2p),ax1));


 		float Ap  = min(dt0p,min(dt1p,dt2p)); Ap = max(dr,Ap);
 		float Am  = min(dt0m,min(dt1m,dt2m)); Am = max(dr,Am);

 		S = max(S,0.5*PI*dr*(Ap+Am));

 	}

 	if ( (sng<=(max(a1,a2)/min(l1,l2))) && (abs(t1)<=l1/2+a1 || abs(t2)<=l2/2+a2) ) {
 		// parallel case - projection:
 		// r2-r1:
 		float3	b = make_float3(	shrArr[i+0*stride] - probe[0],
 									shrArr[i+1*stride] - probe[1],
 									shrArr[i+2*stride] - probe[2]	);
 		float	t1a =    copysign(	l1/2, dotProd(c1,b) );
 		float	t1b = -1*copysign(	l1/2, dotProd(c1,b) );
 		float	t2a =    copysign(	l2/2, dotProd(c2,b) );
 		float	t2b = -1*copysign(	l2/2, dotProd(c2,b) );

 		float	t1ac = dotProd(  r1 + c1*t1a,c1);
 		float	t1bc = dotProd(  r1 + c1*t1b,c1);
 		float	t2ac = dotProd(b+r1 + c2*t2a,c1);
 		float	t2bc = dotProd(b+r1 + c2*t2b,c1);

 		float p1c = dotProd(c1,b)>0 ? min(t1ac,t2ac) : max(t1ac,t2ac);
 		float p2c = dotProd(c1,b)>0 ? max(t1bc,t2bc) : min(t1bc,t2bc);

 		S = max(S,abs(p2c-p1c)*2*dr);
 	}

 	return  S;
 }

inline __device__ float surf_to_cylinder_capped( float *probe, float *shrArr, unsigned int i , float3 Dtt, float rlim) {
	return surf_to_cylinder_capped( probe, shrArr ,blockDim.x, i, Dtt, rlim);
}

inline __device__ float surf_to_cylinder_capped( float *probe, float *shrArr , float3 Dtt, float rlim) {
	return surf_to_cylinder_capped( probe, shrArr ,1,0, Dtt, rlim);
}
