// Basic operations with ellipses and circles, used for intersection area calculation
#pragma once
#ifndef ELLIPSE_H
#define ELLIPSE_H
#include "vectorMath.h"
//----------------------------------------------------------------------------------
// Auxilary operations:
//----------------------------------------------------------------------------------
//solve square equation at^2+bt+c=0:
inline __device__ float sqe_rootP(float a,float b, float c) {			// positive root of at^2+bt+c=0 
	return (-b+powf(b*b-4*a*c,0.5f))/2/a;
}
inline __device__ float sqe_rootN(float a,float b, float c) {			// negative root of at^2+bt+c=0
	return (-b-powf(b*b-4*a*c,0.5f))/2/a;
}
inline __device__ float elli_tilt_angle(float a, float b, float c) {	// tilt angle between X axis and eigen axis
	return 0.5f*acosf((a-c)*rsqrtf((a-c)*(a-c)-b*b));
}
inline __device__ float2 elli_half_axii(float a, float b, float c) {	// eigen half axii of ellipse in form ax^2+bxy+cy^2
	float l1 = 0.5f*((a+c)+sqrtf((a-c)*(a-c)+b*b));
	float l2 = 0.5f*((a+c)-sqrtf((a-c)*(a-c)+b*b));
	return make_float2(l1,l2);
}
inline __device__ float ellipse(float a, float b, float t) {					// elliptic equation in polar coordinates
	return a*b*rsqrtf(b*cosf(t)*b*cosf(t) + a*sinf(t)*a*sinf(t));
}
//----------------------------------------------------------------------------------
// AREA:
//----------------------------------------------------------------------------------
inline __device__ float tria_area(float a, float b, float phi) {		// triangle area
	return 0.5f*a*b*sinf(phi);
}
inline __device__ float tria_area(float3 A, float3 B) {					// triangle area
	return 0.5f*norm3(vecProd(A,B));
}
inline __device__ float circ_area(float R) {							// circle surface
	return PI*R*R;
}
inline __device__ float circ_area_seg(float R,float phi) {				// area of circle segment corresponding to angle 0<phi<Pi
	return R*(phi-0.5f*sinf(phi));
}
inline __device__ float elli_area(float a, float b) {					// ellipse surface in canonical form
	return PI*a*b;
}
inline __device__ float elli_area(float a, float b, float c) {			// ellipse surface in rotated form
	return 2*PI*rsqrtf(4*a*c-b*b);
}
inline __device__ float elli_area_sec(float a, float b, float phi) {	// surface of elliptic sector from 0 to phi (0<pi<PI/2)
	if (phi<PI/2) {
		return a*b*atanf(a/b*tanf(phi));
	} else {
		return 0.5f*PI*a*b - a*b*atanf(a/b*tanf(PI-phi));
	}
}
inline __device__ float elli_area_sec(float a, float b, float phi1, float phi2) {	// surface of elliptic sector from phi1 to phi2
	return abs( (phi2>=0?1:-1)*elli_area_sec(a,b,abs(phi2)) - (phi1>=0?1:-1)*elli_area_sec(a,b,abs(phi1)) );
}
inline __device__ float elli_area_seg(float a, float b, float phi1, float phi2) {	// surface of elliptic segment from phi1 to phi2
	float tri = tria_area(a,b,phi2-phi1);
	float sec = elli_area_sec(a,b,phi1);
	return sec-tri;
}
inline __device__ float elli_area_symseg(float a, float b, float x) {	// surface of symmetric elliptic segment
	return PI*a*b/2+b/a*(x*sqrt(a*a-x*x)+a*a*asinf(x/a));
}
//----------------------------------------------------------------------------------
// Fitting ellipse by 3 points:
//----------------------------------------------------------------------------------
inline __device__ float3 ellipse_fit(double2 x1, double2 x2, double2 x3) {	// fit ellips coefficients in equation ax^2+bxy+cy^2=1

	double A0[3][3] = {	{x1.x*x1.x,x1.x*x1.y,x1.y*x1.y},
						{x2.x*x2.x,x2.x*x2.y,x2.y*x2.y},
						{x3.x*x3.x,x3.x*x3.y,x3.y*x3.y}	};
	double det0 = MatrDet3x3(A0);
	double A1[3][3] = {	{		1 ,x1.x*x1.y,x1.y*x1.y},
						{		1 ,x2.x*x2.y,x2.y*x2.y},
						{		1 ,x3.x*x3.y,x3.y*x3.y}	};
	double det1 = MatrDet3x3(A1);
	double A2[3][3] = {	{x1.x*x1.x,		1 ,x1.y*x1.y},
						{x2.x*x2.x,		1 ,x2.y*x2.y},
						{x3.x*x3.x,		1 ,x3.y*x3.y}	};
	double det2 = MatrDet3x3(A2);
	double A3[3][3] = {	{x1.x*x1.x,x1.x*x1.y,		1 },
						{x2.x*x2.x,x2.x*x2.y,		1 },
						{x3.x*x3.x,x3.x*x3.y,		1 }	};
	double det3 = MatrDet3x3(A3);

	double a = det1/det0;
	double b = det2/det0;
	double c = det3/det0;

	return make_float3(a,b,c);
}
inline __device__ float3 ellipse_fit(float2 x1, float2 x2, float2 x3) {	// fit ellips coefficients in equation ax^2+bxy+cy^2=1

	float A0[3][3] = {	{x1.x*x1.x,x1.x*x1.y,x1.y*x1.y},
						{x2.x*x2.x,x2.x*x2.y,x2.y*x2.y},
						{x3.x*x3.x,x3.x*x3.y,x3.y*x3.y}	};
	float det0 = MatrDet3x3(A0);
	float A1[3][3] = {	{		1 ,x1.x*x1.y,x1.y*x1.y},
						{		1 ,x2.x*x2.y,x2.y*x2.y},
						{		1 ,x3.x*x3.y,x3.y*x3.y}	};
	float det1 = MatrDet3x3(A1);
	float A2[3][3] = {	{x1.x*x1.x,		1 ,x1.y*x1.y},
						{x2.x*x2.x,		1 ,x2.y*x2.y},
						{x3.x*x3.x,		1 ,x3.y*x3.y}	};
	float det2 = MatrDet3x3(A2);
	float A3[3][3] = {	{x1.x*x1.x,x1.x*x1.y,		1 },
						{x2.x*x2.x,x2.x*x2.y,		1 },
						{x3.x*x3.x,x3.x*x3.y,		1 }	};
	float det3 = MatrDet3x3(A3);

	float a = det1/det0;
	float b = det2/det0;
	float c = det3/det0;

	return make_float3(a,b,c);
}
//----------------------------------------------------------------------------------
// Cylinder-Cylinder intersection area:
//----------------------------------------------------------------------------------
inline float __device__ isec_area(float R1, float R2, float y2, float3 c2) {
// considering that cyl1 is along Z axis with the origin in zero
// and the second one has an origin on Y axis

// find intersection points:
	// first cylinder axis:
	float t1 = +pow((R2*R2-(R1-y2)*(R1-y2))/(pow(c2.x*c2.z,2)+pow(c2.z*c2.z-1,2)),0.5f);
	float3	P1 = make_float3(0,R1,t1);
	// second cylinder axis:
	t1 = +pow((R1*R1-pow(y2-R2,2)),0.5f)/c2.x;
	float3	P3 = make_float3(c2.x*t1,(y2-R2),c2.z*t1);

	// circle-ellipse intersection:
	float a = c2.x*c2.x;
	float b = 2*y2*(1-c2.x*c2.x);
	float c = R2*R2-R1*R1-(1-c2.x*c2.x)*y2*y2;
	float y1 = sqe_rootP(a,b,c);
	float x1 = +pow(R1*R1-y1*y1,0.5f);
	float3 P5 = make_float3(x1,y1,0);
	// Unwrap arount z axis:
	P1 = unwrapZ(P1);
	P3 = unwrapZ(P3);
	P5 = unwrapZ(P5);

	// fit ellipse
	float3 coeff = ellipse_fit(make_float2(P1.x,P1.z),make_float2(P3.x,P3.z),make_float2(P5.x,P5.z));

return elli_area(coeff.x,coeff.y,coeff.z);
}
#endif
