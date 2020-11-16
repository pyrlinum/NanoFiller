#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <math.h>
#ifndef VECTORMATH_H
#define VECTORMATH_H
#define PI	3.141593f
#define float_prec 1.0E-6
#define Matr_Prec 0.5E-2
//-------------------------------------------------------------------------------------------------
// OPERATORS:
//-------------------------------------------------------------------------------------------------
inline __device__ float3 operator*(float3 a,float s) {		// scale operator:
	float3 vec;
	vec.x = a.x*s;
	vec.y = a.y*s;
	vec.z = a.z*s;
	return vec;
}

inline __device__ float3 operator+(float3 a, float3 b) {	// vector plus:
	float3 vec;
	vec.x = a.x + b.x;
	vec.y = a.y + b.y;
	vec.z = a.z + b.z;
	return vec;
}

inline __device__ float3 operator-(float3 a) {				// vector negate:
	float3 vec;
	vec.x = -a.x;
	vec.y = -a.y;
	vec.z = -a.z;
	return vec;
}
inline __device__ float3 operator-(float3 a, float3 b) {	// vector minus:
	float3 vec;
	vec.x = a.x - b.x;
	vec.y = a.y - b.y;
	vec.z = a.z - b.z;
	return vec;
}
inline __device__ float norm3(float3 c) {					// vector norm in 3D
	//return pow(c.x*c.x+c.y*c.y+c.z*c.z,0.5f);
	return sqrtf(c.x*c.x+c.y*c.y+c.z*c.z);
}
inline __device__ float translate(float x, float ref) {		// Translate coordinate to satisfy periodic boundary conditions:
	float	d = (x-ref);
	if (d >  0.5) {
		return x-1;
	} else {
		if (d < -0.5) {
			return x+1;
		} else {
			return x;
		}
	}
}
inline __device__ float3 transl3(float3 r, float3 ref) {	// Translate point coordinates to satisfy periodic boundary conditions:
	float x = translate(r.x,ref.x);
	float y = translate(r.y,ref.y);
	float z = translate(r.z,ref.z);
	return make_float3(x,y,z);
}
inline __device__ float3 unwrapZ(float3 r0) {									// unwrap cylindrical surface around Z axis
	float rad = sqrtf(r0.x*r0.x+r0.y*r0.y);
	return make_float3((r0.x>=0?1:-1)*rad*acosf(r0.y/rad),0.0f,r0.z);
}
inline __device__ float  trace(float A[3][3]) {
	float tr = 0;
	for(int i=0;i<3;i++) tr += A[i][i];
	return tr;
}
inline __device__ double  trace(double A[3][3]) {
	double tr = 0;
	for(int i=0;i<3;i++) tr += A[i][i];
	return tr;
}
inline __device__ char sign(float x) {
	 return ( x>0 ? 1 : (x<0?-1:0) );
 } 
inline __device__ char sign(double x) {
	 return ( x>0 ? 1 : (x<0?-1:0) );
 }
inline __device__ float2 Ort2Angle(float3 vec) {
	float nrmXY = sqrt(vec.x*vec.x+vec.y*vec.y);
	float theta =  asinf(nrmXY);
	float phi	= (nrmXY > float_prec ? acosf(vec.x/nrmXY) : 0);
	return make_float2(theta,phi);
}
//-------------------------------------------------------------------------------------------------
// Diagonalisation:
//-------------------------------------------------------------------------------------------------
// row swap:
inline __device__ void swapRow(char I, char J, float  A[3][3]) {
	float  buf;
	for(int k=0;k<3;k++) {
		buf = A[I][k];
		A[I][k] = A[J][k];
		A[J][k] = buf;
	}
}
inline __device__ float  pivot(char I, char Ic, float  A[3][3])	{
	float  max = abs(A[I][Ic]);
	char rm = I;
	for(char j=I+1;j<3;j++)
		if(abs(A[j][Ic])>max) {
			rm = j;
		}
		if ((rm != I)) {
			swapRow(I,rm,A);
		}
	return A[I][Ic];
}
inline __device__ void ForwardElim(float  A[3][3]) {
	// forward substitution:
	for(char I=0; I<3; I++) {
		char Ic = I;
		while ((abs(pivot(I,Ic,A))<=1.0E-6)&&(Ic<2)) Ic++;
		if(abs(A[I][Ic])>=1.0E-6)
			for(char J=I+1;J<3;J++)	{
				for(char K=2;K>=Ic;K--)
					A[J][K] -= A[I][K]*A[J][Ic]/A[I][Ic];
			}
	}
}
//-------------------------------------------------------------------------------------------------
// MATRIX DETERMINANTS:
//-------------------------------------------------------------------------------------------------
inline __device__ float MatrDet2x2(float A11, float A12, float A21, float A22) {
	return A11*A22-A12*A21;
}
inline __device__ float MatrDet3x3(float A[3][3]) {
	float	R1 = A[0][0]*MatrDet2x2(A[1][1],A[1][2],A[2][1],A[2][2]);
	float	R2 =-A[0][1]*MatrDet2x2(A[1][0],A[1][2],A[2][0],A[2][2]);
	float	R3 = A[0][2]*MatrDet2x2(A[1][0],A[1][1],A[2][0],A[2][1]);
	return R1+R2+R3;
}
inline __device__ double MatrDet2x2(double A11, double A12, double A21, double A22) {
	return A11*A22-A12*A21;
}
inline __device__ double MatrDet3x3(double A[3][3]) {
	double	R1 = A[0][0]*MatrDet2x2(A[1][1],A[1][2],A[2][1],A[2][2]);
	double	R2 =-A[0][1]*MatrDet2x2(A[1][0],A[1][2],A[2][0],A[2][2]);
	double	R3 = A[0][2]*MatrDet2x2(A[1][0],A[1][1],A[2][0],A[2][1]);
	return R1+R2+R3;
}
inline __device__ char MatrRank3x3(float A[3][3]) {
	char rk = 0;
	 for(char I=0;I<3;I++)
		 for(char J=0;J<3;J++)
			 if(abs(A[I][J])>float_prec) rk=1;
	 if (rk>0)	{
		float R1 = abs(MatrDet2x2(A[0][0],A[0][1],A[1][0],A[1][1]));
		float R2 = abs(MatrDet2x2(A[1][0],A[1][1],A[2][0],A[2][1]));
		float R3 = abs(MatrDet2x2(A[0][1],A[0][2],A[1][1],A[1][2]));
		float R4 = abs(MatrDet2x2(A[1][1],A[1][2],A[2][1],A[2][2]));
		R1 = ( R2 > R1 ? R2 : R1);
		R3 = ( R4 > R3 ? R4 : R3);
		R1 = ( R3 > R1 ? R3 : R1);
		if (R1>float_prec) {
			rk=2;
			if (abs(MatrDet3x3(A))>Matr_Prec) rk=3;
		}}
	return rk;
}

//-------------------------------------------------------------------------------------------------
// VECTOR AND MATRIX PRODUCTS:
//-------------------------------------------------------------------------------------------------
// scalar product of vectors a and b:
inline __device__ float dotProd(float3 a, float3 b) {	
	return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ float3 vecProd(float3 a, float3 b) {
	float3 vec;
	vec.x =  MatrDet2x2(a.y,a.z,b.y,b.z);
	vec.y = -MatrDet2x2(a.x,a.z,b.x,b.z);
	vec.z =  MatrDet2x2(a.x,a.y,b.x,b.y);
	return vec;
}

inline __device__ float mixProd(float3 a, float3 b, float3 c) {
	float3 vec = vecProd(b,c);
	return dotProd(a,vec);
}
inline __device__ void VecMatProd3x3(float A[3][3], float B[3]) {
	float C[3];
	for(int i=0;i<3;i++)
		C[i] = A[i][0]*B[0]+A[i][1]*B[1]+A[i][2]*B[2];
	for(int i=0;i<3;i++)
		B[i]=C[i];
}
inline __device__ void VecMatProd3x3(double A[3][3], double B[3]) {
	double C[3];
	for(int i=0;i<3;i++)
		C[i] = A[i][0]*B[0]+A[i][1]*B[1]+A[i][2]*B[2];
	for(int i=0;i<3;i++)
		B[i]=C[i];
}
inline __device__ void MatProd3x3(float A[3][3], float B[3][3]) {
	float C[3][3];
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			C[i][j] = A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			B[i][j] = C[i][j];
}
inline __device__ void MatProd3x3(double A[3][3], double B[3][3]) {
	double C[3][3];
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			C[i][j] = A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			B[i][j] = C[i][j];
}
//-------------------------------------------------------------------------------------------------
// DISTANCES AND ANGLES IN 3D:
//-------------------------------------------------------------------------------------------------
inline __device__ float dist3(float3 A, float3 B) { // distance between points under periodic boundary conditions
	return norm3(A-transl3(B,A));
}
inline __device__ float dist2pnt(float3 r, float3 r0) {
	return norm3(r-r0);
}
inline __device__ float dist2line(float3 r, float3 r0, float3 c) {
	return norm3((r-r0)-c*dotProd((r-r0),c));
}
inline __device__ float cosVec2Vec(float3 v1, float3 v2) {
	return dotProd(v1,v2)/norm3(v1)/norm3(v2);
}
//-------------------------------------------------------------------------------------------------
// VECTOR ROTATION:
//-------------------------------------------------------------------------------------------------
inline __device__ void rot_X(float alpha, float V0[3]) {
	float MatX[3][3] =	{	{ 1,			  0,			0 },
							{ 0,	cosf(alpha), -sinf(alpha) },
							{ 0,	sinf(alpha),  cosf(alpha) }	};
	VecMatProd3x3(MatX,V0);
}
inline __device__ void rot_Y(float alpha, float V0[3]) {
	float MatY[3][3] =	{	{ cosf(alpha), 0, -sinf(alpha) },
							{			0, 1,			0  },
							{ sinf(alpha), 0,  cosf(alpha) }	};
	VecMatProd3x3(MatY,V0);
}
inline __device__ void rot_Z(float alpha, float V0[3]) {
	float MatZ[3][3] =	{	{ cosf(alpha), -sinf(alpha), 0 },
							{ sinf(alpha),  cosf(alpha), 0 },
							{			0,			  0, 1 }	};
	VecMatProd3x3(MatZ,V0);
}

inline __device__ float3 rot_X(float alpha, float3 V0) {
	float V1[3] = {V0.x,V0.y,V0.z};
	float MatX[3][3] =	{	{ 1,			  0,			0 },
							{ 0,	cosf(alpha), -sinf(alpha) },
							{ 0,	sinf(alpha),  cosf(alpha) }	};
	VecMatProd3x3(MatX,V1);
	return make_float3(V1[0],V1[1],V1[2]);
}
inline __device__ float3 rot_Y(float alpha, float3 V0) {
	float V1[3] = {V0.x,V0.y,V0.z};
	float MatY[3][3] =	{	{ cosf(alpha), 0, -sinf(alpha) },
							{			0, 1,			0  },
							{ sinf(alpha), 0,  cosf(alpha) }	};
	VecMatProd3x3(MatY,V1);
	return make_float3(V1[0],V1[1],V1[2]);
}
inline __device__ float3 rot_Z(float alpha, float3 V0) {
	float V1[3] = {V0.x,V0.y,V0.z};
	float MatZ[3][3] =	{	{ cosf(alpha), -sinf(alpha), 0 },
							{ sinf(alpha),  cosf(alpha), 0 },
							{			0,			  0, 1 }	};
	VecMatProd3x3(MatZ,V1);
	return make_float3(V1[0],V1[1],V1[2]);
}
 
// PERIODIC TRANSLATION:
inline __device__ float3 get_relative_crd(float3 A,float3 B,float3 physDim) {
	float3 AB=B-A;

	if (AB.x>0.5*physDim.x) {
		AB.x = AB.x-physDim.x;
	} else if (AB.x<-0.5*physDim.x) {
		AB.x = AB.x+physDim.x;
	}

	if (AB.y>0.5*physDim.y) {
		AB.y = AB.y-physDim.y;
	} else if (AB.y<-0.5*physDim.y) {
		AB.y = AB.y+physDim.y;
	}

	if (AB.z>0.5*physDim.z) {
		AB.z = AB.z-physDim.z;
	} else if (AB.z<-0.5*physDim.z) {
		AB.z = AB.z+physDim.z;
	}

	return A+AB;
}

#endif
