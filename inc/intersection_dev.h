//====================================================================================================================
//										 <<< INTERSECTION DEVICE FUNCTIONS >>>
//--------------------------------------------------------------------------------------------------------------------
#pragma once
#include "CNT.h"

//----------------------------------------------------------------------------------------------------------------------
// Check intersections between CNTs - shared memory implementation:
//----------------------------------------------------------------------------------------------------------------------
inline  __device__ bool cnt_intersec_shr(CNT_t probe, float *shrArr ,int stride, int i, float epsilon, float sep, float scf)  {
	// till the very end d stores c1*c2, vector b = r2-r1

	if (probe.l<=0) {
		return false;
	} else {
		float d;
		// find common perpendicular:
		float t[2] = {0,0};
		float3 a,b = make_float3(shrArr[i+0*stride]-probe.r.x,
								 shrArr[i+1*stride]-probe.r.y,
								 shrArr[i+2*stride]-probe.r.z);			// r2-r1

		d = probe.c.x*shrArr[i+3*stride]+
			probe.c.y*shrArr[i+4*stride]+
			probe.c.z*shrArr[i+5*stride];								// (c2*c1)

		if ( 1-abs(d) > epsilon) {	// not parallel:
			// solving linear system
			// obtained from eq: (r2-r1)+(c2*t2-c1*t1) = c1xc2*f
			// xc1:	1*t1-c1c2*t2 = (r2-r1)*c1
			// xc2:	c1c2*t1-1*t2 = (r2-r1)*c2
			// det = (c1c2)^2-1
			// det1 = (r2-r1)*((c1c2)*c2-c1); t1=det1/det
			// det2 = (r2-r1)*(c2-(c1c2)*c1); t2=det2/det

			a = make_float3(d*shrArr[i+3*stride]-probe.c.x,
							d*shrArr[i+4*stride]-probe.c.y,
							d*shrArr[i+5*stride]-probe.c.z);			// (c1*c2)c2-c1
			t[0] = a.x*b.x+a.y*b.y+a.z*b.z;								// det1

			a = make_float3(shrArr[i+3*stride]-d*probe.c.x,
							shrArr[i+4*stride]-d*probe.c.y,
							shrArr[i+5*stride]-d*probe.c.z);			// c2-(c1*c2)c1
			t[1] = a.x*b.x+a.y*b.y+a.z*b.z;								// det2
			t[0] /= d*d-1;
			t[1] /= d*d-1;

		} else { // parallel
			// r2+c2*t2 = r2 - c2*(r2-r1) +c1*t1
			t[0] = copysignf(probe.l/2,dotProd(b,probe.c));
			t[1] = -1*(b.x*shrArr[i+3*stride]+b.x*shrArr[i+4*stride]+b.x*shrArr[i+5*stride]) + d *t[0];
		}

		bool flag1 = (abs(t[0])<=(probe.l/2			+probe.a			+epsilon));
		bool flag2 = (abs(t[1])<=(shrArr[i+6*stride]/2+shrArr[i+7*stride]	+epsilon));

		if  ( !(flag1&&flag2) ) {
			// if at least one of the endpoints is out of limits - change the point that is further from the end
			bool flag3 = (abs(t[0])-probe.l/2)<(abs(t[1])-shrArr[i+6*stride]/2);
			t[flag3] = copysignf( (flag3 ? shrArr[i+6*stride]/2 : probe.l/2) , t[flag3] );

			a.x = flag3 ? probe.c.x : shrArr[i+3*stride];
			a.y = flag3 ? probe.c.y : shrArr[i+4*stride];
			a.z = flag3 ? probe.c.z : shrArr[i+5*stride];

			t[!flag3] = (2*flag3-1)*dotProd(b,a) + d*t[flag3];

			float len = flag3 ? probe.l/2 : shrArr[i+6*stride]/2;
			if ( abs(t[!flag3]) > len )	t[!flag3] = copysignf( len, t[!flag3] );
		}
		// distance between the points:
		a.x = probe.r.x + t[0]*probe.c.x - shrArr[i+0*stride] - t[1]*shrArr[i+3*stride];
		a.y = probe.r.y + t[0]*probe.c.y - shrArr[i+1*stride] - t[1]*shrArr[i+4*stride];
		a.z = probe.r.z + t[0]*probe.c.z - shrArr[i+2*stride] - t[1]*shrArr[i+5*stride];

		d = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
			
		return (d<((shrArr[i+7*stride]+probe.a)*scf+sep +epsilon));
	}
}

inline  __device__ bool cnt_intersec_shr(CNT_t probe, float *shrArr ,int stride, int i, float epsilon, float sep)  {
	return cnt_intersec_shr(probe, shrArr ,stride, i, epsilon, sep, 1.0);
}
inline  __device__ float cnt_minDist(float3 *Dtt, CNT_t probe, float *shrArr ,int stride, int i, float epsilon)  {

	float d;
	// find common perpendicular:
		float t[2] = {0,0};
			float3 a,b = make_float3(shrArr[i+0*stride]-probe.r.x,
									 shrArr[i+1*stride]-probe.r.y,
									 shrArr[i+2*stride]-probe.r.z);			// r2-r1

			d = probe.c.x*shrArr[i+3*stride]+
				probe.c.y*shrArr[i+4*stride]+
				probe.c.z*shrArr[i+5*stride];								// (c2*c1)

			if ( 1-abs(d) > epsilon) {	// not parallel:
				// solving linear system
				// obtained from eq: (r2-r1)+(c2*t2-c1*t1) = c1xc2*f
				// xc1:	1*t1-c1c2*t2 = (r2-r1)*c1
				// xc2:	c1c2*t1-1*t2 = (r2-r1)*c2
				// det = (c1c2)^2-1
				// det1 = (r2-r1)*((c1c2)*c2-c1); t1=det1/det
				// det2 = (r2-r1)*(c2-(c1c2)*c1); t2=det2/det

				a = make_float3(d*shrArr[i+3*stride]-probe.c.x,
								d*shrArr[i+4*stride]-probe.c.y,
								d*shrArr[i+5*stride]-probe.c.z);			// (c1*c2)c2-c1
				t[0] = a.x*b.x+a.y*b.y+a.z*b.z;								// det1

				a = make_float3(shrArr[i+3*stride]-d*probe.c.x,
								shrArr[i+4*stride]-d*probe.c.y,
								shrArr[i+5*stride]-d*probe.c.z);			// c2-(c1*c2)c1
				t[1] = a.x*b.x+a.y*b.y+a.z*b.z;								// det2
				t[0] /= d*d-1;
				t[1] /= d*d-1;

			} else { // parallel
				// r2+c2*t2 = r2 - (c2*(r2-r1))*c2 +c1*t1
				t[0] = copysignf(probe.l/2,dotProd(b,probe.c));
				t[1] = -1*(b.x*shrArr[i+3*stride]+b.x*shrArr[i+4*stride]+b.x*shrArr[i+5*stride]) + d *t[0];
			}

			Dtt->y = t[0];
			Dtt->z = t[1];

			bool flag1 = (abs(t[0])<=(probe.l/2			+probe.a			+epsilon));
			bool flag2 = (abs(t[1])<=(shrArr[i+6*stride]/2+shrArr[i+7*stride]	+epsilon));

			if  ( !(flag1&&flag2) ) {

				bool flag3 = (abs(t[0])-probe.l/2)<(abs(t[1])-shrArr[i+6*stride]/2);

				t[flag3] = copysignf( (flag3 ? shrArr[i+6*stride]/2 : probe.l/2) , t[flag3] );

				a.x = flag3 ? probe.c.x : shrArr[i+3*stride];
				a.y = flag3 ? probe.c.y : shrArr[i+4*stride];
				a.z = flag3 ? probe.c.z : shrArr[i+5*stride];

				t[!flag3] = (2*flag3-1)*dotProd(b,a) + d*t[flag3];

				float len = flag3 ? probe.l/2 : shrArr[i+6*stride]/2;
				if ( abs(t[!flag3]) > len )	t[!flag3] = copysignf( len, t[!flag3] );
			}

			// distance between the points:
			a.x = probe.r.x + t[0]*probe.c.x - shrArr[i+0*stride] - t[1]*shrArr[i+3*stride];
			a.y = probe.r.y + t[0]*probe.c.y - shrArr[i+1*stride] - t[1]*shrArr[i+4*stride];
			a.z = probe.r.z + t[0]*probe.c.z - shrArr[i+2*stride] - t[1]*shrArr[i+5*stride];

			d = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);

			// distance along common perpendicular:
			a.x = probe.r.x + Dtt->y*probe.c.x - shrArr[i+0*stride] - Dtt->z*shrArr[i+3*stride];
			a.y = probe.r.y + Dtt->y*probe.c.y - shrArr[i+1*stride] - Dtt->z*shrArr[i+4*stride];
			a.z = probe.r.z + Dtt->y*probe.c.z - shrArr[i+2*stride] - Dtt->z*shrArr[i+5*stride];

			Dtt->x = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);

	return d;
}
