//====================================================================================================================
//										 <<< Mutual position based statistic kernel >>>
//====================================================================================================================
#pragma once
#include "CNT.h"
#include "simPrms.h"
#include "intersection_dev.h"
#include "memoryIO.h"
#include "vectorMath.h"
#include "ellipse_fit.h"

#define	SRC_ELEC	UINT_MAX
//#define	SRC_ELEC	500000
#define	SNK_ELEC	SRC_ELEC-1

// check for boundary cell:
inline  __device__ int get_dmnIndex(	unsigned int	selfID,
                                        		 int	bufLayer ) {
                                        
	int pos = 0;

	if ( (peri_dir&1)==0 ) { pos = (selfID%grdExt_cond[0])					- bufLayer; }
	if ( (peri_dir&2)==0 ) { pos = (selfID/grdExt_cond[0])%grdExt_cond[1]	- bufLayer; }
	if ( (peri_dir&4)==0 ) { pos = (selfID/grdExt_cond[0])/grdExt_cond[1]	- bufLayer; }

	int dmnID = selfID;
	if (pos < 0) { dmnID = -2; }
	if (pos >= (grdExt_cond[((7-peri_dir)>>1)]-2*bufLayer) ) { dmnID = -1; }

	return dmnID;
}

inline  __device__ bool is_internal(	unsigned int	selfID,
										unsigned int	bufLayer ) {
	return ( get_dmnIndex(selfID,bufLayer)>=0 );
}

// interinclusion contacts - full:
extern "C"
__global__ void contact_count_kernel_full(	int				count,
											char3			stride,
											int*			d_dmnAddr,
											float*			d_coords,
											unsigned int*	d_individ_counts)
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
// neigbour cell:
__shared__ float3		neigDspl;	// stores (-1,0,1) per each coordinate to account for periodic boundary conditions
__shared__ int			neigID;		// the global ID of  neighbouring cell
__shared__ int			dmnAddr_startNeig;	// start position of neighbouring block
__shared__ unsigned int neigCNT;	// the number of cnts in a neibouring cell
// cashed particle coords
__shared__ float shrArr[9*BLOCK];	// inclusion coordinates

if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
	__syncthreads();
	if ( selfID < grdExt_size ) {
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnAddr[selfID]-dmnAddr_startSelf;
		}
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf, d_coords,glbPtchSz_cond);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );

		shrClean(shrArr);

		// get cnts from neighbouring cell:
		if (threadIdx.x == 0 ) {
			neigID = stride2glbID_directed(peri_dir,stride, selfID, grdExt_cond);
			if (neigID>-1) {
				char3 shift  = stride2dspl(stride,selfID, grdExt_cond);
				dmnAddr_startNeig = (neigID>0?d_dmnAddr[neigID-1]:0);
				neigCNT =  d_dmnAddr[neigID]-dmnAddr_startNeig;
				neigDspl = make_float3(	shift.x*phsDim_cond[0],
										shift.y*phsDim_cond[1],
										shift.z*phsDim_cond[2]);
			}
		}
		__threadfence_block();
		__syncthreads();

		if (neigID>-1) {
			GlbRead(neigCNT, shrArr, dmnAddr_startNeig, d_coords,glbPtchSz_cond);
			tarnsCNT(neigCNT, neigDspl, shrArr);

			// MAIN CODE:-----------------------------------------------------------------
			int contactNum=0;
			// check distance:
			if (threadIdx.x < selfCNT)
			 	for(int i=0;i<neigCNT;i++)
			 		if ( (selfID!=neigID)||(i!=threadIdx.x) ) // not the same inclusion
			 			if (cnt_intersec_shr(probe,shrArr,blockDim.x, i,cond_precision,range_lim)) contactNum++;
			__threadfence_block();
			__syncthreads();

			if (threadIdx.x < selfCNT)	d_individ_counts[dmnAddr_startSelf+threadIdx.x] += contactNum;
		}
	}
}

// elimination:
extern "C"
__global__ void reductKernel(	int		count,
								int		threshold,
								unsigned int bufLayer,
								int*	d_dmnAddr,
								short*	d_Occ,
								float*	d_coords,
								unsigned int*	d_conts )
{
// current cell:
__shared__ unsigned int selfID;		// Block Idx + count*GridDim - glob
__shared__ bool 		intFlg;		// flag whether cell is internal (not from buffer layer)
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
__shared__ float shrArr[9*BLOCK];	// inclusion coordinates
__shared__ int dmnAddr_startSelf[1];	// start position of current block
__shared__ int shrRmnd[1];


	if (threadIdx.x == 0) {
		selfID = blockIdx.x + count*gridDim.x;
		intFlg = is_internal(selfID,bufLayer);
	}
	__syncthreads();
	if ((intFlg) && (selfID < grdExt_size)) {
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			dmnAddr_startSelf[0] = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnAddr[selfID]-dmnAddr_startSelf[0];
		} 
		__threadfence_block();
		__syncthreads();
		CNT_t probe;
		GlbRead(selfCNT, shrArr, dmnAddr_startSelf[0], d_coords,glbPtchSz_cond);
		probe = ( threadIdx.x < selfCNT ? shrID2regCNT(threadIdx.x,shrArr) : make_emptyCNT() );

		if ((threadIdx.x < selfCNT)&&(d_conts[dmnAddr_startSelf[0]+threadIdx.x] < threshold))	probe =  make_emptyCNT();

		__syncthreads();
		
		
		shrClean(shrArr);
		GlbClean(selfCNT, shrArr, dmnAddr_startSelf[0], d_coords, glbPtchSz_cond);
		
		// WRITE RESULTS TO GLOBAL MEMORY:
		if (selfCNT>0)
			reWriteGlbMem(selfCNT, probe, shrRmnd, dmnAddr_startSelf, shrArr, d_coords, glbPtchSz_cond);
	
		// write created CNTs per domain cell number:
		if ((threadIdx.x == 0)&&(selfCNT > 0))
			d_Occ[selfID] = (short) shrRmnd[0];
		
	}
}
//========================================================================================================
// Internal contacts:
// Distance between inclusions axii (analogous to cnt_intersec_shr from intersection_dev.h - later replace into one):
__device__ float3 dist_to_inc( CNT_t probe, float *shrArr ,int stride, int i)  {

	// find common perpendicular:
		float t[2] = {0,0};
			float3 a,b = make_float3(shrArr[i+0*stride]-probe.r.x,
									 shrArr[i+1*stride]-probe.r.y,
									 shrArr[i+2*stride]-probe.r.z);			// r2-r1

			float d = probe.c.x*shrArr[i+3*stride]+
				probe.c.y*shrArr[i+4*stride]+
				probe.c.z*shrArr[i+5*stride];								// (c2*c1)

			if ( 1-abs(d) > cond_precision) {	// not parallel:
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
				// r2+c2*t2 = r2 - c2*((r2-r1)*c2) +c1*t1
					t[0] = copysignf(probe.l/2,dotProd(b,probe.c));
					t[1] = -1*(b.x*shrArr[i+3*stride]+b.x*shrArr[i+4*stride]+b.x*shrArr[i+5*stride]) + d *t[0];
			}

			bool flag1 = (abs(t[0])<=(probe.l/2			+probe.a			+cond_precision));
			bool flag2 = (abs(t[1])<=(shrArr[i+6*stride]/2+shrArr[i+7*stride]	+cond_precision));

			if  ( !(flag1&&flag2) ) {
				// if at least one of the end-points is out of limits - change the point that is further from the end
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

			float3 Dtt = make_float3(-1,0,0);
			Dtt.x = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
			Dtt.y = t[0];
			Dtt.z = t[1];

	return  Dtt;
}
// mark inclusions by belonging to the electrode:
extern "C"
__global__ void		mark_elec_inclusions(	int		count,
											int		bufLayer,
											int*	d_dmnAddr,
											char*	d_incFlg	)
{
// current cell:
__shared__ int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell
__shared__ int selfFLG;

	// get starting address of this cell inclusions
	// also mark cell as electrode or internal (selfFLG):
	if (threadIdx.x == 0)	{
		// default values for a cell not to be processed:
		dmnAddr_startSelf = -1;
		selfCNT = 0;
		selfFLG = 0;

		selfID = blockIdx.x + count*gridDim.x;
		if ( ( selfID < grdExt_size ) ) {
			dmnAddr_startSelf = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnAddr[selfID]-dmnAddr_startSelf;
			selfFLG = get_dmnIndex(selfID,bufLayer);
		}
	}
	__threadfence_block();
	__syncthreads();

	// mark inclusion as belonging to electrode or conducting net
	if (threadIdx.x < selfCNT )	{
		int pidx = dmnAddr_startSelf + threadIdx.x;
		d_incFlg[pidx] = ( selfFLG>=0 ? 0 : ( selfFLG==-1 ?  -1 : -2 ) );
	}
}

// replace inclusion indices in contact tuples with electrode flags:
extern "C"
__global__ void		mark_elec_contacts(		unsigned int	arr_size,
											unsigned int*	d_contact_Idx,
											char*			d_incFlg 		) {

	unsigned int	expIdx = threadIdx.x+blockIdx.x*blockDim.x;
	if (expIdx < arr_size) {
		unsigned int	incIdx = d_contact_Idx[expIdx];
		d_contact_Idx[expIdx] = ( d_incFlg[incIdx]==0 ? incIdx : ( d_incFlg[incIdx]==-1 ? SNK_ELEC : SRC_ELEC ) );
	}

}
// Generate virtual inclusions, representing contacts:
extern "C"
__global__ void		generate_virt_inclusions(	unsigned int	num_contacts,
												unsigned int*	d_idxI,
												unsigned int*	d_idxJ,
												unsigned int*	d_vOcc,
												float*			d_vInc,
												float*			d_result	) {

__shared__ unsigned int		selfCNT;
__shared__ unsigned int		glbAddr;
__shared__ float 	shrArr[9*BLOCK];	// inclusion coordinates

	CNT_t	virt_inc	= make_emptyCNT();

	unsigned int selfIDX = threadIdx.x+blockIdx.x*blockDim.x;
	if ((threadIdx.x==0)&&(selfIDX<num_contacts))	{
		selfCNT = ( (num_contacts - selfIDX)>blockDim.x  ? blockDim.x : (num_contacts - selfIDX) );
		glbAddr = selfIDX;
	}
	__threadfence_block();
	__syncthreads();

	if (selfIDX<num_contacts)	{
		unsigned int	inc_I = d_idxI[selfIDX];
		unsigned int	inc_J = d_idxJ[selfIDX];

		for(char i=0; i<9; i++)
			shrArr[threadIdx.x+blockDim.x*i]	= d_result[inc_I+glbPtchSz_cond*i];

		CNT_t probe = shrID2regCNT(threadIdx.x,shrArr);

		for(char i=0; i<9; i++)
			shrArr[threadIdx.x+blockDim.x*i]	= d_result[inc_J+glbPtchSz_cond*i];


		// checking for contact over periodic boundary:
		float3 d = make_float3(	(probe.r.x-shrArr[threadIdx.x+blockDim.x*0]),
								(probe.r.y-shrArr[threadIdx.x+blockDim.x*1]),
								(probe.r.z-shrArr[threadIdx.x+blockDim.x*2]) );
		char3 disp = make_char3(	(peri_dir&&1 ? ( d.x>0.5*phsDim_cond[0] ? -1 : (d.x<-0.5*phsDim_cond[0] ? 1 : 0) ) : 0),
									(peri_dir&&2 ? ( d.y>0.5*phsDim_cond[1] ? -1 : (d.y<-0.5*phsDim_cond[1] ? 1 : 0) ) : 0),
									(peri_dir&&4 ? ( d.z>0.5*phsDim_cond[2] ? -1 : (d.z<-0.5*phsDim_cond[2] ? 1 : 0) ) : 0));
		probe.r.x += disp.x*phsDim_cond[0];
		probe.r.y += disp.y*phsDim_cond[1];
		probe.r.z += disp.z*phsDim_cond[2];

		float3 Dtt = dist_to_inc( probe, shrArr , blockDim.x, threadIdx.x);
		//if (Dtt.x<=shrArr[threadIdx.x+blockDim.x*7] + probe.a + range_lim + 2*cond_precision) {
			float3	pnt0	=	probe.r + probe.c*Dtt.y;
			float3	pnt1	=	make_float3(	shrArr[threadIdx.x+blockDim.x*0] + Dtt.z*shrArr[threadIdx.x+blockDim.x*3],
												shrArr[threadIdx.x+blockDim.x*1] + Dtt.z*shrArr[threadIdx.x+blockDim.x*4],
												shrArr[threadIdx.x+blockDim.x*2] + Dtt.z*shrArr[threadIdx.x+blockDim.x*5]);

			virt_inc.r	= (pnt0 + pnt1)*0.5;
			virt_inc.c	= pnt1-pnt0;
			virt_inc.l	= norm3(virt_inc.c);
			virt_inc.c = virt_inc.c*(1.0/virt_inc.l);
			virt_inc.a	= min(probe.a,shrArr[threadIdx.x+blockDim.x*7]);
			virt_inc.k	= shrArr[threadIdx.x+blockDim.x*8];
		//}
	}
	__threadfence_block();
	__syncthreads();

	shrClean(shrArr);
	reg2shrCNT(virt_inc, shrArr);
	GlbWrite(selfCNT, shrArr, glbAddr, d_vInc, num_contacts);

	// calculating occupancies:
	// global atomic version:
	if ( (virt_inc.k>=0) && (virt_inc.k<grdExt_size) )
		atomicAdd(&(d_vOcc[(unsigned int) virt_inc.k]),1);

}
// verify sorted structure:
extern "C"
__global__ void		count_incInCell(	int				count,
										unsigned int	offset,
										float*			d_virt_inc,
										unsigned int*	d_virtAddr,
										unsigned int*	d_testOcc,
										unsigned int*	d_testErr	) {
__shared__ int selfID;		// Block Idx + count*GridDim - glob
__shared__ int dmnAddr_startSelf;	// start position of current block
__shared__ unsigned int selfCNT;	// the number of cnts in current cell

	// get starting address of this cell inclusions
	if (threadIdx.x == 0)	{
		// default values for a cell not to be processed:
		dmnAddr_startSelf = -1;
		selfCNT = 0;

		selfID = blockIdx.x + count*gridDim.x;
		if ( ( selfID < grdExt_size ) ) {
			dmnAddr_startSelf = ( selfID>0 ? d_virtAddr[selfID-1] : 0 );
			selfCNT = d_virtAddr[selfID]-dmnAddr_startSelf;
		}
	}
	__threadfence_block();
	__syncthreads();

	// mark inclusion as belonging to electrode or conducting net
	int N_probe = (int) ceil((float)selfCNT/blockDim.x);
	for(int i=threadIdx.x*N_probe; i<(threadIdx.x+1)*N_probe; i++) {
		if (i < selfCNT )	{
			int pidx = dmnAddr_startSelf + i;
			int	probe_k = (int) floor(d_virt_inc[pidx]+0.5);
			if (probe_k == selfID) {
				atomicAdd(&(d_testOcc[selfID]),1);
			} else {
				atomicAdd(&(d_testErr[selfID]),1);
			}
		}
	}
}
