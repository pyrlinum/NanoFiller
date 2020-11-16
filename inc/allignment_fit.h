#pragma once
#include "statDevice_functions.h"

//-------------------------------------------------------------------------------------------------
// Eigen values and eigen vectors:
//-------------------------------------------------------------------------------------------------
inline __device__ void solve_eqn3_vieted(float  solution[3], float a, float b, float c) {
// cubic equation x^3+a*x^2+b*x+c=0

	for(char i=0;i<3;i++)
		solution[i] = 0.0;
	float det = -4*pow(a,3)*c+pow(a*b,2)-4*pow(b,3)+18*a*b*c-27*c*c;
	float Q = (a*a-3*b)/9;
	float R = (2*pow(a,3)-9*a*b+27*c)/54;
	float S = pow(Q,3)-R*R;
	if (S>1.0E-6) {
		float phi = acos(R/sqrt(pow(Q,3)))/3;
		solution[0] = -2*sqrt(Q)*cos(phi)-a/3;
		solution[1] = -2*sqrt(Q)*cos(phi+2*PI/3)-a/3;
		solution[2] = -2*sqrt(Q)*cos(phi-2*PI/3)-a/3;
	} else if (S>-1.0E-6) {
		if (abs(R)>1.0E-6) {
			solution[0] = -2*( R>0 ? 1 : (R<0?-1:0) )*powf(abs(R),1.0/3.0)-a/3;
			solution[1] =    ( R>0 ? 1 : (R<0?-1:0) )*powf(abs(R),1.0/3.0)-a/3;
		} else solution[0] =  -a/3;
	} else {
		// considering imaginary part small
		float phi;
		float csh;
		if(Q>1.0E-6)	{
			phi = acoshf(abs(R)/powf(Q,1.5))/3;
			csh = coshf(phi);
		} else {
			phi = asinhf(abs(R)/powf(Q,1.5))/3;
			csh = sinhf(phi);
		}
		solution[0] = -2*sqrt(abs(Q))*csh-a/3;
		solution[1] = -2*sqrt(Q)*csh-a/3;
		solution[2] = -2*sqrt(Q)*csh-a/3;
	}
}
inline __device__ void eigenvals(float solution[3], float  A[3][3]) {
	// Eigen equation: L^3-tr(A)*L^2-1/2*(tr(A^2)-tr(A)^2)*L-det(A)=0;
	float a = -(A[0][0]+A[1][1]+A[2][2]);
	float c = -MatrDet3x3(A);
	float  A2[3][3];
	for(char i=0;i<3;i++)
		for(char j=0;j<3;j++)
			A2[i][j] = A[i][j];

	MatProd3x3(A2,A2);
	float  b = -0.5*((A2[0][0]+A2[1][1]+A2[2][2])-a*a);
	solve_eqn3_vieted(solution,a,b,c);

	// sort eigenvalues:
	float  buf;
	if (abs(solution[1])>abs(solution[0]))	{
		buf = solution[0];
		solution[0] = solution[1];
		solution[1] = buf;
	}
	if (abs(solution[2])>abs(solution[1]))	{
		buf = solution[2];
		solution[2] = solution[1];

		if (abs(buf)>abs(solution[0]))	{
			solution[1] = solution[0];
			solution[0] = buf;
		} else solution[1] = buf;
	}		  
}
inline __device__ char eigenvec(float vec[3][3], float l, float A[3][3]) {
	float Al[3][3];
	for(char I=0;I<3;I++)
		for(char J=0;J<3;J++)
			Al[I][J] = A[I][J] - ( J==I ? l : 0);

	// find rank:
	ForwardElim(Al);

	// normalise coefficients:
	{
		float max = 0;
	for(char I=0;I<3;I++)
		for(char J=0;J<3;J++)
			if (abs(Al[I][J]) > max) max = abs(Al[I][J]);
	if (max>1.0E-3)
		for(char I=0;I<3;I++)
			for(char J=0;J<3;J++)
				Al[I][J] /= max;
	}

	for(char I=0;I<3;I++)	{
		float maxI = 0;
		for(char J=0;J<3;J++)
			if (abs(Al[I][J]) > maxI) maxI = abs(Al[I][J]);
		if(maxI>1.0E-3) {
			for(char J=0;J<3;J++)
				if (abs(Al[I][J])/maxI < 1.0E-2) Al[I][J] = 0;
		} else 
			for(char J=0;J<3;J++)
				Al[I][J] = 0;
	}
	char rank = 0;
	for(int I=0;I<3;I++) {
		if ( (abs(Al[I][0])>=.5E-2) || (abs(Al[I][1])>=.5E-2) || (abs(Al[I][2])>=.5E-2) ) rank += 1;
		}

	// find eigenvectors:
	if(rank==3) {
		vec[0][0] = 0; vec[0][1] = 0; vec[0][2] = 0;
		vec[1][0] = 0; vec[1][1] = 0; vec[1][2] = 0;
		vec[2][0] = 0; vec[2][1] = 0; vec[2][2] = 0;
	} if(rank==2) {
		if(abs(Al[0][0])<=.5E-3) {
			vec[0][0] = 1;
			vec[0][1] = 0;
			vec[0][2] = 0;
		} else {
			if(Al[1][1] != 0) {
				vec[0][2] = 1;
				vec[0][1] = -Al[1][2]/Al[1][1];
			} else {
				vec[0][2] = 0;
				vec[0][1] = 1;
			}
			vec[0][0] = -(Al[0][1]*vec[0][1]+Al[0][2]*vec[0][2])/Al[0][0];
		}
	} else if (rank==1) {
		char I=0;
		char J=0;
		while ((abs(Al[I][0])<=.5E-3)&&(abs(Al[I][1])<=.5E-3)&&(abs(Al[I][2])<=.5E-3)&&(I<2)) I++;
		while ((abs(Al[I][J])<=.5E-3)&&(J<2)) J++;
			vec[0][(J+1)%3] = 1;
			vec[0][(J+2)%3] = 0;
			vec[0][J] = -(Al[I][(J+1)%3]*vec[0][(J+1)%3]+Al[I][(J+2)%3]*vec[0][(J+2)%3])/Al[I][J];

			vec[1][(J+1)%3] = 0;
			vec[1][(J+2)%3] = 1;
			vec[1][J] = -(Al[I][(J+1)%3]*vec[0][(J+1)%3]+Al[I][(J+2)%3]*vec[0][(J+2)%3])/Al[I][J];
	} else if (rank==0) {
		vec[0][0] = 1; vec[0][1] = 0; vec[0][2] = 0;
		vec[1][0] = 0; vec[1][1] = 1; vec[1][2] = 0;
		vec[2][0] = 0; vec[2][1] = 0; vec[2][2] = 1;
	}
	

	return 3-rank;		
 }
//=============================================================================================

// Kernels to control sponatneous allignment of aspherical inclusions
// 01. cross matrix collection:
__global__ void crossMtxBuild(	int		count,
								int		*d_dmnAddr,
								short	*d_dmnOcc,
								float	*d_incCoords,
								float	*d_CrossMtx)			// correlator matrix components 11,22,33,12,13,23
{	
	__shared__ int		selfID;					//	current cell ID
	__shared__ int		startAddr_self;			//	starting adress to read
	__shared__ int		selfCNT;				//	number of is in current cell to check
	__shared__ int		grdSize;

	// Results:
	extern __shared__ volatile float dynCrossMtx[];


	if (threadIdx.x == 0)
		selfID = blockIdx.x + count*gridDim.x;
		grdSize = dmnExt[0]*dmnExt[1]*dmnExt[2];
	__syncthreads();

	if (selfID < dmnExt[0]*dmnExt[1]*dmnExt[2]) {
	
		// get cnts from current cell:
		if (threadIdx.x == 0) {
			startAddr_self = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
		} 
		__syncthreads();

		for(char i=0;i<6;i++) dynCrossMtx[threadIdx.x+i*blockDim.x] = 0.0f;
		__syncthreads();

			
		//Orientation moment accumulation:
		float3 incVec = make_float3(0,0,0);	
		if (threadIdx.x < selfCNT) {
			incVec.x = d_incCoords[startAddr_self + threadIdx.x + 3*numCNT];
			incVec.y = d_incCoords[startAddr_self + threadIdx.x + 4*numCNT];
			incVec.z = d_incCoords[startAddr_self + threadIdx.x + 5*numCNT];
		
			dynCrossMtx[threadIdx.x+0*blockDim.x] = incVec.x*incVec.x;
			dynCrossMtx[threadIdx.x+1*blockDim.x] = incVec.y*incVec.y;
			dynCrossMtx[threadIdx.x+2*blockDim.x] = incVec.z*incVec.z;
			dynCrossMtx[threadIdx.x+3*blockDim.x] = incVec.x*incVec.y;
			dynCrossMtx[threadIdx.x+4*blockDim.x] = incVec.x*incVec.z;
			dynCrossMtx[threadIdx.x+5*blockDim.x] = incVec.y*incVec.z;
		}
		__syncthreads();

		for(char i=0;i<6; i++)
			reduce(blockDim.x,&dynCrossMtx[i*blockDim.x]);

		if(threadIdx.x==0) 
			for(char i=0;i<6; i++)
				d_CrossMtx[selfID+i*grdSize] = dynCrossMtx[i*blockDim.x]/selfCNT;
	}			
}

// 02. orientation determination:

__global__ void resetOrt(	float	*d_CrossMtx,
							float	*d_phiOrt,
							float	*d_thetaOrt)			// correlator matrix components 11,22,33,12,13,23
{	
	int cellID = threadIdx.x+blockIdx.x*blockDim.x;
	int grdSize = dmnExt[0]*dmnExt[1]*dmnExt[2];	

	if( cellID < grdSize)	{

		float M[3][3];
		M[0][0] = d_CrossMtx[cellID+0*grdSize];
		M[1][1] = d_CrossMtx[cellID+1*grdSize];
		M[2][2] = d_CrossMtx[cellID+2*grdSize];

		M[0][1] = d_CrossMtx[cellID+3*grdSize];
		M[0][2] = d_CrossMtx[cellID+4*grdSize];
		M[1][2] = d_CrossMtx[cellID+5*grdSize];

		M[1][0] = M[0][1];
		M[2][0] = M[0][2];
		M[2][1] = M[1][2];

		float eVals[3];
		float	eVecs[3][3];
		for(char j=0;j<3;j++) {
			eVals[j] = 0;
			for(char k=0;k<3;k++)
				eVecs[j][k] = 0;
		}

		float val[3];
		eigenvals(val, M);
				
		char nVec = 0;
		char deg = 0;
		for(char i=0;i<3;i++)
			if(nVec<3) {
				float vec[3][3];
				deg = eigenvec(vec, val[i], M);
				if(deg>0) {
					for(char j=0;j<deg;j++) {
						eVals[nVec+j] = val[i]; 
						for(char k=0;k<3;k++)
							eVecs[nVec+j][k] = vec[j][k];
						nVec++;
					}
				}
			}
		
			if (nVec == 3) {
				char nOrt;
				if (abs((eVals[1]-eVals[2])/eVals[1]) < 0.1)	{	nOrt = 0; }	// use the largest eigenvalue (cone-like orientation)
				else											{	nOrt = 2; }	// use the smallest eigenvalue (disk-like orientation)

				float theta = ( abs(eVecs[nOrt][2])<=1-float_prec ? acosf(eVecs[nOrt][2]) : (sign(eVecs[nOrt][2]) > 0 ? 0 : PI));
				float cosphi = (abs(sinf(theta))>float_prec ? eVecs[nOrt][0]/sinf(theta) : 1);
				d_thetaOrt[cellID] = theta;
				d_phiOrt[cellID] = ( abs(cosphi)<=1-float_prec ? acosf(cosphi) : (sign(cosphi) > 0 ? 0 : PI) );	
			} 
	}
}

// 03. avarage theta & phi:
__global__ void resetPhiTheta(	int		count,
								int		*d_dmnAddr,
								short	*d_dmnOcc,
								float	*d_incCoords,
								float	*d_phiOrt,
								float	*d_thetaOrt,
								float	*d_phiAvg,
								float	*d_thetaAvg,
								float	*d_phiDev,
								float	*d_thetaDev)	{

	__shared__ int		selfID;					//	current cell ID
	__shared__ int		startAddr_self;			//	starting adress to read
	__shared__ int		selfCNT;				//	number of is in current cell to check

	__shared__ float	ortPhi;
	__shared__ float	ortThe;

	extern __shared__ volatile float pool[];
	float* avgPhi = (float*) &pool[0*blockDim.x];
	float* avgThe = (float*) &pool[1*blockDim.x];
	float* devPhi = (float*) &pool[2*blockDim.x];
	float* devThe = (float*) &pool[3*blockDim.x];

	if (threadIdx.x == 0)	{
		selfID = blockIdx.x + count*gridDim.x;
		if (selfID < dmnExt[0]*dmnExt[1]*dmnExt[2]) {
			startAddr_self = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
			ortPhi = d_phiOrt[selfID];
			ortThe = d_thetaOrt[selfID];
		}}
	__syncthreads();

	if (selfID < dmnExt[0]*dmnExt[1]*dmnExt[2]) {

		avgPhi[threadIdx.x] = 0;
		avgThe[threadIdx.x] = 0;
		devPhi[threadIdx.x] = 0;
		devThe[threadIdx.x] = 0;
		__syncthreads();

		//Orientation moment accumulation:
		if (threadIdx.x < selfCNT) {
			float3 incVec = make_float3(0,0,0);
			incVec.x = d_incCoords[startAddr_self + threadIdx.x + 3*numCNT];
			incVec.y = d_incCoords[startAddr_self + threadIdx.x + 4*numCNT];
			incVec.z = d_incCoords[startAddr_self + threadIdx.x + 5*numCNT];

			rot_Z(ortPhi,incVec);
			rot_Y(-ortThe,incVec);

			float2 theta_phi = Ort2Angle(incVec);
		
			avgPhi[threadIdx.x] = theta_phi.y;
			avgThe[threadIdx.x] = theta_phi.x;
			devPhi[threadIdx.x] = theta_phi.y*theta_phi.y;
			devThe[threadIdx.x] = theta_phi.x*theta_phi.x;
		}
		__syncthreads();

		reduce(blockDim.x,avgPhi);
		reduce(blockDim.x,avgThe);
		reduce(blockDim.x,devPhi);
		reduce(blockDim.x,devThe);


		if (threadIdx.x == 0)	{
			d_phiAvg[selfID]	= avgPhi[0]/selfCNT;
			d_thetaAvg[selfID]	= avgThe[0]/selfCNT;
			d_phiDev[selfID]	= (devPhi[0]-avgPhi[0]*avgPhi[0]/selfCNT)/selfCNT;
			d_thetaDev[selfID]	= (devThe[0]-avgThe[0]*avgThe[0]/selfCNT)/selfCNT;
		}
	}
}
// 03. avarage vector:
__global__ void resetAvgVec(	int		count,
								int		*d_dmnAddr,
								short	*d_dmnOcc,
								float	*d_incCoords,
								float	*d_phiOrt,
								float	*d_thetaOrt,
								float	*d_phiAvg,
								float	*d_thetaAvg,
								float	*d_phiDev,
								float	*d_thetaDev)	{

	__shared__ int		selfID;					//	current cell ID
	__shared__ int		startAddr_self;			//	starting adress to read
	__shared__ int		selfCNT;				//	number of is in current cell to check

	__shared__ float	ortPhi;
	__shared__ float	ortThe;

	__shared__ bool		ortFlag;
	__shared__ float	phi;
	__shared__ float	theta;

	extern __shared__ volatile float pool[];

	float* avgX = (float*) &pool[0*blockDim.x];
	float* avgY = (float*) &pool[1*blockDim.x];
	float* avgZ = (float*) &pool[2*blockDim.x];

	if (threadIdx.x == 0)	{
		selfID = blockIdx.x + count*gridDim.x;
		if (selfID < dmnExt[0]*dmnExt[1]*dmnExt[2]) {
			startAddr_self = ( selfID>0 ? d_dmnAddr[selfID-1] : 0 );
			selfCNT = d_dmnOcc[selfID];
			ortPhi = d_phiOrt[selfID];
			ortThe = d_thetaOrt[selfID];
		}}
	__syncthreads();

	if (selfID < dmnExt[0]*dmnExt[1]*dmnExt[2]) {

		avgX[threadIdx.x] = 0;
		avgY[threadIdx.x] = 0;
		avgZ[threadIdx.x] = 0;
		__syncthreads();

		//Center vector accumulation:
		float3 incVec = make_float3(0,0,0);

		if (threadIdx.x < selfCNT) {
			incVec.x = d_incCoords[startAddr_self + threadIdx.x + 3*numCNT];
			incVec.y = d_incCoords[startAddr_self + threadIdx.x + 4*numCNT];
			incVec.z = d_incCoords[startAddr_self + threadIdx.x + 5*numCNT];

			rot_Z(ortPhi,incVec);
			rot_Y(-ortThe,incVec);

			avgX[threadIdx.x] = incVec.x;
			avgY[threadIdx.x] = incVec.y;
			avgZ[threadIdx.x] = incVec.z;
		}
		__syncthreads();

		reduce(blockDim.x,avgX);
		reduce(blockDim.x,avgY);
		reduce(blockDim.x,avgZ);

		if (threadIdx.x == 0)	{
			float norm = sqrtf(avgX[0]*avgX[0]+avgY[0]*avgY[0]+avgZ[0]*avgZ[0]);
			ortFlag = (norm/selfCNT > incOrt_minNorm);

			if (ortFlag) {
				avgX[0] /= norm;
				avgY[0] /= norm;
				avgZ[0] /= norm;

				float normXY = sqrtf(avgX[0]*avgX[0]+avgY[0]*avgY[0]);
				theta = asinf(normXY);
				phi = (normXY > float_prec ? acosf(avgX[0]/normXY) : 0);
			}
		}
		__syncthreads();

		if(ortFlag) {
			// Dispersion calculation:
			if (threadIdx.x < selfCNT) {
				float normXY = sqrtf(incVec.x*incVec.x+incVec.y*incVec.y);
				float dt = asinf(normXY);
				float dp = (normXY > float_prec ? acosf(incVec.x/normXY) : 0);
		
				avgX[threadIdx.x] = (dp-phi)*(dp-phi);
				avgZ[threadIdx.x] = (dt-theta)*(dt-theta);
			}
			__syncthreads();

			reduce(blockDim.x,avgX);
			reduce(blockDim.x,avgZ);

			if (threadIdx.x == 0)	{
				d_phiAvg[selfID]	= phi;
				d_thetaAvg[selfID]	= theta;
				d_phiDev[selfID]	= avgX[0]/selfCNT;
				d_thetaDev[selfID]	= avgZ[0]/selfCNT;
			}
		}
	}
}

//=============================================================================================
// delete unlikely tubes:
//=============================================================================================
__device__ float check_probability(	CNT_t	probe,
									int		cell,
									int		vol,
									float	*shrAnglParam) {
	float3 incVec = probe.c;
	rot_Z( shrAnglParam[cell+0*vol],incVec);
	rot_Y(-shrAnglParam[cell+1*vol],incVec);
	float2 theta_phi = Ort2Angle(incVec);
	/*
	float Ftheta	= ( shrAnglParam[cell+5*vol]>0 ?
							expf(-powf((theta_phi.x-shrAnglParam[cell+3*vol])/shrAnglParam[cell+5*vol],2)/2)
							/(2*erf(Pi/2/shrAnglParam[cell+5*vol]/sqrtf(2.0f))) : 1/Pi );
	float Fphi		= ( shrAnglParam[cell+4*vol]>0 ?
							expf(-powf((theta_phi.y-shrAnglParam[cell+2*vol])/shrAnglParam[cell+4*vol],2)/2)
							/(2*erf(Pi/shrAnglParam[cell+4*vol]/sqrtf(2.0f))) : 0.5f/Pi );
							*/
	float Ftheta	= ( shrAnglParam[cell+5*vol]>0 ?
							expf(-powf((theta_phi.x-shrAnglParam[cell+3*vol])/shrAnglParam[cell+5*vol],2)/2)
						 : 1 );
	float Fphi		= ( shrAnglParam[cell+4*vol]>0 ?
							expf(-powf((theta_phi.y-shrAnglParam[cell+2*vol])/shrAnglParam[cell+4*vol],2)/2)
						 : 1 );
	return Fphi*Ftheta;
}
__global__ void randomThinOut(	int		*d_dmnAddr,
								short	*d_dmnOcc,
								int		*d_masks,
								float	*d_result,
								short	*d_Crtd,
								float	*d_phiOrt,
								float	*d_theOrt,
								float	*d_phiAvg,
								float	*d_theAvg,
								float	*d_phiDev,
								float	*d_theDev,
								int		*d_RNGseed)
{
	__shared__ int mask[6];
	__shared__ int vol;
	__shared__ unsigned int selfCNT;	// the number of cnts to be proccessed by this Block
	__shared__ float shrArr[9*BLOCK];
	extern __shared__ int dynShrInt[];


	// obtain segment boundaries:
	if (threadIdx.x == 0) {
		selfCNT = 0;
		vol = getMask(mask,d_masks);
	}
	__syncthreads();

	// set shared arrays: - 11*vol*sizeof(float)
	int	*devAddr = (int*)	&(dynShrInt[0*vol]);				// 2*vol*sizeof(float)
	int	*shrAddr = (int*)	&(dynShrInt[2*vol]);				// 2*vol*sizeof(float)
	int	*shrCrtd = (int*)	&(dynShrInt[4*vol]);				// 1*vol*sizeof(float)
	float	*shrAnglParam = (float*)	&(dynShrInt[5*vol]);	// 6*vol*sizeof(float)

	curandState	cuRNGstate;
	curand_init(d_RNGseed[blockIdx.x * blockDim.x + threadIdx.x],0,0,&cuRNGstate);

	// prepare dynamic array to store cell occupancy and border addresses: replaced with corrected code:
	int glbPos = prepareDynamicShr(vol, mask, devAddr, d_dmnAddr, d_Crtd,  shrCrtd, d_dmnOcc, dmnExt); // here d_Crtd (effectively 0) and d_dmnOcc (all existing inclusions) are exchanged 
	if(threadIdx.x < vol) {
		shrAnglParam[threadIdx.x + 0*vol] = d_phiOrt[glbPos];
		shrAnglParam[threadIdx.x + 1*vol] = d_theOrt[glbPos];
		shrAnglParam[threadIdx.x + 2*vol] = d_phiAvg[glbPos];
		shrAnglParam[threadIdx.x + 3*vol] = d_theAvg[glbPos];
		shrAnglParam[threadIdx.x + 4*vol] = d_phiDev[glbPos];
		shrAnglParam[threadIdx.x + 5*vol] = d_theDev[glbPos];
	}

	if (threadIdx.x < vol) atomicAdd(&selfCNT,shrCrtd[threadIdx.x]);
	__syncthreads();

	// Load CNTs in register file:
	CNT_t probe;
	int cntGlbAddr = -1;		// the address of current cnt in global memory
	short cell	= -1;			// number of current cell
	short pos	= threadIdx.x;	// position of the current thread in cell
	if (threadIdx.x < selfCNT) {
		bool flag = true;
		while (flag) {
			cell++;
			flag = (pos+1 > shrCrtd[cell]);
			if (flag)
				pos -= shrCrtd[cell];
		}
		cntGlbAddr = devAddr[cell+0*vol]+pos;
	}

	for (int i=0;i<9;i++)
		shrArr[threadIdx.x + i*blockDim.x] = (cntGlbAddr>=0?d_result[cntGlbAddr + i*numCNT]:-1);
	__threadfence_block();
	__syncthreads(); 

	// delete randomly:
	if (threadIdx.x < selfCNT)  {
		probe = shr2regCNT(shrArr);
		float probability = check_probability(probe,cell,vol,shrAnglParam);
		float sample = curand_uniform(&cuRNGstate);
		if (probability*thinOut_probFactor < sample) probe = make_emptyCNT(); 
	}
	__syncthreads();

	if (threadIdx.x < selfCNT) reg2shrCNT( probe, shrArr); 
	
	// clean memory before write:
	if (threadIdx.x < selfCNT) 
		for (int i=0;i<9;i++)
			d_result[cntGlbAddr + i*numCNT] = 0;
	__threadfence();
	
	// WRITE RESULTS TO GLOBAL MEMORY:
	writeGlbMem(vol, mask, devAddr, shrAddr, d_Crtd, shrArr, d_result, dmnExt, numCNT);

}
