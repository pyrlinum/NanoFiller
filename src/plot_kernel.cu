//====================================================================================================================
//										 <<< Greyscale plotting functions >>>
//====================================================================================================================
//#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "simPrms.h"
#include "plot_kernel.h"
#include "grscl_TIFFwrite.h"

#define PATCH 32


float color(float n, float nmax) {return (n<nmax?33+n/nmax*222:255);}

__constant__ float resScl;		// pixel patch side 
__constant__ float dmnScl;		// cell grid side
__constant__ float	epsln;
__constant__ int	neiMarg;
__constant__ int	glbPtchSz;
__constant__ int	grdExt[3];
__constant__ float	gaussKern[PATCH];

int	plotSlab(const char*		fname,
			float		Zlvl,
			float		Zhth,
			int			pxlCntXY[2],
			simPrms		*Simul,
			float		def_Tol)
{	
	printf("Ploting TIFF image: ...");
	cudaError cuErr;
	float	*h_pixels = (float*) malloc(pxlCntXY[0]*pxlCntXY[1]*sizeof(float));
	float	*d_pixels;	cudaMalloc(&d_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float));
	for(int i=0;i<pxlCntXY[0]*pxlCntXY[1];i++)	h_pixels[i]=0.0f;
	cudaMemcpy(d_pixels,h_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float),cudaMemcpyHostToDevice);

	float	pxlSize = Simul->physDim[0]/pxlCntXY[0];
	int		dmnCellsInpxlPatch = ((int) ceil((float)Simul->Domains.ext[0]/pxlCntXY[0])+2*Simul->NeiOrder+1)%Simul->Domains.ext[0];
	int		dmnCellsInSlab = ((int) ceil((float)Zhth/Simul->physDim[2]*Simul->Domains.ext[2])+2*Simul->NeiOrder+1)%Simul->Domains.ext[2];

	dmnCellsInpxlPatch = (dmnCellsInpxlPatch>0 ? dmnCellsInpxlPatch : 1);
	dmnCellsInSlab = (dmnCellsInSlab>0 ? dmnCellsInSlab : 1);

	dim3	block;	block.x = PATCH; block.y = PATCH; block.z = 1;
	dim3	grid;	grid.x = pxlCntXY[0]/block.x;
					grid.y = pxlCntXY[1]/block.y;
					grid.z = 1;

	float	h_resScl = block.x*pxlSize;
	float	h_dmnScl = Simul->physDim[0]/Simul->Domains.ext[0];
	cudaMemcpyToSymbol(resScl,&h_resScl,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dmnScl,&h_dmnScl,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(epsln,&def_Tol,sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(neiMarg,&(Simul->NeiOrder),sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(glbPtchSz,&(Simul->Domains.ttlCNT),sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grdExt,Simul->Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);

	// project picture:
	int3		displ = make_int3(1,1,1);
	for(displ.x=0;displ.x<dmnCellsInpxlPatch;displ.x++)
		for(displ.y=0;displ.y<dmnCellsInpxlPatch;displ.y++)
			for(displ.z=0;displ.z<dmnCellsInSlab;displ.z++) 
				cuda_plotXY<<<grid,block>>>(displ,
											Zlvl,
											Zhth,
											Simul->Domains.d_dmnAddr,
											Simul->Domains.d_dmnOcc,
											Simul->d_result,
											d_pixels);
	printf("1: %s \n",cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(h_pixels,d_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float),cudaMemcpyDeviceToHost);
	printf("2: %s \n",cudaGetErrorString(cudaGetLastError()));
	float mxcount =0.0f;
	for(int i=0;i<pxlCntXY[0]*pxlCntXY[1];i++)	if (h_pixels[i]>mxcount) mxcount = h_pixels[i];
	printf("Max Count: %f \n",mxcount);
	for(int i=0;i<pxlCntXY[0]*pxlCntXY[1];i++)	h_pixels[i] = color(h_pixels[i],mxcount);
	cudaMemcpy(d_pixels,h_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float),cudaMemcpyHostToDevice);
	printf("3: %s \n",cudaGetErrorString(cudaGetLastError()));

	//*
	// apply Gaussian filter:
	float	sigma = 0.45f;	// default resolution in microns
	int gaussKnlSize = (int) ceil(3.0f*sigma/pxlSize)+1;
	float *h_gaussKnl =(float*) malloc(gaussKnlSize*sizeof(float));
	float norm = 1.0f;
	h_gaussKnl[0] = 1.0f;
	for(int i=1;i<gaussKnlSize;i++)	{
		h_gaussKnl[i] = expf(-i*i*pxlSize*pxlSize/2.0f/sigma/sigma);
		norm += 2*h_gaussKnl[i]; 
	}
	for(int i=0;i<gaussKnlSize;i++)	{
		h_gaussKnl[i] /= norm; 
	}
	cudaMemcpyToSymbol(gaussKern,h_gaussKnl,gaussKnlSize*sizeof(float),0,cudaMemcpyHostToDevice);
	printf("3.5: %s \n",cudaGetErrorString(cudaGetLastError()));
	int extendedPatchSize = (block.x + 2*(gaussKnlSize-1))*(block.x + 2*(gaussKnlSize-1))*sizeof(float);
	float	*d_tmp;	cudaMalloc(&d_tmp,pxlCntXY[0]*pxlCntXY[1]*sizeof(float));
	cudaMemcpy(d_tmp,h_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float),cudaMemcpyHostToDevice);
	printf("4: %s \n",cudaGetErrorString(cudaGetLastError()));


	gaussBlur<<<grid,block,extendedPatchSize>>>(gaussKnlSize,d_pixels,d_tmp,0);
	printf("4.5: %s \n",cudaGetErrorString(cudaGetLastError()));
	gaussBlur<<<grid,block,extendedPatchSize>>>(gaussKnlSize,d_tmp,d_pixels,1);
	printf("5: %s \n",cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(h_pixels,d_pixels,pxlCntXY[0]*pxlCntXY[1]*sizeof(float),cudaMemcpyDeviceToHost);
	printf("6: %s \n",cudaGetErrorString(cudaGetLastError()));
	for(int i=0;i<pxlCntXY[0]*pxlCntXY[1];i++)	h_pixels[i] = color(h_pixels[i],mxcount);
	//*/
	
	
	cuErr = cudaGetLastError();
	
	if (cuErr == cudaSuccess) {
		char *imgData = (char*) malloc(pxlCntXY[0]*pxlCntXY[1]*sizeof(char));
		for(int i=0;i<pxlCntXY[0]*pxlCntXY[1];i++)	imgData[i] = (char) h_pixels[i];
		free(h_pixels);
		grscl_img_dscr_t img;
						 img.width = pxlCntXY[0];
						 img.length = pxlCntXY[1];
						 img.data = imgData;
		grscl_TIFFwrite(fname, img);
		printf("File %s created! \n",fname);
		free(imgData);
		return 1;
	} else {
		printf(" Plotting failed: ");
		printf( cudaGetErrorString(cuErr) );
		printf("\n");
		return 0;
	}

}


// projector kernel:----------------------------------------------------------------------------------------------------------
__global__ void cuda_plotXY(int3	displ,
							float	Z0,
							float	Heith,
							int		*d_dmnAddr,
							short	*d_dmnOcc,
							float	*d_result,
							float   *imgPxls)
{
	__shared__ int		dmnPatchStrt[3];	// container to keep base point of patch to check for contribution
	__shared__ int		start;				//	starting adress to read
	__shared__ int		CNTload;			//	number of CNTs in current cell to check
	__shared__ float	shrArr[9*BLOCK];	//	
	__shared__ float	pxlPatch[PATCH*PATCH];	//	patch of pixels to check

	if (threadIdx.x == 0) {
		set_dmnPatch(dmnPatchStrt,Z0,Heith,resScl,dmnScl);
		int key = set_keyInPeriodic(dmnPatchStrt,displ,grdExt);
		start = (key!=0?d_dmnAddr[key-1]:0);
		CNTload = d_dmnOcc[key];
	}
	__threadfence_block();
	__syncthreads();

	readImg2Shr(imgPxls,pxlPatch);

	loadCNT2shr(d_result,shrArr,start,CNTload,glbPtchSz);

	for (int i=0;i<CNTload;i++) {
		int tid = threadIdx.x+threadIdx.y*blockDim.x;
		pxlPatch[tid] += impact(i,shrArr,resScl/blockDim.x);
		//pxlPatch[tid] = pxlPatch[tid]||impact(i,shrArr,resScl/blockDim.x);
	}

	__threadfence_block();
	__syncthreads();

	writeShr2Img(imgPxls,pxlPatch);
}
// blurr kernel:------------------------------------------------------------------------------------------------------
__global__ void gaussBlur(	int		gaussKnlSize,
							float	*d_pixelsOld,
							float	*d_pixelsNew,
							char	dir)
{
	extern	__shared__ float pxlPatchExt[];	//	patch of pixels to check
	//read extended sample:
	extendedPatchRead(pxlPatchExt,gaussKnlSize-1, d_pixelsOld);

	int pos = extendedPatchPos(gaussKnlSize-1);
	float sum = pxlPatchExt[pos]*gaussKern[0];
	int strideLn = (dir==0?1:blockDim.x+2*(gaussKnlSize-1));
	for(int i=1;i<gaussKnlSize-1;i++) {
		sum += (pxlPatchExt[pos+i*strideLn]+pxlPatchExt[pos-i*strideLn])*gaussKern[i];
	}
	int glbAddr	=	(threadIdx.x+blockIdx.x*blockDim.x) +
					(threadIdx.y+blockIdx.y*blockDim.y)*gridDim.x*blockDim.x;
	d_pixelsNew[glbAddr] = sum;

}

// Auxilary functions:------------------------------------------------------------------------------------------------
// prepare extended patch:
inline __device__ int extendedPatchPos(int sidebar) {
	return (threadIdx.x+sidebar) + (threadIdx.y+sidebar)*(blockDim.x+2*sidebar);
}
inline __device__ int extendedPatchRead(float *shrPatch,int sidebar,float *glbImg){
	int Xstride = gridDim.x*blockDim.x;
	int Ystride = gridDim.y*blockDim.y;
	int blkOrig	= blockIdx.x*blockDim.x+blockIdx.y*blockDim.y*Xstride;
	int shrPos	= extendedPatchPos(sidebar);
	//read self-patch:
	shrPatch[shrPos] = glbImg[blkOrig+threadIdx.x+threadIdx.y*Xstride];
		//read left bar:
	if (threadIdx.x<sidebar) {
		int xcoord = (threadIdx.x-sidebar	+ blockIdx.x*blockDim.x + Xstride)%(Xstride);
		int ycoord = (threadIdx.y			+ blockIdx.y*blockDim.y );
		shrPatch[shrPos-sidebar] = glbImg[xcoord+ycoord*Xstride];
	}
	//read right bar:
	if ((blockDim.x-1-threadIdx.x)<sidebar) {
		int xcoord = (threadIdx.x+sidebar	+ blockIdx.x*blockDim.x)%(Xstride);
		int ycoord = (threadIdx.y			+ blockIdx.y*blockDim.y);
		shrPatch[shrPos+sidebar] = glbImg[xcoord+ycoord*Xstride];
	}
	//read upper bar:
	if (threadIdx.y<sidebar) {
		int xcoord = (threadIdx.x			+ blockIdx.x*blockDim.x);
		int ycoord = (threadIdx.y-sidebar	+ blockIdx.y*blockDim.y+Ystride)%(Ystride);
		shrPatch[shrPos-sidebar*(blockDim.x+2*sidebar)] = glbImg[xcoord+ycoord*Xstride];
	}
	//read lower bar:
	if ((blockDim.y-1-threadIdx.y)<sidebar) {
		int xcoord = (threadIdx.x			+ blockIdx.x*blockDim.x);
		int ycoord = (threadIdx.y+sidebar	+ blockIdx.y*blockDim.y)%(Ystride);
		shrPatch[shrPos+sidebar*(blockDim.x+2*sidebar)] = glbImg[xcoord+ycoord*Xstride];
	}
	__threadfence_block();
	__syncthreads();
	return 0;
}
// load cnt from current cell into shared memory
inline __device__ int loadCNT2shr(float *d_result, float *shr_arr, int st_addr, int load, int glb_patch) {
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int shr_patch = blockDim.x*blockDim.y;
	if (tid<load) {
		#pragma unrol
		for(int i=0;i<9;i++)
			shr_arr[tid+i*shr_patch] = d_result[st_addr+i*glb_patch];
	}
	__threadfence_block();
	__syncthreads();

	return 0;
}
// read pixel patch from device memory:
inline __device__ int readImg2Shr(float *d_imgArr,float *shr_imgArr){
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int glbAddr = threadIdx.x + blockIdx.x*blockDim.x+
				 (threadIdx.y + blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
	shr_imgArr[tid] = d_imgArr[glbAddr];
	__threadfence_block();
	__syncthreads();

	return 0;
}
// write pixel patch to device memory:
inline __device__ int writeShr2Img(float *d_imgArr,float *shr_imgArr){
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int glbAddr = threadIdx.x + blockIdx.x*blockDim.x+
				 (threadIdx.y + blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
	d_imgArr[glbAddr] = shr_imgArr[tid];
	__threadfence();
	__syncthreads();

	return 0;
}
// compute domain grid patch basepoint:
inline __device__ int set_dmnPatch(int dmnPtch[3],float z0, float h, float resScl, float dmnScl) {
		dmnPtch[0] =(int) floor(blockIdx.x*resScl/dmnScl)	- neiMarg;	// xlo
		dmnPtch[1] =(int) floor(blockIdx.y*resScl/dmnScl)	- neiMarg;	// ylo
		dmnPtch[2] =(int) floor(z0/dmnScl)					- neiMarg; // zlo

	return 0;
}
// compute current grid cell key:
inline __device__ int set_keyInPeriodic(int basePnt[3], int3 displ, int dmnGrdExt[3]) {

	int x[3];
	x[0] = basePnt[0] + displ.x;
	x[1] = basePnt[1] + displ.y;
	x[2] = basePnt[2] + displ.z;
#pragma unroll
	for(int i=0;i<3;i++){
		x[i] = (x[i]<0?x[i]+dmnGrdExt[i]:x[i]);
		x[i] = (x[i]>=dmnGrdExt[i]?x[i]%dmnGrdExt[i]:x[i]);
	}
	return x[0] + x[1]*dmnGrdExt[0] + x[2]*dmnGrdExt[0]*dmnGrdExt[1];
}
// MAIN AUXILARY: cnt impact in current cell color:
// the simpliest impact function - 1/0
inline __device__ float impact(int i,float *shr_arr, float scale){
	int shr_patch = blockDim.x*blockDim.y;
	float	xlo = scale*(threadIdx.x   + blockIdx.x*blockDim.x);
	float	xhi = scale*(threadIdx.x+1 + blockIdx.x*blockDim.x);
	float	ylo = scale*(threadIdx.y   + blockIdx.y*blockDim.y);
	float	yhi = scale*(threadIdx.y+1 + blockIdx.y*blockDim.y);

	float	limD = shr_arr[i+6*shr_patch]/2+shr_arr[i+7*shr_patch];

	// check for line intersection:
	float2	Quad[2],A0,CX;
			Quad[0] = make_float2(xlo,ylo);
			Quad[1] = make_float2(xhi,yhi);
			A0		= make_float2(shr_arr[i+0*shr_patch],shr_arr[i+1*shr_patch]);
			CX		= make_float2(shr_arr[i+3*shr_patch],shr_arr[i+4*shr_patch]);
			
			if ( (lineQuadroIsec2D(A0,CX,Quad)) && (PointQuadDist(A0,CX,Quad) < limD) )
				return 1.0f;
			else	return 0.0f;
}
// returns 1 if line and quadro intersects:
inline __device__ bool lineQuadroIsec2D(float2 a0, float2 cos, float2 quad[2]){
	bool flag = 0;
	if		(cos.x==0.0f) flag = (a0.x>=quad[0].x)&&(a0.x<=quad[1].x);
	else if	(cos.y==0.0f) flag = (a0.y>=quad[0].y)&&(a0.y<=quad[1].y);
	else {
		char vflag[4];
		vflag[0] = ( ((quad[0].x-a0.x)/cos.x-(quad[0].y-a0.y)/cos.y)>=0-epsln ? 1 : -1 );
		vflag[1] = ( ((quad[0].x-a0.x)/cos.x-(quad[1].y-a0.y)/cos.y)>=0-epsln ? 1 : -1 );
		vflag[2] = ( ((quad[1].x-a0.x)/cos.x-(quad[0].y-a0.y)/cos.y)>=0-epsln ? 1 : -1 );
		vflag[3] = ( ((quad[1].x-a0.x)/cos.x-(quad[1].y-a0.y)/cos.y)>=0-epsln ? 1 : -1 );

		flag = (vflag[0]*vflag[1]<0)||(vflag[2]*vflag[3]<0)||(vflag[0]*vflag[2]<0)||(vflag[1]*vflag[3]<0);
	}
	return flag;
}
inline __device__ float PointQuadDist(float2 a0,float2 cx, float2 quad[2]){
	float d1 = (abs(cx.x)>epsln?abs((a0.x - quad[0].x)/cx.x):999);
	float d2 = (abs(cx.x)>epsln?abs((a0.x - quad[1].x)/cx.x):999);
			d2 = (d1<d2?d1:d2);
	float d3 = (abs(cx.y)>epsln?abs((a0.y - quad[0].y)/cx.y):999);
	float d4 = (abs(cx.y)>epsln?abs((a0.y - quad[1].y)/cx.y):999);
			d4 = (d3<d4?d3:d4);
	return (d2<d4?d2:d4);	
}
