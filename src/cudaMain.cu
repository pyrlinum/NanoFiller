#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include "simPrms.h"
#include "IO.h"

//#define _DEBUG

// CUDA declarations:
//__constant__ float	Pi;									// on-device representation of PI
__constant__ float	DEF_L;								// bale CNT model length
__constant__ float	DEF_A;								// base CNT model diameter
__constant__ int	dmnExt[3];							// Domain grid extents
__constant__ float	phsScl[3];							// Scale factors of the model (rato of physDim/dmnExt)
__constant__ int	neiOrder;							// Number of neighbouring domains to check intersection with
__constant__ int	numCNT;								// Maximum number of CNT to generate - used for device memory allignment
__constant__ int	numIntPoints;						// Number of points per unit length for probability density integration 
__constant__ float	epsilon;							// precision for math operations while checking for intersection
__constant__ float	separation;							// minimum separation between CNT models
__constant__ float	sc_frac;							// inclusion soft core fraction of diameter
__constant__ int3	texDim;								// texture extents

__constant__	float Phi_avg;
__constant__	float Phi_dev;
__constant__	float Theta_avg;
__constant__	float Theta_dev;
__constant__	float prefDir[2];
__constant__	int		incOrt_threshold;
__constant__	float	incOrt_minNorm;
__constant__	float	thinOut_threshold;
__constant__	float	thinOut_probFactor;

texture<float,3,cudaReadModeElementType>	denTex;		// to handle probability density mesh texture
cudaArray *den_arr = 0;									// to store probability density mesh texture


//const int	KER_REG = 23;				// registers per kernel
#include "set_Params.h"
#include "den2tex.h"			// write porbability density to texture
#include "cudaDmnIntegrate.h"	// CUDA integration kernel
#include "cuda_CNTgenerate.h"	// CUDA CNT-generation kernel
#include "cuda_Clusterize.h"	// CUDA CNT-clusterization kernel
#include "intersection_kern.h"	// CUDA CNT-intersection check
#include "auxilary_kernels.h"	// Auxiliary functions (renewOcc)
#include "plot_kernel.h"		// Projection droiwng kernel

#include "allignment_fit.h"

int	cudaGenStep(int iter, int maxGridSize, simPrms *simRef,  int OldCrtdSum );					// defined below
int	cudaThinOUT(int iter, int maxGridSize, simPrms *simRef );												// defined below
float 	*crtdDistrCheck( int maxGridSize, simPrms *simRef);										// defined below
void save_state_BIN(const char* fname, simPrms Simul);

int	cudaCheck_Allignment(int iter, int maxGridSize, simPrms *simRef );
int	cudaCheck_corMtx(int iter, int total, int maxGridSize, simPrms *simRef );

void save_state_BIN(const char* fname, simPrms Simul);	// saving state

// Entry point:
//CNT_t *cudaMain(int *cntNum, float procent, float	d_L, float	d_R, float SEP, char *fname ) {
int	cudaMain(simPrms *Simul ) {

	int flag = 1;
	
	if (Simul->crdDsrtFile=="NONE") {
		printf("Using default grid \n");
			flag *= Simul->make_default_den();
			//write_Meshvtk("defmesh.vtk",Simul->ext,Simul->physDim, Simul->mesh);
	}else	flag *= Simul->make_den(Simul->crdDsrtFile.c_str());
	//write_Meshvtk("defmesh.vtk",Simul->ext,Simul->physDim, Simul->mesh);

	flag *= Simul->set_numCNT(Simul->l,Simul->a,Simul->vf);
	density2Tex(Simul->mesh, Simul->ext);	

	flag *= Simul->make_dmnGrid(Simul->Block.x,Simul->MARGE);
	Simul->Domains.segNum = Simul->distributeDmn(Simul->Block.x);
	flag *= setDevConst(Simul);

	Simul->maxVol = Simul->maxLeaf();
	int maxGridSize = Simul->device_config();
	//int maxGridSize = Simul->device_config(KER_REG);

	int Mem_TTL = Simul->allocateDevArr();
#ifdef _DEBUG
	printf("_DEBUG: Max Vol = %i \n",Simul->maxVol);
	printf("_DEBUG: Total memory allocated on device: %5.3f Mb \n",((float)Mem_TTL)/MegaByte);
#endif

	
	if (!((flag>0)&&(cudaGetLastError()==cudaSuccess))) {
		printf("_DEBUG: Device initialization failed! %s \n",cudaGetErrorString(cudaGetLastError()));
		//getchar();
	} else {
        
            printf("Device setup finished. Starting generation cycle: \n");
        
            cudaError_t cuErr = cudaSuccess;
            cudaEvent_t start,stop,iStart,iStop;
            float time,iTime;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);

            int total = 0, total0 = 0;
            int counter = 0;
            int vfpos = 0;

            // to check intersection count:
            //short *d_watchList; cuErr = cudaMalloc(&d_watchList,dmnGridSize*sizeof(short));
            
            printf("================================================================================\n");
            while ((cuErr == cudaSuccess)&&(total < Simul->numCNT*(1-Simul->PRECISION))&&
                    ((counter==0)||(total-total0>0))&&(counter<1000)) {
            //while (counter < 21)	{
                    cudaEventCreate(&iStart);
                    cudaEventCreate(&iStop);
                    cudaEventRecord(iStart,0);
                total0 = total;
                total = cudaGenStep(counter,maxGridSize,Simul,total);
                    


                if ( 	 (counter>0)	&&
                	  (  (Simul->self_alling_step>0) &&
                		( ((counter%Simul->self_alling_step)==0) || (total-total0<(Simul->numCNT-total)/100.0f) ) ) ) {
                        cudaCheck_corMtx(counter, total, maxGridSize, Simul );
                        if (Simul->reduce_flag) {
                            int remain = cudaThinOUT(counter,maxGridSize, Simul);
                            printf( "ThinOUT: %i inclusions remained \n",remain);
                            //Simul->set_dmnOrt();
                        }
                    }

                // check concentration list:
                if (Simul->vfcount >0)
                    if  (Simul->get_VolFrac(total) >= Simul->vfs[vfpos]) {
                        char buffer[255];
                        sprintf(buffer,"state_%4.2f.bin",Simul->get_VolFrac(total));
                        save_state_BIN(buffer, *Simul);
                        vfpos++;
                    }

                    cudaEventRecord(iStop,0);
                    cudaEventSynchronize(iStop);
                    cudaEventElapsedTime(&iTime,iStart,iStop);
                    printf( "%i iter. took %f ms %i inclusions Created. Total: %i \n",counter,iTime,total-total0,total );
                    counter++;
                    }
            printf("================================================================================\n");
		
            Simul->numCNT = total;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		printf( "Total Time: %f ms \n",time );
	}

	return Simul->numCNT;
}
//------------------------------------------------------------------------------------------------------------------------
// CUDA domain integration wrapper
//------------------------------------------------------------------------------------------------------------------------
double integrateDmnGrid(int ext[3],float *h_dmnInt) {
	
	int size = ext[0]*ext[1]*ext[2];
	float	*d_dmnInt;
#ifdef _DEBUG
	cudaError_t	cuErr;
#endif
	
	cudaMalloc((void **) &d_dmnInt,size*sizeof(float));

	dim3 Grid;
	dim3 Block;

	Block.x = 1024;
	Grid.x = (int) ceil(((float)size)/Block.x);

	int	numP = 10;
	// CUDA Integration:
	cudaMemcpyToSymbol(numIntPoints,(void *) &numP,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dmnExt,(void *) ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaDmnIntegrate<<<Grid,Block>>>(d_dmnInt);
	cudaMemcpy((void *) h_dmnInt,(void *) d_dmnInt,size*sizeof(float),cudaMemcpyDeviceToHost);
#ifdef _DEBUG
	printf( "_DEBUG: CUDA domain integration wrapper ");
	cuErr = cudaGetLastError();
	printf( cudaGetErrorString(cuErr) );
	printf("\n");
#endif

	double sum = std::accumulate(h_dmnInt,h_dmnInt+size,0.0);
	return sum;
}
//------------------------------------------------------------------------------------------------------------------------
// generation cycle iteration:
//------------------------------------------------------------------------------------------------------------------------
// auxilary functions:
int	checkArray(int size, short* h_Crtd, short* d_Crtd) {
	cudaMemcpy((void*) h_Crtd,(void*) d_Crtd,size*sizeof(short),cudaMemcpyDeviceToHost);
#ifdef _DEBUG
	cudaError_t	cuErr;
	cuErr = cudaGetLastError();
	printf(" %i short values read!\n",size);
	printf( cudaGetErrorString(cuErr) );
	printf("\n");
#endif
	int CrtdSum = 0;
	for(int i=0;i<size;i++) {
		if (h_Crtd[i]!=0)
		CrtdSum+=h_Crtd[i];
	}
	return CrtdSum;
}

// main entry
int	cudaGenStep(int iter, int maxGridSize, simPrms *simRef, int OldCrtdSum ) {

#ifdef _DEBUG
    int GenSum,RemSelf,RemNei,RemSum;
#endif
	int dmnGridSize = simRef->Domains.ext[0]*simRef->Domains.ext[1]*simRef->Domains.ext[2];
	short *h_Crtd = (short*) malloc(dmnGridSize*sizeof(short));
	int shrdMem = (size_t) simRef->maxVol*10*sizeof(int);
	// GENERATION:	

	printf("Generation ...");
	cuda_CNTgenerate_orient<<<simRef->Grid,simRef->Block,shrdMem>>>(	simRef->Domains.d_dmnAddr,
																		simRef->Domains.d_dmnOcc,
																		simRef->Domains.d_dmnCrt,
																		simRef->Domains.d_masks,
																		simRef->d_RNGseed,
																		simRef->d_thetaMed,
																		simRef->d_thetaDev,
																		simRef->d_phiMed,
																		simRef->d_phiDev,
																		&simRef->d_prefOrt[0*dmnGridSize],
																		&simRef->d_prefOrt[1*dmnGridSize],
																		simRef->d_result);
	//*/
	printf("Finished!\n");
#ifdef _DEBUG
    GenSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt); 	
	int seed = 1234;
#else
	int seed = clock();
#endif
	simRef->seed_cuRND(seed);
	
	// Clusterize if needed: - Source of high freq resonance
	//*
	if (simRef->clusterize_flag) {
		printf("Clusterize \n");
		for(int k = -simRef->NeiOrder; k <= simRef->NeiOrder; k++ )	
			for(int j = -simRef->NeiOrder; j <= simRef->NeiOrder; j++ )
				for(int i = -simRef->NeiOrder; i <= simRef->NeiOrder; i++ )
					if ( simRef->NeiOrder+1 > sqrtf(i*i+j*j+k*k) ) {
						char3	stride = make_char3(i,j,k);		
						for (int count = 0; count < simRef->kernelSplit; count++)	{
							int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
							cuda_Clusterize2Old<<<lnchGSize,simRef->Block.x>>>(	count,
																				stride,
																				simRef->clusterize_angl,
																				simRef->clusterize_dist,
																				simRef->Domains.d_dmnAddr,
																				simRef->Domains.d_dmnOcc,
																				simRef->d_result,
																				simRef->Domains.d_dmnCrt);
						}
					}
	}
	//*/
	//*
	// INTRENAL INTERSECTIONS: 
	shrdMem = (size_t) simRef->maxVol*5*sizeof(int);
	cuda_InternalIsect_noLck<<<simRef->Grid,simRef->Block,shrdMem>>>(	simRef->Domains.d_dmnAddr,
																		simRef->Domains.d_dmnOcc,
																		simRef->Domains.d_masks,
																		simRef->d_result,
																		simRef->Domains.d_dmnCrt);
#ifdef _DEBUG
	RemSelf = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
#endif
//*
	// BORDER INTERSECTIONs:
	char3 stride;
	int xl = 1; int xh = simRef->NeiOrder;
	int yl = 0; int yh = simRef->NeiOrder;
	int zl = 0; int zh = simRef->NeiOrder;
	
		for(int k = zl; k <= zh; k++ )	{
			for(int j = yl; j <= yh; j++ )	{
				for(int i = xl; i <= xh; i++ )
					if ( simRef->NeiOrder+1 > sqrtf(i*i+j*j+k*k) ) {
						stride = make_char3(i,j,k);		
						for (char part = 0; part < 3 ; part++)	{
							for (int count = 0; count < simRef->kernelSplit; count++)	{
								int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
								cuda_NewExtIsectLB_parted<<<lnchGSize,simRef->Block.x>>>(	part,
																							count,
																							stride,
																							simRef->Domains.d_dmnAddr,
																							simRef->Domains.d_dmnOcc,
																							simRef->d_result,
																							simRef->Domains.d_dmnCrt,
																							simRef->Domains.d_dmnLck);
							}
						}
					}
				xl = -xh;
			}
			yl = -yh;
		}
#ifdef _DEBUG
	RemNei = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
#endif
//*		
	// OLD INTERSECTIONS:
		for(int k = -simRef->NeiOrder; k <= simRef->NeiOrder; k++ )	
			for(int j = -simRef->NeiOrder; j <= simRef->NeiOrder; j++ )
				for(int i = -simRef->NeiOrder; i <= simRef->NeiOrder; i++ )
					if ( simRef->NeiOrder+1 > sqrtf(i*i+j*j+k*k) ) {
						char3	stride = make_char3(i,j,k);		
						for (int count = 0; count < simRef->kernelSplit; count++)	{
	//{{int count = 0; char3	stride = make_char3(0,0,0);	
							int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
							cuda_OldExtIsectLB<<<lnchGSize,simRef->Block.x>>>(	count,
							//cuda_OldExtIsectLB_wNumIsecStore<<<lnchGSize,simRef->Block.x>>>(	count,
																				stride,
																				simRef->Domains.d_dmnAddr,
																				simRef->Domains.d_dmnOcc,
																				simRef->d_result,
																				simRef->Domains.d_dmnCrt);
						}
					}
	
				//*/

#ifdef _DEBUG
	RemSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
	printf("%i new: %i self: %6.2f nei: %6.2f old: %6.2f kept: %i - %6.2f \n",iter,GenSum,(float)(GenSum-RemSelf)/GenSum*100,(float)(RemSelf-RemNei)/GenSum*100,(float)(RemNei-RemSum)/GenSum*100,RemSum,(float)RemSum/GenSum*100);
#endif
	cuda_renewOcc<<<ceil(dmnGridSize/512.0f),512>>>(RENEW_ADD,simRef->Domains.d_dmnOcc,simRef->Domains.d_dmnCrt);
	int CrtdSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnOcc);
	free((void*) h_Crtd);
	
	return CrtdSum;

}
//------------------------------------------------------------------------------------------------------------------------
// Delete not probable:
int	cudaThinOUT(int iter, int maxGridSize, simPrms *simRef ) {

	int dmnGridSize = simRef->Domains.ext[0]*simRef->Domains.ext[1]*simRef->Domains.ext[2];
	int CrtdSum,OccSum;
	short *h_Crtd = (short*) malloc(dmnGridSize*sizeof(short));
	short *h_Occ = (short*) malloc(dmnGridSize*sizeof(short));
	int shrdMem = (size_t) simRef->maxVol*11*sizeof(int);	


	CrtdSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
	OccSum = checkArray(dmnGridSize,h_Occ,simRef->Domains.d_dmnOcc);
	printf("Before reduction: %i created, %i occupied\n",CrtdSum,OccSum);

	float factor = 2;
	cudaMemcpyToSymbol(thinOut_probFactor,&factor,sizeof(float),0,cudaMemcpyHostToDevice);

			randomThinOut<<<simRef->Grid,simRef->Block,shrdMem>>>(	simRef->Domains.d_dmnAddr,
																	simRef->Domains.d_dmnOcc,
																	simRef->Domains.d_masks,
																	simRef->d_result,
																	simRef->Domains.d_dmnCrt,
																	&simRef->d_prefOrt[0*dmnGridSize],
																	&simRef->d_prefOrt[1*dmnGridSize],
																	simRef->d_phiMed,
																	simRef->d_thetaMed,
																	simRef->d_phiDev,
																	simRef->d_thetaDev,
																	simRef->d_RNGseed);

	//*/
	CrtdSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
	OccSum = checkArray(dmnGridSize,h_Occ,simRef->Domains.d_dmnOcc);
	printf("After reduction: %i created, %i occupied\n",CrtdSum,OccSum);
	cuda_renewOcc<<<ceil(dmnGridSize/512.0f),512>>>(RENEW_REPLACE,simRef->Domains.d_dmnOcc,simRef->Domains.d_dmnCrt);
	CrtdSum = checkArray(dmnGridSize,h_Crtd,simRef->Domains.d_dmnCrt);
	OccSum = checkArray(dmnGridSize,h_Occ,simRef->Domains.d_dmnOcc);
	printf("After renewal: %i created, %i occupied\n",CrtdSum,OccSum);

	int seed = clock();//1234*iter+4321;
	simRef->seed_cuRND(seed);
	return CrtdSum;

}

//------------------------------------------------------------------------------------------------------------------------
// Check mesh of created models:
//------------------------------------------------------------------------------------------------------------------------
float *crtdDistrCheck( int maxGridSize, simPrms *simRef) {

#ifdef _DEBUG
	printf( "_DEBUG: Calculating obtained distribution: \n");
	cudaError_t  cuErr;
	cudaEvent_t	 mStart,mStop;
	float		 mTime;	
#endif
	
	int		texSize = simRef->ext[0]*simRef->ext[1]*simRef->ext[2];
	float	*h_curMesh = (float*) malloc(texSize*sizeof(float));
	float	*d_curMesh;	cudaMalloc((void**) &d_curMesh,texSize*sizeof(float));
	int		dmnGridSize = simRef->Domains.ext[0]*simRef->Domains.ext[1]*simRef->Domains.ext[2];
	int		patchMem =   	((int) ceil( ((float) simRef->ext[0])/simRef->Domains.ext[0])+2)*
					((int) ceil( ((float) simRef->ext[1])/simRef->Domains.ext[1])+2)*
					((int) ceil( ((float) simRef->ext[2])/simRef->Domains.ext[2])+2)*sizeof(float);

		
		for (int i=0;i<texSize;i++) h_curMesh[i]=0.0f;
		cudaMemcpy((void*) d_curMesh, (void*) h_curMesh, texSize*sizeof(float), cudaMemcpyHostToDevice);
#ifdef _DEBUG
		printf( "_DEBUG: device memory initialisation: ");
		cuErr = cudaGetLastError();
		printf( cudaGetErrorString(cuErr) );
		printf("\n");

		
		printf( "_DEBUG: Allocating %i bytes of memory for shared patches \n",patchMem);
		cudaEventCreate(&mStart);
		cudaEventCreate(&mStop);
		cudaEventRecord(mStart,0);
		printf( "_DEBUG: Running kernell: ");
#endif
		for (int count = 0; count < simRef->kernelSplit; count++)	{
			int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
			cuda_plotMesh<<<lnchGSize,simRef->Block.x,patchMem>>>(	count,
																	simRef->Domains.d_dmnAddr,
																	simRef->Domains.d_dmnOcc,
																	simRef->Domains.d_dmnCrt,
																	simRef->d_result,
																	d_curMesh);
		}
#ifdef _DEBUG
		printf( "_DEBUG: ");
		cuErr = cudaGetLastError();
		printf( cudaGetErrorString(cuErr) );
		cudaEventRecord(mStop,0);
		cudaEventSynchronize(mStop);
		cudaEventElapsedTime(&mTime,mStart,mStop);
		printf( " Reading probability mesh took: %f ms \n",mTime );
#endif

		cudaMemcpy((void*) h_curMesh, (void*) d_curMesh, texSize*sizeof(float), cudaMemcpyDeviceToHost);
#ifdef _DEBUG
		printf( "_DEBUG: Copying memory from device: ");
		cuErr = cudaGetLastError();
		printf( cudaGetErrorString(cuErr) );
		printf("\n");
		double texSUM = 0.0;
		for(int i=0;i<texSize;i++) texSUM += h_curMesh[i];
		printf( "_DEBUG: Sum of %i found: %f \n",texSize,texSUM);
#endif		
		return h_curMesh;
}

//------------------------------------------------------------------------------------------------------------------------

int	cudaCheck_Allignment(int iter, int maxGridSize, simPrms *simRef ) {

	int dmnGridSize = simRef->Domains.ext[0]*simRef->Domains.ext[1]*simRef->Domains.ext[2];

	float	*h_prefOrt	 =	(float *) malloc(2*dmnGridSize*sizeof(float));
	float	*h_theta_avg =	(float *) malloc(dmnGridSize*sizeof(float));					// avarage theta angle 
	float	*h_theta_msd =	(float *) malloc(dmnGridSize*sizeof(float));					// mean squared deviation of theta
	float	*h_phi_avg	=	(float *) malloc(dmnGridSize*sizeof(float));					// avarage phi angle 
	float	*h_phi_msd	=	(float *) malloc(dmnGridSize*sizeof(float));					// mean squared deviation of phi
/*
	for (int count = 0; count < simRef->kernelSplit; count++)	{
		int lnchGSize = ( (dmnGridSize - count*this->kernelSize) < this->kernelSize ? (dmnGridSize - count*this->kernelSize) : this->kernelSize );
		fit_Gauss1D<<<lnchGSize,simRef->Block.x,10*32*sizeof(float)>>>(	count,
													simRef->Domains.d_dmnAddr,
													simRef->Domains.d_dmnOcc,
													simRef->d_result,
													simRef->d_thetaMed,
													simRef->d_thetaDev,
													simRef->d_phiMed,
													simRef->d_phiDev,
													&simRef->d_prefOrt[0*dmnGridSize],
													&simRef->d_prefOrt[1*dmnGridSize]);
													
	}
*/
	simRef->def_prefOrt[0] = 0;
	simRef->def_prefOrt[1] = 0;
	simRef->h_prefOrt = (float*) malloc(6*dmnGridSize*sizeof(float));
	cudaMemcpy(simRef->h_prefOrt,simRef->d_prefOrt,2*dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_prefOrt+2*dmnGridSize,simRef->d_thetaMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_prefOrt+3*dmnGridSize,simRef->d_thetaDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_prefOrt+4*dmnGridSize,simRef->d_phiMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_prefOrt+5*dmnGridSize,simRef->d_phiDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);

	write_NInput_dat("lambda", 6, dmnGridSize, simRef->h_prefOrt);

	for(int i=0;i<dmnGridSize;i++) {
		simRef->def_prefOrt[0] += simRef->h_prefOrt[i+0*dmnGridSize];
		simRef->def_prefOrt[1] += simRef->h_prefOrt[i+1*dmnGridSize];
	}
	free(simRef->h_prefOrt);
	printf("Prefered orientation angles: %f - %f \n",simRef->def_prefOrt[0]/dmnGridSize,simRef->def_prefOrt[1]/dmnGridSize);
	simRef->def_thetaMed = 0;
	simRef->def_thetaDev = 0;
	simRef->h_thetaMed = (float*) malloc(dmnGridSize*sizeof(float));
	simRef->h_thetaDev = (float*) malloc(dmnGridSize*sizeof(float));
	cudaMemcpy(simRef->h_thetaMed,simRef->d_thetaMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_thetaDev,simRef->d_thetaDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	for(int i=0;i<dmnGridSize;i++) {
		simRef->def_thetaMed += simRef->h_thetaMed[i];
		simRef->def_thetaDev += simRef->h_thetaDev[i];
	}
	free(simRef->h_thetaDev);
	free(simRef->h_thetaMed);
	printf("Theta: %f - %f \n",simRef->def_thetaMed/dmnGridSize,simRef->def_thetaDev/dmnGridSize);

	simRef->def_phiMed = 0;
	simRef->def_phiDev = 0;
	simRef->h_phiMed = (float*) malloc(dmnGridSize*sizeof(float));
	simRef->h_phiDev = (float*) malloc(dmnGridSize*sizeof(float));
	cudaMemcpy(simRef->h_phiMed,simRef->d_phiMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(simRef->h_phiDev,simRef->d_phiDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	for(int i=0;i<dmnGridSize;i++) {
		simRef->def_phiMed += simRef->h_phiMed[i];
		simRef->def_phiDev += simRef->h_phiDev[i];
	}
	free(simRef->h_phiDev);
	free(simRef->h_phiMed);
	printf("Phi: %f - %f \n",simRef->def_phiMed/dmnGridSize,simRef->def_phiDev/dmnGridSize);

	return 0;

}
//-------------------------------------------------
// Correlation matrix collection:
int	cudaCheck_corMtx(int iter, int total, int maxGridSize, simPrms *simRef ) {

	int dmnGridSize = simRef->Domains.ext[0]*simRef->Domains.ext[1]*simRef->Domains.ext[2];

	float	*d_crossMtx;	 cudaMalloc(&d_crossMtx,7*dmnGridSize*sizeof(float));

	for (int count = 0; count < simRef->kernelSplit; count++)	{
		int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
		crossMtxBuild<<<lnchGSize,simRef->Block.x,6*simRef->Block.x*sizeof(float)>>>(	count,
																						simRef->Domains.d_dmnAddr,
																						simRef->Domains.d_dmnOcc,
																						simRef->d_result,
																						d_crossMtx);
													
	}
	
	int addGrd = (int)ceil((float)dmnGridSize/simRef->Block.x);
	resetOrt<<<addGrd,simRef->Block.x,6*simRef->Block.x*sizeof(float)>>>(	d_crossMtx,	&simRef->d_prefOrt[0],&simRef->d_prefOrt[1]);

	float phi;
	float theta;

	/*
	float	*h_Ort =	(float *) malloc(3*dmnGridSize*sizeof(float));
	cudaMemcpy(h_Ort,simRef->d_prefOrt,2*dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0;i<dmnGridSize;i++){
		phi = h_Ort[i+0*dmnGridSize];
		theta = h_Ort[i+1*dmnGridSize];

		h_Ort[i+0*dmnGridSize] = cos(phi)*sin(theta);
		h_Ort[i+1*dmnGridSize] = sin(phi)*sin(theta);
		h_Ort[i+2*dmnGridSize] = cos(theta);
	}
	
	char fname0[255];
	sprintf(fname0,"ort_%i.vtk",iter);
	write_vtkVectorField(fname0,simRef->Domains.ext,h_Ort); //*/
	//*
	for (int count = 0; count < simRef->kernelSplit; count++)	{
		int lnchGSize = ( (dmnGridSize - count*simRef->kernelSize) < simRef->kernelSize ? (dmnGridSize - count*simRef->kernelSize) : simRef->kernelSize );
		resetAvgVec<<<lnchGSize,simRef->Block.x,4*simRef->Block.x*sizeof(float)>>>(	count,
																						simRef->Domains.d_dmnAddr,
																						simRef->Domains.d_dmnOcc,
																						simRef->d_result,
																						&simRef->d_prefOrt[0],
																						&simRef->d_prefOrt[1],
																						simRef->d_phiMed,
																						simRef->d_thetaMed,
																						simRef->d_phiDev,
																						simRef->d_thetaDev);
	}

	/*
	float	*h_PhiThe =	(float *) malloc(3*dmnGridSize*sizeof(float));
	cudaMemcpy(&h_PhiThe[0*dmnGridSize],simRef->d_phiMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_PhiThe[1*dmnGridSize],simRef->d_thetaMed,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0;i<dmnGridSize;i++){
		phi = h_PhiThe[i+0*dmnGridSize];
		theta = h_PhiThe[i+1*dmnGridSize];

		h_PhiThe[i+0*dmnGridSize] = cos(phi)*sin(theta);
		h_PhiThe[i+1*dmnGridSize] = sin(phi)*sin(theta);
		h_PhiThe[i+2*dmnGridSize] = cos(theta);
	}
	
	char fname1[255];
	sprintf(fname1,"phi-theta_%i.vtk",iter);
	write_vtkVectorField(fname1,simRef->Domains.ext,h_PhiThe); //*/

	float	*h_Dev =	(float *) malloc(2*dmnGridSize*sizeof(float));
	cudaMemcpy(&h_Dev[0*dmnGridSize],simRef->d_phiDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_Dev[1*dmnGridSize],simRef->d_thetaDev,dmnGridSize*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0;i<dmnGridSize;i++){
		phi += h_Dev[i+0*dmnGridSize];
		theta += h_Dev[i+1*dmnGridSize];
	}
	printf("Theta dev: %f \n",theta/dmnGridSize);
	printf(" Phi  dev: %f \n",phi/dmnGridSize);
	//*/

	return 0;

}
