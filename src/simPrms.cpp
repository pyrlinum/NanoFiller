#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include "CNT.h"
#include "IO.h"
#include "kdTree.h"
#include "simPrms.h"
#include "parameter_input.h"
//#include "core_cell_interaction_kernel.h"

//#define _DEBUG
//#define VERBOUS

double	integrateDmnGrid(int ext[3],float *h_dmnInt);
int	dmnIdx_old2new(int idx, int oldGridExt[3],int newGridExt[3]);
int	dmnIdx_new2old(int idx, int oldGridExt[3],int newGridExt[3]);
vector<float>	offset_new2old(int idx, int oldGridExt[3],int newGridExt[3], float physDim[3]);

simPrms::simPrms() {
	// set default parameters:
	this->vf = 0.05f/100.0f;
	this->vfcount = -1;
	this->vfs = 0;
	this->density = 4.579f;	// gramm/mole/A^2

	this->crdDsrtFile = "NONE";
	fill(this->physDim,this->physDim+3,0.);
	fill(this->ext,this->ext+3, 0);
	this->maxVol = 0;


	this->ortDsrtFile = "";
	this->def_prefOrt[0] = 0;
	this->def_prefOrt[1] = 0;
	this->def_thetaMed = 0;
	this->def_thetaDev = -1;
	this->def_phiMed = 0;
	this->def_phiDev = -1;
	this->def_minOrtNorm = 0.5;
	this->def_OrtThreshold = 0;

	this->dimDstrFile = "";
	//this->l = 50;		// CNF 
	this->l = 4;		// CNT
	this->dev_l = -1;
	//this->a = 0.075;	//CNF
	this->a = 0.010f;	//CNT
	this->dev_a = -1;

	this->TOL = 1.e-6f;
	this->PRECISION = 0.05f;
	this->MARGE = 0.00f;
	this->sep = 0;
	this->sc_frac = 1.0;

	this->stat_dist_lim = 0.015f;

	this->clusterize_flag = false;
	this->clusterize_dist = 0.015f;
	this->clusterize_angl = Pi/12;
	

	this->wrt_bin_flag = false;
	this->reduce_flag = false;
	this->self_alling_step = 50;

	// internal variables:
	this->numCNT = 0;		

	this->cuRNG_Num = 0;	

	fill(this->Domains.edge,this->Domains.edge+3,0.0f);
	fill(this->Domains.ext,this->Domains.ext+3, 0);
	this->Domains.grdSize = accumulate(this->Domains.ext,this->Domains.ext+3, 0, multiplies<int>());
	this->Domains.numCNT = 0; // this->Domains.ttlCNT = 0; ???
	this->Domains.ttlCNT = 0;
	this->NeiOrder = 1;

	this->Grid.x = 0;	// to be set after mesh analysis by simPrms::device_config
	this->Grid.y = 1;
	this->Grid.z = 1;

	//cudaGetDeviceProperties(&curDevProp,currentDev);

	this->Block.x = 1024;//curDevProp.maxThreadsPerBlock;
	this->Block.y = 1;
	this->Block.z = 1;

	this->kernelSplit = 1;
	this->kernelSize = 0;
	
	// pointers to host and device arrays:

	this->mesh = 0;

	this->h_RNGseed = 0;
	this->h_result = 0;
	this->h_prefOrt = 0;
	this->h_phiDev = 0;
	this->h_phiMed = 0;
	this->h_thetaDev = 0;
	this->h_thetaMed = 0;

	this->d_RNGseed = 0;
	this->d_result = 0;
	this->d_prefOrt = 0;
	this->d_phiDev = 0;
	this->d_phiMed = 0;
	this->d_thetaDev = 0;
	this->d_thetaMed = 0;



}
simPrms::~simPrms(void)
{	
}

int		simPrms::set_ParamsFromInput(void) {
string	line;
string	keyWord;
string	prmString;
stringstream	prmStringStream;
int		posKeyW;

	initParMap();
	while (!cin.eof()) {
		getline(cin,line);
		if (!line.empty()) {
			posKeyW = line.find_first_of(" \t");
			keyWord = line.substr(0,posKeyW);
			prmString = line.substr(posKeyW+1,line.size());

			prmStringStream << prmString;

			prmString = rm_spacers(prmString);
			printf(" parameter string:%s\n", prmString.data());
			cout << " keyword " << keyWord << " line data " << prmString << " \n";
			
			switch (parMap[keyWord]) {

			case	WRT_BIN				:	prmStringStream >> this->wrt_bin_flag;
											
											break;
			case	VOLUME_PERCENT		:	prmStringStream >> this->vf;
											//printf("VOL_FRAC set to %f\n",this->vf);
											this->vf /= 100;
											break;
			case	VOL_FRAC_PRECISION	:	prmStringStream >> this->PRECISION;
											//printf("VOL_FRAC_PRECISION set to %f\n",this->PRECISION);
											break;
			case	SPAT_DISTR_FILE		:	this->crdDsrtFile = prmString;
											//printf("SPAT_DISTR_FILE set to %s\n",this->crdDsrtFile.data());
											break;
			case	PREF_ORT_PHI		:	prmStringStream >> this->def_prefOrt[0];
											def_prefOrt[0] = this->def_prefOrt[0]/180*PI;
											//printf("PREF_ORT_PHI set to %f rad\n",def_prefOrt[0]);
											break;
			case	PREF_ORT_THETA		:	prmStringStream >> this->def_prefOrt[1];
											this->def_prefOrt[1] = this->def_prefOrt[1]/180*Pi;
											//printf("PREF_ORT_THETA set to %f rad\n",def_prefOrt[1]);
											break;
			case	DEF_THETA_MED		:	prmStringStream >> this->def_thetaMed;
											this->def_thetaMed = this->def_thetaMed/180*Pi;
											//printf("DEF_THETA_MED set to %f rad\n",def_thetaMed);
											break;
			case	DEF_PHI_MED			:	prmStringStream >> this->def_phiMed;
											this->def_phiMed = this->def_phiMed/180*Pi;
											//printf("DEF_PHI_MED set to %f rad\n",def_phiMed);
											break;
			case	DEF_THETA_DEV		:	prmStringStream >> this->def_thetaDev;
											this->def_thetaDev = (this->def_thetaDev > 0 ? this->def_thetaDev/180*Pi : -1);
											//printf("DEF_THETA_DEV set to %f rad\n",def_thetaDev);
											break;
			case	DEF_PHI_DEV			:	prmStringStream >> this->def_phiDev;
											this->def_phiDev = (this->def_phiDev > 0 ? this->def_phiDev/180*Pi : -1);
											//printf("DEF_PHI_DEV set to %f rad\n",def_phiDev);
											break;
			case	CLUST_FLAG			:	prmStringStream >> this->clusterize_flag;
											break;
			case	CLUST_DIST_LIM		:	prmStringStream >> this->clusterize_dist;
											break;
			case	CLUST_ANGL_LIM		:	prmStringStream >> this->clusterize_angl;
											break;
			case	STAT_DIST_LIM		:	prmStringStream >> this->stat_dist_lim;
											break;
			case	SPAT_DIMENSIONS		:	break;
			case	ORNT_DISTR_FILE		:	break;
			case	IDIM_DISTR_FILE		:	break;
			case	INC_LEN_MED			:	prmStringStream >> this->l;
											break;
			case	INC_LEN_DEV			:	break;
			case	INC_RAD_MED			:	prmStringStream >> this->a;
											break;
			case	INC_RAD_DEV			:	break;
			case	INC_SEPARATION		:	break;
			case	INC_SOFTCORE		:	break;
			case	MATH_TOLERANCE		:	break;
			case	DMN_CELL_MARGE		:	break;
			default						:	printf("UNRECOGNISED STRING FOUND\n");
											break;			
			}
		}
	}

	return 1;
}

int		simPrms::set_ParamsFromFile(const char* filename) {
ifstream initF;
string	line;
string	keyWord;
string	prmString;
stringstream	prmStringStream;
int		posKeyW;

	initF.open(filename,ios::in);
	if (initF.is_open()) {
		initParMap();
		while (!initF.eof()) {
			getline(initF,line);
			if (!line.empty()) {
				posKeyW = line.find_first_of(" \t");
				keyWord = line.substr(0,posKeyW);
				prmString = line.substr(posKeyW+1,line.size());

				prmString = rm_spacers(prmString);
				//printf(" parameter string:%s\n", prmString.data());
				//cout << " keyword " << keyWord << " line data " << prmString << " \n";

				prmStringStream.clear();
				prmStringStream.str(prmString);
			
				switch (parMap[keyWord]) {

				case	WRT_BIN				:	prmStringStream >> this->wrt_bin_flag;
												//printf("WRT_BIN set to %i\n",this->wrt_bin_flag);
												break;
				case	VOLUME_PERCENT		:	prmStringStream >> this->vf;
												//printf("VOL_FRAC set to %f\n",this->vf);
												this->vf /= 100; // convert % into fraction
												break;
				case	VOLUME_PERCENT_LIST :	prmStringStream >> this->vfcount;
												this->vfs = (float*) malloc(this->vfcount*sizeof(float));
												for(int i=0;i<this->vfcount;i++) {
													prmStringStream >> this->vfs[i];
													this->vfs[i] /=100;
												}
												break;

				case	VOL_FRAC_PRECISION	:	prmStringStream >> this->PRECISION;
												//printf("VOL_FRAC_PRECISION set to %f\n",this->PRECISION);
												break;
				case	SPAT_DISTR_FILE		:	prmStringStream >> this->crdDsrtFile;
												if (this->crdDsrtFile == "NONE") {
													prmStringStream >> this->physDim[0] >> this->physDim[1] >> this->physDim[2];
													prmStringStream >> this->ext[0] >> this->ext[1] >> this->ext[2];
												}

												break;
				case	PREF_ORT_PHI		:	prmStringStream >> this->def_prefOrt[0];
												def_prefOrt[0] = this->def_prefOrt[0]/180*PI;
												//printf("PREF_ORT_PHI set to %f rad\n",def_prefOrt[0]);
												break;
				case	PREF_ORT_THETA		:	prmStringStream >> this->def_prefOrt[1];
												this->def_prefOrt[1] = this->def_prefOrt[1]/180*Pi;
												//printf("PREF_ORT_THETA set to %f rad\n",def_prefOrt[1]);
												break;
				case	DEF_THETA_MED		:	prmStringStream >> this->def_thetaMed;
												this->def_thetaMed = this->def_thetaMed/180*Pi;
												//printf("DEF_THETA_MED set to %f rad\n",def_thetaMed);
												break;
				case	DEF_PHI_MED			:	prmStringStream >> this->def_phiMed;
												this->def_phiMed = this->def_phiMed/180*Pi;
												//printf("DEF_PHI_MED set to %f rad\n",def_phiMed);
												break;
				case	DEF_THETA_DEV		:	prmStringStream >> this->def_thetaDev;
												this->def_thetaDev = (this->def_thetaDev > 0 ? this->def_thetaDev/180*Pi : -1);
												//printf("DEF_THETA_DEV set to %f rad\n",def_thetaDev);
												break;
				case	DEF_PHI_DEV			:	prmStringStream >> this->def_phiDev;
												this->def_phiDev = (this->def_phiDev > 0 ? this->def_phiDev/180*Pi : -1);
												//printf("DEF_PHI_DEV set to %f rad\n",def_phiDev);
												break;
				case	SPAT_DIMENSIONS		:	break;
				case	ORNT_DISTR_FILE		:	break;
				case	IDIM_DISTR_FILE		:	break;
				case	INC_LEN_MED			:	prmStringStream >> this->l;
												break;
				case	INC_LEN_DEV			:	break;
				case	INC_RAD_MED			:	prmStringStream >> this->a;
												break;
				case	INC_RAD_DEV			:	break;
				case	INC_SEPARATION		:	prmStringStream >> this->sep;
												break;
				case	INC_SOFTCORE		:	prmStringStream >> this->sc_frac;
												break;
				case	MATH_TOLERANCE		:	break;
				case	DMN_CELL_MARGE		:	break;
				case	CLUST_FLAG			:	prmStringStream >> this->clusterize_flag;
												break;
				case	CLUST_DIST_LIM		:	prmStringStream >> this->clusterize_dist;
												break;
				case	CLUST_ANGL_LIM		:	prmStringStream >> this->clusterize_angl;
												break;
				case	STAT_DIST_LIM		:	prmStringStream >> this->stat_dist_lim;
												break;
				case	SELF_ALLIGN_STEP	:	prmStringStream >> this->self_alling_step;
												//printf("SELF_ALLIGN_STEP is set to %i\n",this->self_alling_step);
												break;
				case	REDUCE_FLAG			:	prmStringStream >> this->reduce_flag;
												//printf("REDUCE_FLAG is set to %i\n",this->reduce_flag);
												break;
				default:		printf("UNRECOGNISED STRING FOUND\n");
								break;			
				}
			}
		}
		return 1;
	} else {
		printf("File %s not found!!! \n",filename);
		return 0;
	}
}
//-----------------------------------------------------------------------------------------------------
CNT_t	*simPrms::asmbl_cnt_data(void) {

		CNT_t *h_cnt;
		int s = 9;
		int tot = 0;

		this->h_result = (float *) malloc((size_t) s*this->Domains.ttlCNT * sizeof(float));
		this->h_res.x  = &(this->h_result[0*this->Domains.ttlCNT]);
		this->h_res.y  = &(this->h_result[1*this->Domains.ttlCNT]);
		this->h_res.z  = &(this->h_result[2*this->Domains.ttlCNT]);
		this->h_res.cx = &(this->h_result[3*this->Domains.ttlCNT]);
		this->h_res.cy = &(this->h_result[4*this->Domains.ttlCNT]);
		this->h_res.cz = &(this->h_result[5*this->Domains.ttlCNT]);
		this->h_res.l  = &(this->h_result[6*this->Domains.ttlCNT]);
		this->h_res.a  = &(this->h_result[7*this->Domains.ttlCNT]);
		this->h_res.k  = &(this->h_result[8*this->Domains.ttlCNT]);

		cudaError cuErr = cudaMemcpy((void*) this->h_result,(void*) this->d_result,s*this->Domains.ttlCNT*sizeof(float),cudaMemcpyDeviceToHost);
		if (cuErr!=cudaSuccess)
			printf(" ERROR OCCURED WHILE COPYING DATA: %s !!!\n",cudaGetErrorString(cuErr));
		else
			if (this->Domains.ttlCNT > 0) {
				h_cnt =(CNT_t *) malloc(this->numCNT*sizeof(CNT_t));
				CNT_t *cntPtr = h_cnt;
				for(int i=0;i<this->Domains.ttlCNT;i++) //{
					if (this->h_res.l[i] > this->TOL) {
						cntPtr->r.x = this->h_res.x[i];
						cntPtr->r.y = this->h_res.y[i];
						cntPtr->r.z = this->h_res.z[i];
						cntPtr->c.x = this->h_res.cx[i];
						cntPtr->c.y = this->h_res.cy[i];
						cntPtr->c.z = this->h_res.cz[i];
						cntPtr->a   = this->h_res.a[i];
						cntPtr->l   = this->h_res.l[i];
						cntPtr->k	= this->h_res.k[i];
						cntPtr++;
#ifdef D_CHECK_RESULT
					} else { 
						int x0 = (((int)this->h_res.k[i])%this->Domains.ext[0]);
						int y0 = (((int)this->h_res.k[i])/this->Domains.ext[0])%this->Domains.ext[1];
						int z0 = (((int)this->h_res.k[i])/this->Domains.ext[0])/this->Domains.ext[1];
	
						bool isX = ((x0+this->TOL)*this->physDim[0]/this->Domains.ext[0] <= this->h_res.x[i])&&(this->h_res.x[i] < (x0+1-this->TOL)*this->physDim[0]/this->Domains.ext[0]);
						bool isY = ((y0+this->TOL)*this->physDim[1]/this->Domains.ext[1] <= this->h_res.y[i])&&(this->h_res.y[i] < (y0+1-this->TOL)*this->physDim[1]/this->Domains.ext[1]);
						bool isZ = ((z0+this->TOL)*this->physDim[2]/this->Domains.ext[2] <= this->h_res.z[i])&&(this->h_res.z[i] < (z0+1-this->TOL)*this->physDim[2]/this->Domains.ext[2]);

						if ((this->h_res.k[i] < 0)||(!(isX&&isY&&isZ))) {
							printf("% i - ERRONEOUS WRITE!!! \n",i);
							printf("%10.6f %10.6f %10.6f \n",h_res.x[i],h_res.y[i],h_res.z[i]);
							printf("%10.6f %10.6f %10.6f \n",h_res.cx[i],h_res.cy[i],h_res.cz[i]);
							printf("%10.6f %10.6f %10.6f \n",h_res.a[i],h_res.l[i],h_res.k[i]);
							printf("%10.6f %10.6f %10.6f \n",x0*this->physDim[0],y0*this->physDim[1],z0*this->physDim[2]);
							getchar();
						}
					}
#endif
						tot++;
				}
					printf("CNTs detected: %i \n",tot);
		free(this->h_result);
	}
	return h_cnt;
}

int		 simPrms::allocateDevArr(void) {
	if (this->Grid.x == 0) {
		printf("CUDA Grid is not set! \n");
		return 0;
	} else {

		// Domain adresses:
		int Mem_DMN = set_dmnAddr();
		//printf("Domain adresses: %i \n",Mem_DMN);

		// Block distribution:
		int Mem_MSK = set_Masks();
		//printf("Block distribution: %i \n",Mem_MSK);

		int Mem_ORT = set_dmnOrt();
		//printf("Per domain orientation: %i \n",Mem_MSK);

		// CUDA RNG configure:

#ifdef _DEBUG
	int seed = 1234;
#else
	int seed = clock();
#endif
	//int seed = 1234;
		int	size = this->Block.x*this->Grid.x*this->Block.y*this->Grid.y*this->Block.z*this->Grid.z;
		cudaMalloc((void **) &(this->d_RNGseed),size*sizeof(int));
		int Mem_RNG = seed_cuRND(seed);
		//printf("CUDA RNG configure: %i \n",Mem_RNG);
		
		// allocate arrays for results:
		int Mem_Res = set_Result();
		//printf("allocate arrays for results: %i \n",Mem_Res);

		return Mem_DMN + Mem_MSK + Mem_RNG + Mem_Res;
	}

}

int		 simPrms::device_config(void) {
	int currentDev = -1;
	cudaDeviceProp curDevProp;
	cudaGetDevice(&currentDev);
	if (currentDev == -1) {
		printf("Device is not set!!! \n");
		return 0;
	} else {
	if (this->Domains.ext[0] == 0) {
		printf("Domain Grid is not set!!! \n");
		return 0;
	} else {
		// set Grid and block dim according to Device limits:
		// Block - is set in class constructor - 1 block per SM

		cudaGetDeviceProperties(&curDevProp,currentDev);
		long DevMem = curDevProp.totalGlobalMem;
				
		//Memory expenses:
		// fixed:
		int MemTex = this->ext[0]*this->ext[1]*this->ext[2]*sizeof(float);		// texture memory
		int	MemMask = this->Domains.segNum*6*sizeof(int);						// memory for load-levelled segments
		int	MemNcnt = 3*this->Domains.ext[0]*this->Domains.ext[1]
					 *this->Domains.ext[2]*sizeof(int);							// memory for Domain cells adresses,occupation values, fresh cnts (occupation and new cnt numbers are short!) and locks.
		// proportional to CUDA Thread Number (per thread):
		int MemCNT = sizeof(CNT_t);												// memory to store 1 CNT
		int MemRNG = sizeof(int);												// memory to seed 1 thread RNG

		int	MaxDevThr = (curDevProp.totalGlobalMem - (MemTex + MemMask + MemNcnt) - MEM_RESERVE)
						/(MemCNT+MemRNG);
		int MaxDevBlk = MaxDevThr /( this->Block.x*this->Block.y*this->Block.z);
		MaxDevBlk /= curDevProp.multiProcessorCount;
		MaxDevBlk *= curDevProp.multiProcessorCount;

		// Grid needed for generation:
		//int	needThrBlk = ceil(((float)this->numCNT)/this->Block.x);
		int	needThrBlk = this->Domains.segNum;
		int maxGrdBlk = curDevProp.maxGridSize[0];
		//maxGrdBlk = (MaxDevBlk<maxGrdBlk?MaxDevBlk:maxGrdBlk);
		this->Grid.x = (needThrBlk<maxGrdBlk?needThrBlk:maxGrdBlk);
		this->Grid.y = 1;
		this->Grid.z = 1;

		// Grid needed for domain processing:
		this->kernelSplit = (int)  ceil(((float)this->Domains.grdSize)/curDevProp.maxGridSize[0]);
		this->kernelSize = (this->kernelSplit<=1 ? this->Domains.grdSize : curDevProp.maxGridSize[0] );

		int arr_size = this->Grid.x * this->Block.x * this->Grid.y * this->Block.y * this->Grid.z * this->Block.z;
		
#ifdef _DEBUG
		printf("_DEBUG:______________________________________________________________________________\n");
		printf("_DEBUG: Found %lld Global Memory on Device - %i blocks could store data \n",DevMem,MaxDevBlk);
		printf("_DEBUG: Task requirements are %i blocks \n",needThrBlk);
		printf("_DEBUG: Opt to %i Blocks per Grid \n",Grid.x);

		printf("_DEBUG: Memory usage: \n");
		printf("_DEBUG: Total memory	: %8.3f \n",((float)curDevProp.totalGlobalMem)/MegaByte);
		printf("_DEBUG: Texture memory  : %8.3f \n",((float)MemTex)/MegaByte);
		printf("_DEBUG: Segment memory  : %8.3f \n",((float)MemMask)/MegaByte);
		printf("_DEBUG: Occupancy memory: %8.3f \n",((float)MemNcnt)/MegaByte);
		printf("_DEBUG: Results memory  : %8.3f \n",((float)MemCNT*this->Domains.ttlCNT)/MegaByte);
		printf("_DEBUG: cuRAND memory   : %8.3f \n",((float)MemRNG*arr_size)/MegaByte);
		printf("_DEBUG: Reserve memory  : %8.3f \n",((float)MEM_RESERVE)/MegaByte);
		//*/
		printf("_DEBUG: Total threads in Grid: %i \n",arr_size);
		printf("_DEBUG:______________________________________________________________________________\n");
#endif

		return curDevProp.maxGridSize[0];//curDevProp.sharedMemPerBlock;
	}}
}


int		 simPrms::distributeDmn(int MaxLoad) {
	//kdTree dmnKdT = kdTree(this->numCNT,this->Domains.ext,this->Domains.numCNT);
	kdTree dmnKdT = kdTree(this->Domains.ttlCNT,this->Domains.ext,this->Domains.numCNT);
	dmnKdT.build(this->Block.x);
	//dmnKdT.print("ColoredMesh_base.vtk");
	int *leaf_msk = dmnKdT.leaves();
	for(int j = 0;j<6;j++)
		this->Domains.mask[j] = (int *) malloc(dmnKdT.leafNum*sizeof(int));

	for(int i = 0;i<dmnKdT.leafNum;i++)
		for(int j = 0;j<6;j++)
			this->Domains.mask[j][i] = leaf_msk[6*i+j];

	free(leaf_msk);
	
	return dmnKdT.leafNum;
}
int		simPrms::reDistributeDmn(int iter, int maxLoad ) {
	char	fileN[255] = "";
	//sprintf(fileN,"ColoredMesh_%i.vtk",iter);

	int size = this->Domains.grdSize;
	int	*mesh = (int*) malloc(size*sizeof(int));
	short *h_Occ = (short*) malloc(size*sizeof(short));
	cudaMemcpy((void*) h_Occ,(void*) this->Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);
	int Load = this->Domains.ttlCNT;
	int curLd = 0;
	for(int i=0;i<size;i++) {
		mesh[i] = this->Domains.numCNT[i] - h_Occ[i];
		curLd += h_Occ[i];
	}
	Load -= curLd;
	kdTree dmnKdT = kdTree(Load,this->Domains.ext,mesh);
	dmnKdT.build(this->Block.x);
#ifdef _DEBUG
	dmnKdT.print(fileN);
#endif
	int *leaf_msk = dmnKdT.leaves();
	
	for(int j = 0;j<6;j++) {
		free(this->Domains.mask[j]);
		this->Domains.mask[j] = (int *) malloc(dmnKdT.leafNum*sizeof(int));
	}

	for(int i = 0;i<dmnKdT.leafNum;i++)
		for(int j = 0;j<6;j++)
			this->Domains.mask[j][i] = leaf_msk[6*i+j];
	free(leaf_msk);
	free(h_Occ);
	free(mesh);
	return dmnKdT.leafNum;
}

int		 simPrms::make_den(const char *name) {
	this->mesh = read_density(name,this->ext, this->physDim);
	if (this->ext[0]==0) {
		printf("Mesh file %s access error!!! \n",name);
		return 0;
	} else {
#ifdef _DEBUG
		printf("_DEBUG: Density map is read succesfully \n");
#endif
		return 1;
	}
}
int		 simPrms::make_default_den(void) {

	this->mesh = (float*) malloc(sizeof(float)*this->ext[0]*this->ext[1]*this->ext[2]);

	for(int i=0;i<this->ext[0];i++) 
		for(int j=0;j<this->ext[1];j++)
			for(int k=0;k<this->ext[2];k++) {
				/*mesh[i*this->ext[1]*this->ext[2]+j*this->ext[2]+k] = sin(2*2*Pi/this->ext[0]*i)*
																	 sin(2*2*Pi/this->ext[1]*j)*
																	 sin(2*2*Pi/this->ext[2]*k)+1; */
				/*mesh[i*this->ext[1]*this->ext[2]+j*this->ext[2]+k] = exp(-10*powf((i-0.5*this->ext[0])/this->ext[0],2))*
																	 exp(-10*powf((j-0.5*this->ext[1])/this->ext[1],2))*
																	 exp(-10*powf((k-0.5*this->ext[2])/this->ext[2],2));*/
				mesh[i*this->ext[1]*this->ext[2]+j*this->ext[2]+k] = 0.5f;	
			}

#ifdef _DEBUG
		printf("_DEBUG: Density map is created succesfully \n");
#endif
		return 1;
}


int		 simPrms::make_dmnGrid(int maxLoad,float margin) {
	if ((this->physDim[0] == 0)||(this->numCNT == 0)||(this->l == 0)) {
		printf("Prerequisites for domain grid building are not set!!! \n");
		return 0;
	} else {
		int	dmnExt[3];
		int	size;
		int Load =0;
		double rescale;
		//double len = this->l;
		//double len = this->physDim[0];
		//double len = this->physDim[0];
		double len = *std::min_element(this->physDim,this->physDim+3);
		printf("Initial box length is %f\n",len);
		float *h_dmnInt;
#ifdef _DEBUG
		printf("_DEBUG:______________________________________________________________________________\n");
#endif
		do {
			dmnExt[0]=(int) floor((this->physDim[0])/len);
			dmnExt[1]=(int) floor((this->physDim[1])/len);
			dmnExt[2]=(int) floor((this->physDim[2])/len);
			size = dmnExt[0]*dmnExt[1]*dmnExt[2];
			h_dmnInt = (float *) malloc( size*sizeof(float));

			double sum = integrateDmnGrid(dmnExt,h_dmnInt);
            printf("Density integral: %f\n",sum);
			rescale = this->numCNT/sum;

			Load = 0;
			int loadI;
			for (int i=0;i<size;i++) {
				//loadI =(int)  floor(rescale*h_dmnInt[i]+0.5) + margin;
				loadI =(int)  floor(rescale*h_dmnInt[i]*(1+margin)+0.5);
				if (Load<loadI)
					Load = loadI;
			}
#ifdef _DEBUG
			printf("_DEBUG: Edge lenght: %f - maxLoad: %i \n",len,Load);
			
#endif
			len*=pow(((double)maxLoad)/Load,1./3.);
		} while(Load > maxLoad);

		this->Domains.ext[0]=dmnExt[0];
		this->Domains.ext[1]=dmnExt[1];
		this->Domains.ext[2]=dmnExt[2];
		this->Domains.grdSize = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];
		transform(	this->physDim,
					this->physDim+3,
					this->Domains.ext,
					this->Domains.edge,
					divides<float>());
		this->Domains.numCNT = (int *) malloc( size*sizeof(int));
		int	Ncnt = 0;
		for (int d = 0;d<size;d++) {
			this->Domains.numCNT[d]=(int) floor(rescale*h_dmnInt[d]*(1+margin)+0.5);
			Ncnt += this->Domains.numCNT[d];
		}
		
		this->Domains.ttlCNT = Ncnt;
		this->NeiOrder =(int)  ceil(this->l/(*min_element(this->Domains.edge,this->Domains.edge+3)));
#ifdef _DEBUG
		printf("_DEBUG: Domain grid dimensions: %i %i %i, edge %e x %e x %e \n",this->Domains.ext[0],this->Domains.ext[1],this->Domains.ext[2],this->Domains.edge[0],this->Domains.edge[1],this->Domains.edge[2]);
		printf("_DEBUG: %i inclusions expected - allocating %i vacancies \n",this->numCNT,Ncnt);
		printf("_DEBUG: Intersection will be searched in neighbouring cells up to %i order \n",this->NeiOrder);
		printf("_DEBUG:______________________________________________________________________________\n");
#endif
		return 1;
	}
}


int		 simPrms::maxLeaf(void) {

	int maxVol = -1;
	int vol;
#ifdef _DEBUG
	printf("_DEBUG: Segnum: %i \n",this->Domains.segNum);
#endif
	for(int i=0; i < this->Domains.segNum; i++) {
		vol = 1;
		for (int j=0; j<3; j++) 
			vol *= this->Domains.mask[2*j+1][i] - this->Domains.mask[2*j][i] + 1;
#ifdef VERBOUS
		//if (this->Domains.mask[2][i]==0)
		//if (i==1)
		printf("_DEBUG: Leaf %i vol %i mask: %i-%i %i-%i %i-%i  \n",i,vol,	this->Domains.mask[0][i],
											this->Domains.mask[1][i],
											this->Domains.mask[2][i],
											this->Domains.mask[3][i],
											this->Domains.mask[4][i],
											this->Domains.mask[5][i]); 
#endif
		if (vol>maxVol) maxVol = vol;
	}
	return maxVol;
}
int		 simPrms::repack(void) {
	printf("Repack:... ");
	int size = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];
	int s = 9;
	cudaError cuErr;

	int	*h_dmnAddr_Old = (int *) malloc(size*sizeof(int));
	int	*h_dmnAddr_New = (int *) malloc(size*sizeof(int));
	short	*h_occ = (short *) malloc(size*sizeof(short));
	cudaMemcpy((void*) h_dmnAddr_Old,(void*) this->Domains.d_dmnAddr,size*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy((void*) h_occ,(void*) this->Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);

	int	accum = 0; 
	// optimize
	for(int i=0;i<size;i++) {
		accum += h_occ[i];
		h_dmnAddr_New[i] = accum;
	}
	printf("Found %i in total... ",accum);
	cuErr = cudaMemcpy((void*) this->Domains.d_dmnAddr,(void*) h_dmnAddr_New,size*sizeof(int),cudaMemcpyHostToDevice);
	if (cuErr!=cudaSuccess) printf(" ERROR OCCURED WHILE COPYING ADDRESS DATA TO DEVICE: %s !!!\n",cudaGetErrorString(cuErr));

	// refactoring inclusion array:
	float *h_inclusions_old = (float *) malloc((size_t) s*this->Domains.ttlCNT * sizeof(float));
	float *h_inclusions_new = (float *) malloc((size_t) s*accum * sizeof(float));

	cuErr = cudaMemcpy((void*) h_inclusions_old,(void*) this->d_result,s*this->Domains.ttlCNT*sizeof(float),cudaMemcpyDeviceToHost);
	if (cuErr!=cudaSuccess) printf(" ERROR OCCURED WHILE COPYING INCLUSION DATA FROM DEVICE: %s !!!\n",cudaGetErrorString(cuErr));

	for(int i=0;i<size;i++) {
		int start_old = ( i>0 ? h_dmnAddr_Old[i-1] : 0);
		int start_new = ( i>0 ? h_dmnAddr_New[i-1] : 0);
		for(int k=0;k<9;k++)
			for(int j=0; j<h_occ[i]; j++) 
				h_inclusions_new[start_new+j+accum*k] = h_inclusions_old[start_old+j+this->Domains.ttlCNT*k];
	}
	free(h_inclusions_old);
	free(h_occ);
	cudaFree(this->d_result);
	//cudaFree(this->Domains.d_dmnOcc);

	cuErr = cudaMalloc((void**) &this->d_result,accum*s*sizeof(float));
	if (cuErr!=cudaSuccess) printf(" ERROR OCCURED WHILE ALLOCATING DEVICE MEM: %s !!!\n",cudaGetErrorString(cuErr));
	cuErr = cudaMemcpy((void*) this->d_result,(void*) h_inclusions_new,s*accum*sizeof(float),cudaMemcpyHostToDevice);
	if (cuErr!=cudaSuccess) printf(" ERROR OCCURED WHILE COPYING INCLUSION DATA TO DEVICE: %s !!!\n",cudaGetErrorString(cuErr));
	free(h_inclusions_new);

	int diff = this->Domains.ttlCNT - this->numCNT;
	printf("space saved %f %% \n",(float)diff/this->Domains.ttlCNT*100);
	this->Domains.ttlCNT = accum;

	return cuErr==cudaSuccess; // resulting inclusion number
}
int		 simPrms::expand(int dir) {
	printf("Expanding in %d direction:... \n",dir);

	// Expanded grid dimensions:
	int newGridExt[3];
	float  newPhysDim[3];
	copy(this->Domains.ext, this->Domains.ext+3, newGridExt);
	copy(this->physDim, this->physDim+3, newPhysDim);

	/*
	this->NeiOrder = (int) ceil((this->l+this->stat_dist_lim)/this->physDim[0]*this->Domains.ext[0]);
	for(char k=0;k<3;k++)
		if(dir&(1<<k))	{
			newGridExt[k] += 2*this->NeiOrder;
			//newPhysDim[k] = this->physDim[k]/this->Domains.ext[0]*newGridExt[k];
			newPhysDim[k] = this->physDim[k]/this->Domains.ext[k]*newGridExt[k];
		}
	//*/
	int k = dir>>1;
	this->NeiOrder = (int) ceil((this->l+this->stat_dist_lim)/this->physDim[k]*this->Domains.ext[k]);
	newGridExt[k] += 2*this->NeiOrder;
	newPhysDim[k] = this->physDim[k]/this->Domains.ext[k]*newGridExt[k];

	int size_old = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];
	int size_new = newGridExt[0]*newGridExt[1]*newGridExt[2];
	int s = 9;

	// Move data to host for reogranization:
	vector<int>		h_dmnAddr_Old(size_old);
	vector<int>		h_dmnAddr_New(size_new);
	vector<short>	h_dmnOcc_Old(size_old,0);
	vector<short>	h_dmnOcc_New(size_new,0);
	cudaMemcpy((void*) h_dmnAddr_Old.data(),	(void*) this->Domains.d_dmnAddr,  size_old*sizeof(int),    cudaMemcpyDeviceToHost);
	cudaMemcpy((void*) h_dmnOcc_Old.data(),		(void*) this->Domains.d_dmnOcc,	  size_old*sizeof(short),  cudaMemcpyDeviceToHost);


	int offset_old	= this->Domains.ttlCNT;
	int	accum_old	= accumulate(h_dmnOcc_Old.begin(), h_dmnOcc_Old.end(),0);
	printf("Found %i/%i in total...",accum_old,offset_old);

	// Set new domain addresses:
	for(int i=0;i<size_new;i++) {
		int idx0 = dmnIdx_new2old(i,this->Domains.ext,newGridExt);
		h_dmnOcc_New[i] = h_dmnOcc_Old[idx0];
	}
	vector<int> buff(h_dmnOcc_New.begin(), h_dmnOcc_New.end());
	std::partial_sum(buff.begin(), buff.end(),h_dmnAddr_New.begin());
	int	offset_new = h_dmnAddr_New[size_new-1];
	int	accum_new = std::accumulate(h_dmnOcc_New.begin(), h_dmnOcc_New.end(),0);
	printf(" will expand to %i/%i\n",accum_new,offset_new);
	vector<int>().swap(buff);

	// refactoring inclusion array:
	float *h_inclusions_old = (float *) malloc((size_t) s*offset_old * sizeof(float));
	float *h_inclusions_new = (float *) malloc((size_t) s*offset_new * sizeof(float));
	cudaMemcpy((void*) h_inclusions_old,		(void*) this->d_result,			s*offset_old*sizeof(float),cudaMemcpyDeviceToHost);

	for(int idx1=0;idx1<size_new;idx1++) {
		int idx0 = dmnIdx_new2old(idx1,this->Domains.ext,newGridExt);
		int start_old = ( idx0>0 ? h_dmnAddr_Old[idx0-1] : 0);
		int start_new = ( idx1>0 ? h_dmnAddr_New[idx1-1] : 0);
		int size2copy = h_dmnOcc_New[idx1];
		vector<float>	offset = offset_new2old(idx1, this->Domains.ext, newGridExt, this->physDim);
		vector<float>	addval(s,0.0);
		copy(offset.begin(),offset.end(), addval.begin());
		for(int k=0;k<s;k++)	{
			// copy from old to new storage adding offset:
			transform(	h_inclusions_old + start_old + offset_old*k,
						h_inclusions_old + start_old + offset_old*k + size2copy,
						h_inclusions_new + start_new + offset_new*k,
						bind2nd(plus<float>(),addval[k])				);
		}
	}

	// Move expanded data back GPU:
	cudaFree(this->d_result);
	cudaFree(this->Domains.d_dmnAddr);
	cudaFree(this->Domains.d_dmnOcc);

	cudaMalloc((void**) &this->Domains.d_dmnAddr,	size_new*sizeof(int));
	cudaMalloc((void**) &this->Domains.d_dmnOcc,	size_new*sizeof(short));
	cudaMalloc((void**) &this->d_result,		s*offset_new*sizeof(float));

	cudaMemcpy((void*) this->Domains.d_dmnAddr,	(void*) h_dmnAddr_New.data(),	size_new*sizeof(int),	cudaMemcpyHostToDevice);
	cudaMemcpy((void*) this->Domains.d_dmnOcc,	(void*) h_dmnOcc_New.data(),	size_new*sizeof(short),	cudaMemcpyHostToDevice);
	cudaMemcpy((void*) this->d_result,			(void*) h_inclusions_new,	s*offset_new*sizeof(float),	cudaMemcpyHostToDevice);

	vector<int>().swap(h_dmnAddr_Old);
	vector<int>().swap(h_dmnAddr_New);
	vector<short>().swap(h_dmnOcc_Old);
	vector<short>().swap(h_dmnOcc_New);
	free(h_inclusions_new);
	free(h_inclusions_old);

	// correct grid dimensions:


	copy(newGridExt,newGridExt+3,this->Domains.ext);
	copy(newPhysDim,newPhysDim+3,this->physDim);
	this->Domains.ttlCNT = offset_new;
	this->Domains.grdSize = size_new;
	this->numCNT = accum_new;
	printf("New grid extents: %i - %i - %i, %f - %f - %f new cnt number: %i\n",
				this->Domains.ext[0],this->Domains.ext[1],this->Domains.ext[2],
				this->physDim[0],this->physDim[1],this->physDim[2],
				this->Domains.ttlCNT);
	this->device_config();
	return accum_new;
}
int		 simPrms::seed_cuRND(int seed) {
	int	size = this->Block.x*this->Grid.x*this->Block.y*this->Grid.y*this->Block.z*this->Grid.z;
	this->cuRNG_Num = size;
	this->h_RNGseed =(int *) malloc(size*sizeof(int));
	
	srand(seed);
	for (int i=0;i<size;i++)
		this->h_RNGseed[i] = rand();
	cudaMemcpy(this->d_RNGseed,this->h_RNGseed,size*sizeof(int),cudaMemcpyHostToDevice);

	free(h_RNGseed);
	return size*sizeof(int); 

}
int		 simPrms::set_Masks() {
	int size = this->Domains.segNum; // size - the number of active Blocks
	int	*h_masks = (int *) malloc(6*size*sizeof(int));
	cudaMalloc((void**) &this->Domains.d_masks,6*size*sizeof(int));
	
	for(int i=0;i<size;i++) 
		for(int j=0;j<6;j++)
			h_masks[j*size+i] = (int) (this->Domains.mask[j][i]);

	cudaMemcpy((void*) this->Domains.d_masks,(void*) h_masks,6*size*sizeof(int),cudaMemcpyHostToDevice);
	
	free(h_masks);
	return 6*size*sizeof(int);
}
int		 simPrms::set_dmnAddr() {
	int size = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];

	int	*h_dmnAddr = (int *) malloc(size*sizeof(int));
	int	*h_Lck = (int *) malloc(size*sizeof(int));
	short	*h_occ = (short *) malloc(size*sizeof(short));
	int	accum = 0; 
	for(int i=0;i<size;i++) {
		accum += this->Domains.numCNT[i];
		h_dmnAddr[i] = accum;
	}
	cudaMalloc((void**) &this->Domains.d_dmnAddr,size*sizeof(int));
	cudaMemcpy((void*) this->Domains.d_dmnAddr,(void*) h_dmnAddr,size*sizeof(int),cudaMemcpyHostToDevice);
	free(h_dmnAddr);

	for(int i=0;i<size;i++) h_occ[i] = 0;
	// set current occupancy:
	cudaMalloc((void**) &this->Domains.d_dmnOcc,size*sizeof(short));
	cudaMemcpy((void*) this->Domains.d_dmnOcc,(void*) h_occ,size*sizeof(short),cudaMemcpyHostToDevice);
	// set newly created cnt numbers:
	cudaMalloc((void**) &this->Domains.d_dmnCrt,size*sizeof(short));
	cudaMemcpy((void*) this->Domains.d_dmnCrt,(void*) h_occ,size*sizeof(short),cudaMemcpyHostToDevice);
	free( h_occ);
	// set locks:												// <--- dont need this!!!
	for(int i=0;i<size;i++) h_Lck[i] = -1;
	cudaMalloc((void**) &this->Domains.d_dmnLck,size*sizeof(int));
	cudaMemcpy((void*) this->Domains.d_dmnLck,(void*) h_Lck,size*sizeof(int),cudaMemcpyHostToDevice);
	free(h_Lck);
	
	return size*sizeof(int);
}
int		 simPrms::set_dmnOrt(void) {
	int size = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];

	this->h_prefOrt = (float*) malloc(2*size*sizeof(float));
	for(int i=0;i<size;i++) {
		//this->h_prefOrt[i+0*size] = (Pi*(i%this->Domains.ext[0]))/this->Domains.ext[0];
		//this->h_prefOrt[i+1*size] = (Pi*(i%this->Domains.ext[0]))/this->Domains.ext[0];

		this->h_prefOrt[i+0*size] = this->def_prefOrt[0];
		this->h_prefOrt[i+1*size] = this->def_prefOrt[1];
	}
	cudaMalloc((void**)&this->d_prefOrt,2*size*sizeof(float));
	cudaMemcpy(this->d_prefOrt,this->h_prefOrt,2*size*sizeof(float),cudaMemcpyHostToDevice);
	free(this->h_prefOrt);

	this->h_thetaMed = (float*) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) {
		this->h_thetaMed[i] = this->def_thetaMed;
	}
	cudaMalloc((void**)&this->d_thetaMed,size*sizeof(float));
	cudaMemcpy(this->d_thetaMed,this->h_thetaMed,size*sizeof(float),cudaMemcpyHostToDevice);
	free(this->h_thetaMed);	

	this->h_thetaDev = (float*) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) {
		this->h_thetaDev[i] = this->def_thetaDev;
	}
	cudaMalloc((void**)&this->d_thetaDev,size*sizeof(float));
	cudaMemcpy(this->d_thetaDev,this->h_thetaDev,size*sizeof(float),cudaMemcpyHostToDevice);
	free(this->h_thetaDev);

	this->h_phiMed = (float*) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) {
		this->h_phiMed[i] = this->def_phiMed;
	}
	cudaMalloc((void**)&this->d_phiMed,size*sizeof(float));
	cudaMemcpy(this->d_phiMed,this->h_phiMed,size*sizeof(float),cudaMemcpyHostToDevice);
	free(this->h_phiMed);	

	this->h_phiDev = (float*) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) {
		this->h_phiDev[i] = this->def_phiDev;
	}
	cudaMalloc((void**)&this->d_phiDev,size*sizeof(float));
	cudaMemcpy(this->d_phiDev,this->h_phiDev,size*sizeof(float),cudaMemcpyHostToDevice);
	free(this->h_phiDev);
	
	return 6*size*sizeof(float);
}

int		 simPrms::set_Result(void) {
	int arr_size = this->Domains.ttlCNT;
	int s = 9;
	cudaError_t cuErr;
		// HOST
		this->h_result = (float *) malloc((size_t) s*arr_size * sizeof(float));
		// DEVICE
		cuErr = cudaMalloc((void**) &(this->d_result), s*arr_size * sizeof(float));
		// initiate results with zeros:
		for(int i=0;i<s*arr_size;i++)	h_result[i] = 0.0f;
		cudaMemcpy((void*) this->d_result,(void*) this->h_result,s*arr_size*sizeof(float),cudaMemcpyHostToDevice); 
		free(this->h_result);
#ifdef _DEBUG
		printf("_DEBUG: Results' memory initialized with zeros. Array size: %li ",s*arr_size*sizeof(float));
		cuErr = cudaGetLastError();
		printf( cudaGetErrorString(cuErr) );
		printf( " !!!\n" );
#endif
	if (cuErr == cudaSuccess) {
		return s*arr_size*sizeof(float);
	} else {
		return 0;
	}
}

int		simPrms::set_numCNT(float length, float rad, float volume_fraction) {
		this->l = length;
		this->a = rad;
		double vcnt = (PI*this->a*this->a*this->l);
		double	vol = this->physDim[0]*this->physDim[1]*this->physDim[2];
		this->numCNT = (int) ceil(volume_fraction*vol/vcnt);	
		this->vf = (float) this->numCNT*vcnt/vol;
//#ifdef _DEBUG
		printf("_DEBUG:______________________________________________________________________________\n");
		printf("Box size: %f x %f x %f  volume: %e \n",this->physDim[0],this->physDim[1],this->physDim[2],vol);
        printf("Inclusion radius: %f length: %f volume: %e \n",this->a,this->l,vcnt);
		printf("Inclusions to be created: %i - volume fraction of %f will be achived \n",this->numCNT,this->vf);
		printf("_DEBUG:______________________________________________________________________________\n");
//#endif
	return 1;
}
int		simPrms::get_numCNT(void) {
	// calculates the number of CNTs currently active on device
	int size = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];
	vector<short> h_Occ(size,0);
	cudaMemcpy((void*) h_Occ.data(),(void*) this->Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);
	this->numCNT = std::accumulate(h_Occ.begin(),h_Occ.end(),0);
	//printf("Found %i in total... \n",this->numCNT);
	vector<short>().swap(h_Occ);
	return this->numCNT;
}
int		simPrms::save_dmnOcc(const char* fname) {
	ofstream ofile;
	ofile.open(fname, ios::out);
	int size = this->Domains.ext[0]*this->Domains.ext[1]*this->Domains.ext[2];
	vector<short> h_Occ(size,0);
	cudaMemcpy(h_Occ.data(),this->Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);
	for(int i = 0; i<h_Occ.size(); i++) {
		ofile << i << " " << h_Occ[i] << endl;
	}
	ofile.close();
	return accumulate(h_Occ.begin(),h_Occ.end(),0);
}

float	simPrms::get_VolFrac(int numCNT) {
		float Vinc = Pi*this->a*this->a*this->l;
		float Vcub = this->physDim[0]*this->physDim[1]*this->physDim[2];
		return numCNT*Vinc/Vcub*100;
}

int		simPrms::set_k2dmnIdx(void)	{
	return cuda_SetK2dmnIdx(this);
}



// Auxilary:============================================================================================
float	*split_den(int	size[3],float *meshOld,float *meshIn) {
	int	vol = size[0]*size[1]*size[2];
	float	*meshOut = (float*) malloc(vol*sizeof(float));
	for(int i=0;i<vol;i++)	meshOut[i] = meshOld[i] - meshIn[i];
	free( meshIn );
	return meshOut;
}
int	dmnIdx_old2new(int idx, int oldGridExt[3],int newGridExt[3]) {

	int x1 = (idx%oldGridExt[0])				+ (newGridExt[0]-oldGridExt[0])/2;
	int y1 = (idx/oldGridExt[0])%oldGridExt[1]	+ (newGridExt[1]-oldGridExt[1])/2;
	int z1 = (idx/oldGridExt[0])/oldGridExt[1]	+ (newGridExt[2]-oldGridExt[2])/2;

	return x1+y1*newGridExt[0]+z1*newGridExt[0]*newGridExt[1];
}

int dmnIdx_new2old(int idx, int oldGridExt[3],int newGridExt[3]) {

	int x0 = (idx%newGridExt[0])				- (newGridExt[0]-oldGridExt[0])/2;
	int y0 = (idx/newGridExt[0])%newGridExt[1]	- (newGridExt[1]-oldGridExt[1])/2;
	int z0 = (idx/newGridExt[0])/newGridExt[1]	- (newGridExt[2]-oldGridExt[2])/2;

	x0 = x0<0 ? x0+oldGridExt[0] : x0;	x0 = x0>=oldGridExt[0] ? x0-oldGridExt[0] : x0;
	y0 = y0<0 ? y0+oldGridExt[1] : y0;	y0 = y0>=oldGridExt[1] ? y0-oldGridExt[1] : y0;
	z0 = z0<0 ? z0+oldGridExt[2] : z0;	z0 = z0>=oldGridExt[2] ? z0-oldGridExt[2] : z0;

	return x0+y0*oldGridExt[0]+z0*oldGridExt[0]*oldGridExt[1];
}

vector<float>	offset_new2old(int idx, int oldGridExt[3],int newGridExt[3], float physDim[3]) {
	int x0 = (idx%newGridExt[0])				- (newGridExt[0]-oldGridExt[0])/2;
	int y0 = (idx/newGridExt[0])%newGridExt[1]	- (newGridExt[1]-oldGridExt[1])/2;
	int z0 = (idx/newGridExt[0])/newGridExt[1]	- (newGridExt[2]-oldGridExt[2])/2;

	vector<float> offset(3,0.0);

	offset[0] = (x0<0) ? offset[0]-physDim[0] : offset[0];	offset[0] = (x0>=oldGridExt[0]) ? offset[0]+physDim[0] : offset[0];
	offset[1] = (y0<0) ? offset[1]-physDim[1] : offset[1];	offset[1] = (y0>=oldGridExt[1]) ? offset[1]+physDim[1] : offset[1];
	offset[2] = (z0<0) ? offset[2]-physDim[2] : offset[2];	offset[2] = (z0>=oldGridExt[2]) ? offset[2]+physDim[2] : offset[2];

	return offset;
}
