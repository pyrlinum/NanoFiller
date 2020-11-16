#pragma once
#include <string>
#include "CNT.h"
#include "definitions.h"

//#define BLOCK	1024
#define H_Cstyle_Mesh
#define Pi 3.14159265359f

const int	MaxBlocksPerMP = 1;
const int	MEM_RESERVE = 10*1<<20;	// Reserved memory
const int	MegaByte = 1<<20;		// 1 Mb in bytes

// Lenear Congruenal Generator:
const long long LCG_mod = 4294967296;
const long long LCG_mul = 1664525;
const long long LCG_inc = 1013904223;


typedef struct {
	int		ext[3];		// domain grid extents
	int		grdSize;	// number of cells in grid
	float	edge[3];	// domain edge length
	int		segNum;		// number of load-levelled segments
	int		ttlCNT;		// total number of cnt-vacances in memory
	int		*numCNT;	// the number of CNTs to be created in domains
	int		*mask[6];	// stores masks for load-leveled block tasks
	int		*d_masks;	// stores masks on device;
	int		*d_dmnAddr;	// device array to store adresses of spatial cells
	short	*d_dmnOcc;	// device array to store current domain occupancies
	short	*d_dmnCrt;	// device array to store newly generated CNTs
	int		*d_dmnLck;	// device array to store domain cells' locks
} DmnGrdDscr_t;

typedef struct {
	bool	RED_FLAG;	// reduce angle deviation to achieve higher volume fractions
	bool	DEL_FLAG;	// delete existing inclusions to achieve higher volume fractions
	int		RED_NSTP;	// number of steps to check the need to reduce angle deviation
	int		DEL_NSTP;	// number of steps to check the need to delete existing inclusions
	float	RED_CRIT;	// criteria to reduce orientation
	float	DEL_CRIT;	// criteria to reduce orientation
} simCtrlDscr_t;

typedef struct {
	float	*x;	 // CNT center coordinates
	float	*y;
	float	*z;
	float	*cx; // CNT directional cosines
	float	*cy;
	float	*cz;
	float	*l;	 // CNT length
	float	*a;	 // CNT radius
	float	*k;
} CNT_s;


class simPrms
{
public:
// Input data: 
	float		vf;			// desired volume fraction
	int			vfcount;	// number of desired volume fractions to generate
	float*		vfs;		// array with the list of desired volume fractions
	float		density;	// particle density in gramm/mole/Angstrom^2

	//Inclusion distribution parameters:
	std::string	crdDsrtFile;	// name of the file with probability density for inclusions coordinates distribution - if empty uniform distribution is generated
	int		ext[3];			// mesh extents
	float	physDim[3];		// sample physical dimensions

	const char	*ortDsrtFile;	// name of the file with spatialy dependent prefered orientations and deviations for inclusions orientation distribution
	float	def_prefOrt[2];	// default prefered orientation if file is not used
	float	def_thetaMed;	// avarage theta angle in degrees (counted from prefered direction) - [0;180)
	float	def_thetaDev;	// standard deviation of theta
	float	def_phiMed;		// avarage phi angle (in plane perpendicular to prefered direction) - [0;180) 
	float	def_phiDev;		// standart deviation of phi
	float	def_OrtThreshold; // threshold to induce orientation;
	float	def_minOrtNorm;	// minimum norm of orientation moment matrix eigenvector to induce orientation

	float*	h_prefOrt;		// default prefered orientation if file is not used
	float*	h_thetaMed;		// avarage theta angle in degrees (counted from prefered direction) - [0;180)
	float*	h_thetaDev;		// standard deviation of theta
	float*	h_phiMed;		// avarage phi angle (in plane perpendicular to prefered direction) - [0;180) 
	float*	h_phiDev;		// standart deviation of phi

	float*	d_prefOrt;		// default prefered orientation if file is not used
	float*	d_thetaMed;		// avarage theta angle in degrees (counted from prefered direction) - [0;180)
	float*	d_thetaDev;		// standard deviation of theta
	float*	d_phiMed;		// avarage phi angle (in plane perpendicular to prefered direction) - [0;180) 
	float*	d_phiDev;		// standart deviation of phi

	const char	*dimDstrFile;	// name of the file with spatial inclusions' dimentions distribution - uniform default
	float	l;				// default "average" length
	float	dev_l;			// default std deviation of length
	float	a;				// default "avarage" radius
	float	dev_a;			// default "avarage" radius

	// precision parameters:
	float	TOL;			// precision of math operations
	float	PRECISION;		// precision of volume fraction
	float	MARGE;			// margin for extra cnt per domain
	float	sep;			// minimum separation for inclusions
	float	sc_frac;			// minimum separation for inclusions

	// Clusterization parameters:
	bool	clusterize_flag;// iteration after which to start clusterization
	float	clusterize_angl;// angle criteria for clusterization
	float	clusterize_dist;// distance criteria for clusterization


	// statistics collection options:
	float	stat_dist_lim;	// mutual distance limit

	// Flags:
	bool	wrt_bin_flag;		// if the state after generation should be written to output.bin
	int		self_alling_step;	// if the angle deviation should be decreased iteratively
	bool	reduce_flag;		// if the previously created inclusions should be deleted


	
// Internal variables:
	float	*mesh;		// Ptr to 3D array of mesh values - probability density distribution mesh structure
	int		numCNT;		// number of CNT to create
	int		maxVol;		// maximum nubmer of cells in segment
	int		NeiOrder;	// maximum order of neighbours to compute interaction with

	int		cuRNG_Num;	// number of CUDA RNG running in parallel
	int		*h_RNGseed;	// cuRND initial seeds array
	int		*d_RNGseed;	// cuRND initial seeds array
	int		kernelSplit;// number of intersection kernell Launches to cover the grid
	int		kernelSize;// number of intersection kernell Launches to cover the grid

	DmnGrdDscr_t Domains;		// domain grid descriptor - holds domain grid extents, domain egde and domains CNT quantities
	
	CNT_s		d_res;			// Struct to store result of generation on Device
	CNT_s		h_res;			// Struct to store result of generation on Host
	float		*d_result;		// the same as 1 structure
	float		*h_result;		// the same as 1 structure

	dim3		Grid;	// structure of blocks in grid
	dim3		Block;// structure of threads in block

	cudaEvent_t	pre_start, pre_stop;
	cudaEvent_t	gen_start, gen_stop;
	cudaEvent_t	isc_start, isc_stop;
	

	simPrms();

	CNT_t		*asmbl_cnt_data(void);
	int			allocateDevArr(void);
	int			device_config(void);
	int			distributeDmn(int maxLoad);
	int			reDistributeDmn(int iter, int maxLoad);
	int			make_den(const char* name);
	int			make_default_den(void);
	int			make_dmnGrid(int maxLoad,float margin);
	int			maxLeaf(void);
	int			repack(void);
	int		 	expand(int dir);
	int			seed_cuRND(int seed);
	int			set_Masks(void);
	int			set_dmnAddr(void);
	int			set_dmnOrt(void);
	int			set_Result(void);
	int			set_numCNT(float length, float rad, float volumefraction);
	int			set_ParamsFromInput(void);
	int			set_ParamsFromFile(const char *filename);
	int			get_numCNT(void);
	int			save_dmnOcc(const char *fname);
	float		get_VolFrac(int numCNT);
	int			set_k2dmnIdx(void);
	
	~simPrms(void);

private:
	int purifyCNT(void);
};

float*		split_den(int size[3],float* meshOld, float* meshIn);
int			cuda_SetK2dmnIdx(	simPrms*	Simul );

