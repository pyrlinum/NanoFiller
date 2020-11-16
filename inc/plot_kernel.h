// Projection droiwng kernel: 
__global__ void cuda_plotXY(int3	displ,
							float	Z0,
							float	Heith,
							int		*d_dmnAddr,
							short	*d_dmnOcc,
							float	*d_result,
							float   *imgPxls);
// gaussian blur kernel:
__global__ void gaussBlur(	int		gaussKnlSize,
							float	*d_pixelsOld,
							float	*d_pixelsNew,
							char	dir);

// Auxilary functions:
inline __device__ int extendedPatchPos(int sidebar);
inline __device__ int extendedPatchRead(float *shrPatch,int sidebar,float *glbImg);
inline __device__ int loadCNT2shr(float *d_result, float *shr_arr, int st_addr, int load, int glb_patch);	// load cnt from current cell into shared memory
inline __device__ int readImg2Shr(float *d_imgArr,float *shr_imgArr);										// read pixel patch from device memory
inline __device__ int writeShr2Img(float *d_imgArr,float *shr_imgArr);										// write pixel patch to device memory
inline __device__ int set_dmnPatch(int dmnPtch[3],float z0, float h, float resScl, float dmnScl);			// compute domain grid patch basepoint
inline __device__ int set_keyInPeriodic(int basePnt[3], int3 displ, int dmnGrdExt[3]);					// compute current grid cell key
// MAIN AUXILARY: cnt impact in current cell color:
inline __device__ float impact(int i,float *shr_arr, float scale);											// the simpliest impact function - 1/0
inline __device__ bool lineQuadroIsec2D(float2 a0, float2 cos, float2 quad[2]);								// returns 1 if line and quadro intersects
inline __device__ float PointQuadDist(float2 a0, float2 cx, float2 quad[2]);


