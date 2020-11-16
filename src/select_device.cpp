#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Beginning of GPU Architecture definitions
inline int ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

int select_device() {
// Detecting CUDA-enabled devices and selecting the most powerful: 
	int currentDev = -1;
	cudaDeviceProp devProp;

// device selection method:
#ifdef _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_
	currentDev = cutGetMaxGflopsDeviceId();
#else
	int devCount = 0;
	int SMperMP;
	int dev_Mem;
	float dev_CC;
	float dev_SP_GFlops;
	float dev_DP_GFlops;
	float max_GF=-1.0;

	printf("______________________________________________________________________________\n");
	cudaGetDeviceCount( &devCount );
	printf("%d devices were found: \n",devCount);

	for (int i=0; i<=devCount-1; i++) {

		cudaGetDeviceProperties(&devProp,i);
		SMperMP = ConvertSMVer2Cores(devProp.major,devProp.minor);
		dev_CC = devProp.major+0.1f*devProp.minor;
		dev_SP_GFlops = 2.0f*devProp.multiProcessorCount*SMperMP*devProp.clockRate/pow(10.0f,6);
		dev_DP_GFlops = ( dev_CC >= 1.3f )?dev_SP_GFlops/8.0f:0.0f;
		dev_Mem = devProp.totalGlobalMem/(1<<20);
		
		printf(" %d: %s - Compute Capability %3.1f \n\tpeak performance: %6.1f SP GFlops %6.1f DP Gflops \n\tTotal Memory: %i Mb\n"
				,i,devProp.name,dev_CC,dev_SP_GFlops,dev_DP_GFlops,dev_Mem);
		if (dev_SP_GFlops > max_GF) {
			max_GF = dev_SP_GFlops;
			currentDev = i;
		}
		
		
	}
#endif

    if ( currentDev == -1 && devCount>0 ) { currentDev=0; }

	if ( currentDev != -1 ) {
		cudaGetDeviceProperties(&devProp,currentDev);
		printf("Using device: %s \n",devProp.name);
		cudaSetDevice( currentDev );
	}
	printf("______________________________________________________________________________\n");
	return currentDev;
}


