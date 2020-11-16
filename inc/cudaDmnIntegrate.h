#pragma once
extern "C"
__global__ void cudaDmnIntegrate(float *d_result) { 

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < dmnExt[0]*dmnExt[1]*dmnExt[2]) {
	
			// first point of domain N tid:
			int tX = tid % dmnExt[0];
			int tY = (tid/dmnExt[0])%dmnExt[1];
			int tZ = (tid/dmnExt[0])/dmnExt[1];

			float	sum = 0.0f;
		
			// simple integration:
			for(int i=0;i<numIntPoints; i++) {
				for(int j=0;j<numIntPoints; j++) {
					for(int k=0;k<numIntPoints; k++) {

						float x = (tX + ((float)i)/numIntPoints )/dmnExt[0] + 0.5/texDim.x;
						float y = (tY + ((float)j)/numIntPoints )/dmnExt[1] + 0.5/texDim.y;
						float z = (tZ + ((float)k)/numIntPoints )/dmnExt[2] + 0.5/texDim.z;

						sum += tex3D(denTex,x,y,z);
			}}}

			d_result[tid] = sum/numIntPoints/numIntPoints/numIntPoints;
	}

	__syncthreads();

}
