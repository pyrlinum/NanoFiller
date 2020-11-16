// write porbability density to texture:
#pragma once

extern "C"
void density2Tex( float *h_density, int ext[3] ) {
	// refactor array - NEEDED ONLY TO SEE INPUT ARRAY BOTH IN C AND CUDA CORRECTLY - CUDA WILL ACCEPT FORTRAN STYLE
	float *h_density_T = (float *) malloc(ext[0]*ext[1]*ext[2]*sizeof(float));
	for (int i=0;i<ext[0];i++) {
		for (int j=0;j<ext[1];j++) {
			for (int k=0;k<ext[2];k++) {
				*(h_density_T+i+j*ext[0]+k*ext[0]*ext[1])=*(h_density+i*ext[2]*ext[1]+j*ext[2]+k);
			}}} 
	h_density = h_density_T; 

	// create channel description
	cudaChannelFormatDesc	den_chnlDsc = cudaCreateChannelDesc<float>();
	const cudaExtent	den_Ext		= make_cudaExtent(ext[0],ext[1],ext[2]);
	cudaMalloc3DArray(&den_arr, &den_chnlDsc, den_Ext);

	// copy density mesh to cudaArray
	cudaMemcpy3DParms		den_cpyPrms = {0};
	den_cpyPrms.extent	 = den_Ext; 
	den_cpyPrms.srcPtr	 = make_cudaPitchedPtr((void *) h_density, den_cpyPrms.extent.width*sizeof(float), den_cpyPrms.extent.width, den_cpyPrms.extent.height);
	den_cpyPrms.dstArray = den_arr;	

	den_cpyPrms.kind	 = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&den_cpyPrms);
#ifdef _DEBUG
	printf("_DEBUG: Probability density copyed to GPU memory! ");
	cudaError_t cuErr = cudaGetLastError();
	printf( cudaGetErrorString(cuErr) );
	printf( "\n" );
#endif

	// set texture parameters
	denTex.addressMode[0] = cudaAddressModeWrap;
	denTex.addressMode[1] = cudaAddressModeWrap;
	denTex.addressMode[2] = cudaAddressModeWrap;
	denTex.filterMode = cudaFilterModeLinear;
	denTex.normalized = true;

	cudaBindTextureToArray( &denTex, den_arr, &den_chnlDsc );

	// Set Device constants to keep texture dimensions:
	cudaMemcpyToSymbol(texDim,ext,3*sizeof(int),0,cudaMemcpyHostToDevice);
#ifdef _DEBUG
	printf("_DEBUG: Extent values: %i %i %i \n",den_cpyPrms.extent.width,den_cpyPrms.extent.height,den_cpyPrms.extent.depth );
	printf("_DEBUG: Set texture dimentions in GPU constant memory: ");
	cudaError_t cuErrTD = cudaGetLastError();
	printf( cudaGetErrorString(cuErrTD) );
	printf( "\n" );
#endif
}
