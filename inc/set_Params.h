#pragma once

extern "C"
int		setDevConst(simPrms *Prms) {

	// write default values and check errors:
		cudaError_t cuErr;
		cuErr = cudaMemcpyToSymbol(incOrt_threshold,&Prms->def_OrtThreshold,sizeof(float),0,cudaMemcpyHostToDevice);
		cuErr = cudaMemcpyToSymbol(incOrt_minNorm,&Prms->def_minOrtNorm,sizeof(float),0,cudaMemcpyHostToDevice);

		cuErr = cudaMemcpyToSymbol(Phi_avg,&Prms->def_phiMed,sizeof(float),0,cudaMemcpyHostToDevice);
		cuErr = cudaMemcpyToSymbol(Phi_dev,&Prms->def_phiDev,sizeof(float),0,cudaMemcpyHostToDevice);
		cuErr = cudaMemcpyToSymbol(Theta_avg,&Prms->def_thetaMed,sizeof(float),0,cudaMemcpyHostToDevice);
		cuErr = cudaMemcpyToSymbol(Theta_dev,&Prms->def_thetaDev,sizeof(float),0,cudaMemcpyHostToDevice);
		cuErr = cudaMemcpyToSymbol(prefDir,&Prms->def_prefOrt,2*sizeof(float),0,cudaMemcpyHostToDevice);


		//cuErr = cudaMemcpyToSymbol(Pi,&Pi,sizeof(float),0,cudaMemcpyHostToDevice);							// set Pi
		cuErr = cudaMemcpyToSymbol(DEF_L,&(Prms->l),sizeof(float),0,cudaMemcpyHostToDevice);					// default CNT length
		cuErr = cudaMemcpyToSymbol(DEF_A,&(Prms->a),sizeof(float),0,cudaMemcpyHostToDevice);					// default CNT radius
		cuErr = cudaMemcpyToSymbol(epsilon,&Prms->TOL,sizeof(float),0,cudaMemcpyHostToDevice);					// default precision treshold
		cuErr = cudaMemcpyToSymbol(sc_frac,&Prms->sc_frac,sizeof(float),0,cudaMemcpyHostToDevice);				// default precision treshold
		cuErr = cudaMemcpyToSymbol(separation,&Prms->sep,sizeof(float),0,cudaMemcpyHostToDevice);				// default precision treshold
		cuErr = cudaMemcpyToSymbol(dmnExt,(void *) Prms->Domains.ext,3*sizeof(int),0,cudaMemcpyHostToDevice);	// extents of domain grid
		cuErr = cudaMemcpyToSymbol(neiOrder,(void *) &(Prms->NeiOrder),sizeof(int),0,cudaMemcpyHostToDevice); // intersection check up to neiOrder neighbouring cells
		cuErr = cudaMemcpyToSymbol(numCNT,&(Prms->Domains.ttlCNT),sizeof(int),0,cudaMemcpyHostToDevice);		// number of CNT to create
		cuErr = cudaMemcpyToSymbol(phsScl,(void *) Prms->Domains.edge,3*sizeof(float),0,cudaMemcpyHostToDevice);		// physical rescale factor
#ifdef _DEBUG	
		printf("_DEBUG: Set constants on device: ");
		printf( cudaGetErrorString(cuErr) ); 
		printf( "\n" );
#endif
		if (cuErr == cudaSuccess) {
			return 1;
		} else {
			return 0;
		}
} 
