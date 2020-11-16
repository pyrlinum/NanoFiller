//====================================================================================================================
//										 <<< DUMP current state to disk or READ from file: >>>
//====================================================================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include "simPrms.h"
#include "IO.h"

void save_state_TXT(const char* fname, simPrms Simul) {
	ofstream ofile;
	ofile.open(fname, ios::out);

	for(int i=0;i<3;i++)	ofile << Simul.physDim[i] << endl;
	for(int i=0;i<3;i++)	ofile << Simul.Domains.edge[i] << endl;
	for(int i=0;i<3;i++)	ofile << Simul.Domains.ext[i] << endl;

	ofile << Simul.l << endl;
	ofile << Simul.a << endl;

	ofile << Simul.Domains.ttlCNT << endl;
	ofile << Simul.numCNT << endl;
	ofile << Simul.TOL << endl;

	// dump CUDA memory:
	int size = Simul.Domains.ext[0]*Simul.Domains.ext[1]*Simul.Domains.ext[2];

	short	*h_dmnOcc = (short *) malloc(size*sizeof(short));	
	cudaMemcpy(h_dmnOcc,Simul.Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);
	for(int i=0;i<size;i++) ofile << setw(8) << h_dmnOcc[i] << '\t';
	free(h_dmnOcc);
	ofile << endl;

	// set kernel size and split - for large grids:

	int	*h_dmnAddr = (int *) malloc(size*sizeof(int));	
	cudaMemcpy(h_dmnAddr,Simul.Domains.d_dmnAddr,size*sizeof(int),cudaMemcpyDeviceToHost);
	for(int i=0;i<size;i++) ofile << setw(8) << h_dmnAddr[i] << '\t';
	free(h_dmnAddr);
	ofile << endl;
	
	float	*h_partData = (float *) malloc(Simul.Domains.ttlCNT*sizeof(float));
	for(int j=0;j<9;j++) {
		cudaMemcpy(h_partData,Simul.d_result+j*Simul.Domains.ttlCNT,Simul.Domains.ttlCNT*sizeof(int),cudaMemcpyDeviceToHost);
		for(int i=0;i<Simul.Domains.ttlCNT;i++) ofile << setw(8) << h_partData[i] << '\t';
		ofile << endl;
	}
	
	ofile.close();

	cout << "Simulation state dumped into file " << fname << ' ' << cudaGetErrorString(cudaGetLastError()) << endl;
	
}

void save_state_BIN(const char* fname, simPrms Simul) {
	ofstream ofile;
	ofile.open(fname, ios::out | ios::binary);

	ofile.write((char*) Simul.physDim,3*sizeof(float)); 
	ofile.write((char*) &Simul.Domains.edge,3*sizeof(float));
	ofile.write((char*) Simul.Domains.ext,3*sizeof(int));
	ofile.write((char*) &Simul.l,sizeof(float));
	ofile.write((char*) &Simul.a,sizeof(float));
	ofile.write((char*) &Simul.Domains.ttlCNT,sizeof(int));	
	ofile.write((char*) &Simul.numCNT,sizeof(int));
	ofile.write((char*) &Simul.TOL,sizeof(float));
	

	int size = Simul.Domains.ext[0]*Simul.Domains.ext[1]*Simul.Domains.ext[2];

	short	*h_dmnOcc = (short *) malloc(size*sizeof(short));	
	cudaMemcpy(h_dmnOcc,Simul.Domains.d_dmnOcc,size*sizeof(short),cudaMemcpyDeviceToHost);
	ofile.write((char*) h_dmnOcc,size*sizeof(short));
	free(h_dmnOcc);

	int	*h_dmnAddr = (int *) malloc(size*sizeof(int));	
	cudaMemcpy(h_dmnAddr,Simul.Domains.d_dmnAddr,size*sizeof(int),cudaMemcpyDeviceToHost);
	ofile.write((char*)h_dmnAddr,size*sizeof(int));
	free(h_dmnAddr);

	float	*h_partData = (float *) malloc(Simul.Domains.ttlCNT*sizeof(float));
	for(int j=0;j<9;j++) {
		cudaMemcpy(h_partData,Simul.d_result+j*(Simul.Domains.ttlCNT),Simul.Domains.ttlCNT*sizeof(float),cudaMemcpyDeviceToHost);
		ofile.write((char*) h_partData,Simul.Domains.ttlCNT*sizeof(float));
	}
	ofile.close();

	cout << "Simulation state dumped into file " << fname << ' ' << cudaGetErrorString(cudaGetLastError()) << endl;
	
}
void read_state_BIN(const char* fname, simPrms *Simul) {
	ifstream ifile;
	ifile.open(fname, ios::in | ios::binary);

	if(ifile.is_open()) {

		ifile.read((char*) Simul->physDim,3*sizeof(float)); 
		ifile.read((char*) &Simul->Domains.edge,3*sizeof(float));
		ifile.read((char*) Simul->Domains.ext,3*sizeof(int));
		ifile.read((char*) &Simul->l,sizeof(float));
		ifile.read((char*) &Simul->a,sizeof(float));
		ifile.read((char*) &Simul->Domains.ttlCNT,sizeof(int));
		ifile.read((char*) &Simul->numCNT,sizeof(int));
		ifile.read((char*) &Simul->TOL,sizeof(float));

		Simul->Domains.grdSize = Simul->Domains.ext[0]*Simul->Domains.ext[1]*Simul->Domains.ext[2];

		short	*h_dmnOcc = (short *) malloc(Simul->Domains.grdSize*sizeof(short));
		ifile.read((char*) h_dmnOcc,Simul->Domains.grdSize*sizeof(short));
		cudaMalloc(&(Simul->Domains.d_dmnOcc),Simul->Domains.grdSize*sizeof(short));
		cudaMemcpy(Simul->Domains.d_dmnOcc,h_dmnOcc,Simul->Domains.grdSize*sizeof(short),cudaMemcpyHostToDevice);
		free(h_dmnOcc);
	
		int	*h_dmnAddr = (int *) malloc(Simul->Domains.grdSize*sizeof(int));
		ifile.read((char*)h_dmnAddr,Simul->Domains.grdSize*sizeof(int));
		cudaMalloc(&(Simul->Domains.d_dmnAddr),Simul->Domains.grdSize*sizeof(int));
		cudaMemcpy(Simul->Domains.d_dmnAddr,h_dmnAddr,Simul->Domains.grdSize*sizeof(int),cudaMemcpyHostToDevice);
		free(h_dmnAddr);

		float	*h_partData = (float *) malloc(Simul->Domains.ttlCNT*sizeof(float));
		cudaMalloc(&(Simul->d_result),9*Simul->Domains.ttlCNT*sizeof(float));
		for(int j=0;j<9;j++) {
			ifile.read((char*) h_partData,Simul->Domains.ttlCNT*sizeof(float));
			cudaMemcpy(Simul->d_result+j*(Simul->Domains.ttlCNT),h_partData,Simul->Domains.ttlCNT*sizeof(float),cudaMemcpyHostToDevice);
		}
	
		ifile.close();

		Simul->device_config();

		cout << "Simulation state read from file " << fname << ' ' << cudaGetErrorString(cudaGetLastError()) << endl;
		cout << Simul->numCNT << " inclusions found " << endl;
	} else printf("File %s does not exists \n",fname);
	
}
