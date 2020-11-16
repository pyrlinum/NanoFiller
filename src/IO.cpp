// This file describes functions used to set up generation
#define DEFAULT_DIR_H
#include "IO.h"


// IO directory name:
extern const char	outDir[255] = "./";
int read_inputfile() {
	return 1;
}
float *read_density(const char *filename, int *ext, float *dim) {
	int		size;

	ifstream parF(filename,ios::in);
	parF >> *dim >> *(dim+1) >> *(dim+2); 
	parF >> *ext >> *(ext+1) >> *(ext+2);
	size = *ext**(ext+1)**(ext+2);
#ifdef _DEBUG
	printf("_DEBUG: Reading file %s \n",filename);
	printf("_DEBUG: Dimensions: %f %f %f \n",*dim,*(dim+1),*(dim+2));
	printf("_DEBUG: Extents: %i %i %i \n",*ext,*(ext+1),*(ext+2));
#endif
	float *data = (float *) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) parF >> data[i];
	/*
	int x,y,z;
	for(int i=0;i<size;i++) {
		x = i%ext[0];
		y = (i/ext[0])%ext[1];
		z = i/(ext[1]*ext[0]);
		data[i] = ((ext[0]/3<x)&&(x<2*ext[0]/3)||(ext[1]/3<y)&&(y<2*ext[1]/3)||(ext[2]/3<z)&&(z<2*ext[2]/3) );
	} //*/
	parF.close();

    return data;
}

void write_CNTvtk(const char *filename, int cntNum, CNT_t *cnt_arr) {
	char	file1[255] = "", file2[255] = "";
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".vtk");

	ofstream vtkF(file1,ios::out | ios::trunc);
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA generated CNTs" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET UNSTRUCTURED_GRID" << endl;
	vtkF << "POINTS " << cntNum << " float" << endl;

	for (int i=0;i<cntNum;i++) {
		vtkF << setw(12) << setprecision(6) << cnt_arr[i].r.x
			 << setw(12) << setprecision(6) << cnt_arr[i].r.y
			 << setw(12) << setprecision(6) << cnt_arr[i].r.z
			 << endl;
	}
	vtkF << "POINT_DATA " << cntNum << endl;
	vtkF << "VECTORS CNT float" << endl; 
	for (int i=0;i<cntNum;i++) {
		vtkF << setw(15) << setprecision(6) << cnt_arr[i].c.x
			 << setw(15) << setprecision(6) << cnt_arr[i].c.y
			 << setw(15) << setprecision(6) << cnt_arr[i].c.z << endl;

	} 
	// diagnostics only:
	vtkF << "SCALARS LAK float 3" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int i=0;i<cntNum;i++) {
		vtkF << setw(12) << setprecision(6) << cnt_arr[i].l
			 << setw(12) << setprecision(6) << cnt_arr[i].a //<< endl;
			 << setw(12) << setprecision(6) << cnt_arr[i].k << endl;
	}
	vtkF.close();

	printf("%s.vtk created! \n",filename);
}
void write_CNT_ORTvtk(const char *filename, int cntNum, CNT_t *cnt_arr) {
	char	file1[255] = "", file2[255] = "";
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".vtk");

	ofstream vtkF(file1,ios::out | ios::trunc);
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA generated CNTs" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET UNSTRUCTURED_GRID" << endl;
	vtkF << "POINTS " << cntNum << " float" << endl;
	for (int i=0;i<cntNum;i++) {
		vtkF << setw(15) << setprecision(6) << cnt_arr[i].c.x
			 << setw(15) << setprecision(6) << cnt_arr[i].c.y
			 << setw(15) << setprecision(6) << cnt_arr[i].c.z << endl;

	} 
	vtkF.close();

	printf("%s.vtk created! \n",filename);
}

void write_CNTlmp(const char *filename, int cntNum, CNT_t *cnt_arr, float *box, float density) {
	char	file1[255] = "", file2[255] = "";
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".lmp.data");

	ofstream lmpF(file1,ios::out | ios::trunc);
	lmpF << "CUDA generated aspherical particles" << endl;
	lmpF << endl;
	lmpF << cntNum << " atoms" << endl;
	lmpF << 1 << " atom types" << endl;
	lmpF << cntNum << " ellipsoids" << endl;
	lmpF << endl;
	lmpF << 0.0f << ' ' << box[0] <<" xlo xhi" << endl;
	lmpF << 0.0f << ' ' << box[1] <<" ylo yhi" << endl;
	lmpF << 0.0f << ' ' << box[2] <<" zlo zhi" << endl;
	lmpF << endl;
	lmpF << "Atoms" << endl;
	lmpF << endl;
	for (int i=0;i<cntNum;i++) {
		lmpF	<< i+1 << ' ' << 1 << ' ' << 1 << ' ' << density
				<< setw(12) << setprecision(6) << cnt_arr[i].r.x
				<< setw(12) << setprecision(6) << cnt_arr[i].r.y
				<< setw(12) << setprecision(6) << cnt_arr[i].r.z << endl;
	}
	lmpF << endl;
	lmpF << "Ellipsoids" << endl;
	lmpF << endl;
	// v = [(1,0,0)x(c.x,c.y,c.z)] = (0,-c.z,c.y)
	// cos(theta/2) = sqrt((1+c.x)/2)
	// sin(theta/2) = sqrt((1-c.x)/2)
	for (int i=0;i<cntNum;i++) {
		lmpF	<< i+1
				<< setw(12) << setprecision(6) << cnt_arr[i].l+2*cnt_arr[i].a
				<< setw(12) << setprecision(6) << 2*cnt_arr[i].a
				<< setw(12) << setprecision(6) << 2*cnt_arr[i].a
				<< setw(15) << setprecision(6) << sqrtf((1+cnt_arr[i].c.x)/2)					// quatw = cos(theta/2)
				<< setw(15) << setprecision(6) << 0.0f											// quatx = v.x*sin(theta/2)
				<< setw(15) << setprecision(6) << -cnt_arr[i].c.z*sqrtf((1-cnt_arr[i].c.x)/2)	// quaty = v.y*sin(theta/2)
				<< setw(15) << setprecision(6) << +cnt_arr[i].c.y*sqrtf((1-cnt_arr[i].c.x)/2)	// quatz = v.z*sin(theta/2)
				<< endl;
	} 
	lmpF.close();

	printf("%s.lmp.data created! \n",filename);
}

void write_CNTdat(const char* filename, int size, float *cnt_arr) {
	
	char	file1[255] = "", file2[255] = "";
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".dat");

	ofstream vtkF(file1,ios::out | ios::trunc);
	/*
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA generated CNTs" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET UNSTRUCTURED_GRID" << endl;
	vtkF << "POINTS " << size*1024 << " float" << endl;
	*/
	int sum1 = 0;
	int sum2 = 0;
	//int l = 0;
	for (int l=0;l<size;l++)
	for (int i=0;i<1024;i++) 
		//if (cnt_arr[i*9+6+l*18*1024]>0) {
		if ((cnt_arr[(2*i+1)*9+7+l*18*1024]>0)&&(cnt_arr[(2*i+1)*9+8+l*18*1024] == 4)) {
		/*vtkF << setw(10) << setprecision(6) << cnt_arr[i*9+0+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+1+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+2+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+3+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+4+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+5+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+6+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+7+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[i*9+8+l*18*1024] << endl;*/
	 vtkF << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+0+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+1+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+2+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+3+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+4+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+5+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+6+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+7+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+0)*9+8+l*18*1024] << endl; 
		vtkF << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+0+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+1+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+2+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+3+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+4+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+5+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+6+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+7+l*18*1024]
			 << setw(10) << setprecision(6) << cnt_arr[(2*i+1)*9+8+l*18*1024] << endl; 
		 if (cnt_arr[i*9+7+l*18*1024]<=0) sum1--;
		 if (cnt_arr[i*9+7+l*18*1024]>0) sum2++;
	}
	sum1+=size*1024;
	
	vtkF.close();

	printf("%s.dat created! %i-%i CNTs found \n",filename,sum1,sum2);
}

void write_Meshvtk(int dim[3], float *uni_mesh) {
	char outFile[255] = "CNTmesh.vtk";
	strcat(outFile,outDir);
	strcat(outFile,outFile);
	ofstream vtkF(outFile,ios::out | ios::trunc);
	int size = dim[0]*dim[1]*dim[2];
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA sampled probability map" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET STRUCTURED_POINTS" << endl;
	vtkF << "DIMENSIONS "	<< setw(10) << setprecision(7) << dim[0]
							<< setw(10) << setprecision(7) << dim[1]
							<< setw(10) << setprecision(7) << dim[2] << endl;
	vtkF << "ORIGIN 0 0 0" << endl;
	vtkF << "SPACING "	<< setw(10) << setprecision(7) << 1./dim[0]
						<< setw(10) << setprecision(7) << 1./dim[0]
						<< setw(10) << setprecision(7) << 1./dim[0] << endl;
	vtkF << "POINT_DATA " << size << endl;
	vtkF << "SCALARS Density float 1" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << uni_mesh[i] << endl;
	} 
	/*for (int k=0;k<dim[2];k++) {
		for (int j=0;j<dim[1];j++) {
			for (int i=0;i<dim[0];i++) {
				vtkF << setw(10) << setprecision(7) << uni_mesh[i*dim[1]*dim[2]+j*dim[2]+k] << endl;
			}}} */
	vtkF.close();

	printf("CNTmesh.vtk created! \n");
}
void write_Meshvtk(const char* filename,int dim[3],float phsDm[3], float *uni_mesh) {
	char outFile[255] = "";
	strcat(outFile,outDir);
	strcat(outFile,filename);
	ofstream vtkF(outFile,ios::out | ios::trunc);
	int size = dim[0]*dim[1]*dim[2];
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA sampled probability map" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET STRUCTURED_POINTS" << endl;
	vtkF << "DIMENSIONS "	<< setw(10) << setprecision(7) << dim[0]
							<< setw(10) << setprecision(7) << dim[1]
							<< setw(10) << setprecision(7) << dim[2] << endl;
	vtkF << "ORIGIN 0 0 0" << endl;
	vtkF << "SPACING "	<< setw(10) << setprecision(7) << phsDm[0]/dim[0]
						<< setw(10) << setprecision(7) << phsDm[1]/dim[0]
						<< setw(10) << setprecision(7) << phsDm[2]/dim[0] << endl;
	vtkF << "POINT_DATA " << size << endl;
	vtkF << "SCALARS Density float 1" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << uni_mesh[i] << endl;
	} 
	/*for (int k=0;k<dim[2];k++) {
		for (int j=0;j<dim[1];j++) {
			for (int i=0;i<dim[0];i++) {
				vtkF << setw(10) << setprecision(7) << uni_mesh[i*dim[1]*dim[2]+j*dim[2]+k] << endl;
			}}} */
	vtkF.close();

	char comment[255] = "";
	strcat(comment,filename);
	strcat(comment," created! \n");
	printf(comment);
}
void write_Meshvtk(const char* filename,int dim[3], int *uni_mesh) {
	char outFile[255] = "";
	strcat(outFile,outDir);
	strcat(outFile,filename);
	ofstream vtkF(outFile,ios::out | ios::trunc);
	int size = dim[0]*dim[1]*dim[2];
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA sampled probability map" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET STRUCTURED_POINTS" << endl;
	vtkF << "DIMENSIONS "	<< setw(16) << setprecision(7) << dim[0]
							<< setw(16) << setprecision(7) << dim[1]
							<< setw(16) << setprecision(7) << dim[2] << endl;
	vtkF << "ORIGIN 0 0 0" << endl;
	vtkF << "SPACING "	<< setw(16) << setprecision(7) << 1./dim[0]
						<< setw(16) << setprecision(7) << 1./dim[0]
						<< setw(16) << setprecision(7) << 1./dim[0] << endl;
	vtkF << "POINT_DATA " << size << endl;
	vtkF << "SCALARS Density int 1" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(16) << uni_mesh[i] << endl;
	} 
	vtkF.close();

	char comment[255] = "";
	strcat(comment,filename);
	strcat(comment," created! \n");
	printf(comment);
}
void write_RNDMeshvtk( int size, float *rnd_mesh) {
	
	ofstream vtkF("D:/sergey/Documents/Visual Studio 2008/Projects/CUDA-F_Share/CUDA_RNDMesh.vtk",ios::out | ios::trunc);
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "CUDA randomly sampled mesh" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET UNSTRUCTURED_GRID" << endl;
	vtkF << "POINTS " << size << " float" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << *(rnd_mesh+i*sizeof(float)+0)
			 << setw(10) << setprecision(7) << *(rnd_mesh+i*sizeof(float)+1)
			 << setw(10) << setprecision(7) << *(rnd_mesh+i*sizeof(float)+2) << endl;
	}
	vtkF << "POINT_DATA " << size << endl;
	vtkF << "SCALARS mesh float 1" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << *(rnd_mesh+i*sizeof(float)+3) << endl;
	} 
	vtkF.close();

	printf("RNDMesh.vtk created! \n");
}

void write_vtkVectorField(const char* filename, int dim[3], float *vecfield){
	int size = dim[0]*dim[1]*dim[2];
	ofstream vtkF(filename,ios::out | ios::trunc);
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "collected orientations:" << endl;
	vtkF << "ASCII" << endl;
	vtkF << "DATASET STRUCTURED_POINTS" << endl;
	vtkF << "DIMENSIONS "	<< setw(16) << setprecision(7) << dim[0]
							<< setw(16) << setprecision(7) << dim[1]
							<< setw(16) << setprecision(7) << dim[2] << endl;
	vtkF << "ORIGIN 0 0 0" << endl;
	vtkF << "SPACING "	<< setw(16) << setprecision(7) << 1./dim[0]
						<< setw(16) << setprecision(7) << 1./dim[0]
						<< setw(16) << setprecision(7) << 1./dim[0] << endl;
	vtkF << "POINT_DATA " << size << endl;
	vtkF << "VECTORS Ort float" << endl; //*/
	
	for (int i=0;i<size;i++) {
		vtkF << setw(15) << setprecision(6) << vecfield[i+0*size]
			 << setw(15) << setprecision(6) << vecfield[i+1*size]
			 << setw(15) << setprecision(6) << vecfield[i+2*size] << endl;

	} 
	vtkF.close();

	printf("File %s created! \n",filename);

}
void write_dat(const char *filename, int size, float *data) {
	char	file1[255] = "", file2[255] = "";
	
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".dat");

	printf(" Writing file %s \n",file1);
	ofstream datF(file1,ios::out | ios::trunc);
	for(int i = 0; i < size; i++) {
			datF << setw(10) << setprecision(7) << data[i] << endl;
	}

	datF.close();
}
void write_NInput_dat(const char *filename, int Ncol, int size, float *data) {
	char	file1[255] = "", file2[255] = "";
	
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".dat");

	printf(" Writing file %s \n",file1);
	ofstream datF(file1,ios::out | ios::trunc);
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < Ncol; j++)
			datF << setw(16) << setprecision(6) << data[i+j*size];
		datF << endl;
	}

	datF.close();
}
void write_dat(const char *filename, int size, double *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	for(int i = 0; i < size; i++) {
		datF << setw(10) << setprecision(7) << data[i] << endl;
	}
	printf("File %s created \n",filename);
	
	datF.close();
}
void write_mat(const char *filename, int size, double *data) {
	char	file1[255] = "", file2[255] = "";
	
	

	printf(" Writing file %s \n",filename);
	ofstream datF(filename,ios::out | ios::trunc);
	datF << "gamma={\t";
	for(int i = 0; i < size-1; i++) {
			datF << setw(10) << setprecision(7) << data[i] << "," << endl;
	}
	datF << setw(10) << setprecision(7) << data[size-1] << "\t}" << endl;

	datF.close();
}
void write_dat(const char *filename, int size, short *data) {
	char	file1[255] = "", file2[255] = "";
	
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".dat");

	ofstream datF(file1,ios::out | ios::trunc);
	int sum1 = 0;
	for(int i = 0; i < size; i++) {
		datF << setw(10) << setprecision(7) << data[i] << endl;
		sum1+=data[i];
	}
	printf("Hit rato from %i \n",sum1);
	
	datF.close();
}
void write_dat(const char *filename, int size, int *data) {
	char	file1[255] = "", file2[255] = "";
	
	strcat(file1,outDir);
	strcat(file1,filename);
	strcat(file1,".dat");

	ofstream datF(file1,ios::out | ios::trunc);
	int sum1 = 0;
	for(int i = 0; i < size; i++) {
		datF << setw(10) << setprecision(7) << data[i] << endl;
		sum1+=data[i];
	}
	printf("Hit rato from %i \n",sum1);
	
	datF.close();
}
void write_dat(const char *filename, int size, unsigned int *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	for(int i = 0; i < size; i++) {
		datF << setw(10) << setprecision(7) << data[i] << endl;
	}
	printf("File %s created \n",filename);
	
	datF.close();
}
void write_Mathematica2Di(const char *filename, int size[2], unsigned int *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	datF << "rho = {";

	for(int j = 0; j < size[1]; j++) {
		datF << "\t{";
		for(int i = 0; i < size[0]; i++)	{
			datF << setw(12)  << data[i+j*size[0]];
			//datF << data[i+j*size[0]];
			if (i != size[0]-1) datF << ",";
		}
		datF << "}";
		if (j != size[1]-1) datF << "," << endl << "\t";
	}

	datF << "}";
	datF.close();
	printf("file %s created!\n",filename);
}
void write_Mathematica2Df(const char *filename, int size[2], float *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	datF << "rho = {";

	for(int j = 0; j < size[1]; j++) {
		datF << "\t{";
		for(int i = 0; i < size[0]; i++)	{
			datF << setiosflags(ios::fixed) << setprecision(8) << setw(12)  << data[i+j*size[0]];
			//datF << data[i+j*size[0]];
			if (i != size[0]-1) datF << ",";
		}
		datF << "}";
		if (j != size[1]-1) datF << "," << endl << "\t";
	}

	datF << "}";
	datF.close();
	printf("file %s created!\n",filename);
}
void write_Mathematica2D(const char *filename, int size[2], double *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	datF << "rho = {";

	for(int j = 0; j < size[1]; j++) {
		datF << "\t{";
		for(int i = 0; i < size[0]; i++)	{
			datF << setiosflags(ios::fixed) << setprecision(8) << setw(12)  << data[i+j*size[0]];
			//datF << data[i+j*size[0]];
			if (i != size[0]-1) datF << ",";
		}
		datF << "}";
		if (j != size[1]-1) datF << "," << endl << "\t";
	}

	datF << "}";
	datF.close();
	printf("file %s created!\n",filename);
}

void write_gnuplotMatrix(const char *filename, int size[2], float *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	for(int j = 0; j < size[1]; j++) {
		for(int i = 0; i < size[0]; i++)	{
			datF << setprecision(8) << setw(12)  << data[i+j*size[0]] << "\t";
		}
		if (j != size[1]-1) datF << endl;
	}
	datF.close();
	printf("file %s created!\n",filename);
}

void write_gnuplot2Dfunc(const char *filename, int size[2], float step[2], float *data) {

	ofstream datF(filename,ios::out | ios::trunc);
	for(int j = 0; j < size[1]; j++)
		for(int i = 0; i < size[0]; i++)
			datF << "\t" << setprecision(6) << setw(10)  << i*step[0] << "\t" << j*step[1]<< "\t" << data[i+j*size[0]] <<  endl;

	datF.close();
	printf("file %s created!\n",filename);
}
void write_sparse(const char *filename, int size, int *rows,int *cols, float* data){
// dump matrix saved in COO sparse matrix format

	ofstream datF(filename,ios::out | ios::trunc);
	datF << size << endl;
	for(int i = 0; i < size; i++) {
			datF << setw(10) << rows[i] << setw(10) << cols[i]  << setw(10) << setprecision(7) << data[i] << endl;
	}
	datF.close();
	printf(" File %s created \n",filename);
};

void write_sparse2(const char *filename, int size, unsigned int *rows, unsigned int *cols, float* data1, float* data2) {
// dump matrix saved in COO sparse matrix format

	ofstream datF(filename,ios::out | ios::trunc);
	datF << size << endl;
	for(int i = 0; i < size; i++) {
			datF << setw(10) << rows[i] << setw(10) << cols[i]  << setw(12) << setprecision(6) << data1[i] << setw(12) << setprecision(6) << data2[i] << endl;
	}

	datF.close();
	printf(" File %s created \n",filename);
};
