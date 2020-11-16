// kdTree class definitions:
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <string.h>
#include "kdTree.h"

// IO directory name:
extern const char outDir[255];

//public:

kdTree::kdTree(int Total, int ext[], float *Mesh) {

	mesh = Mesh;
	leafNum = 1;
	total = Total;
	
	root.lt = NULL;
	root.rt = NULL;

	root.mask[0][0] = 0; root.mask[0][1] = ext[0]-1;
	root.mask[1][0] = 0; root.mask[1][1] = ext[1]-1;
	root.mask[2][0] = 0; root.mask[2][1] = ext[2]-1;

	integrate(&(root));
}
kdTree::kdTree(int Total, int ext[], int *Mesh) {
	int	size = ext[0]*ext[1]*ext[2];
	float *fmesh = (float *) malloc(size*sizeof(float));
	for(int i=0;i<size;i++) {
		fmesh[i] = (float) Mesh[i];
	}
	mesh = fmesh;
	leafNum = 1;
	total = Total;
	
	root.lt = NULL;
	root.rt = NULL;

	root.mask[0][0] = 0; root.mask[0][1] = ext[0]-1;
	root.mask[1][0] = 0; root.mask[1][1] = ext[1]-1;
	root.mask[2][0] = 0; root.mask[2][1] = ext[2]-1;

	integrate(&(root));
}

void		kdTree::build(int maxNorm) {
	this->Cryt = ((float) maxNorm);
	branch(&(this->root));
#ifdef _DEBUG
	printf("_DEBUG: Leafs created: %i \n",this->leafNum);
#endif
}

int			*kdTree::leaves() {
	int	*lf_arr =(int *) malloc(6*this->leafNum*sizeof(int));
	int	*lf_ptr;
	lf_ptr = lf_arr;
	fill_leaves(&lf_ptr,this->root);
	
return lf_arr;
}



void		kdTree::print(const char *base_name) {

int	dim[3];
dim[0] = this->root.mask[0][1]+1;
dim[1] = this->root.mask[1][1]+1;
dim[2] = this->root.mask[2][1]+1;
int size = dim[0]*dim[1]*dim[2];

float	*clr_tree = colorTree();

	char outFile[255] = "";
	strcat(outFile,outDir);
	strcat(outFile,base_name);
	ofstream vtkF(outFile,ios::out | ios::trunc);
	
	vtkF << "# vtk DataFile Version 2.0" << endl;
	vtkF << "Tree-like splited probability map" << endl;
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
	
#ifdef H_Cstyle_Mesh
	for (int k=0;k<dim[2];k++) {
		for (int j=0;j<dim[1];j++) {
			for (int i=0;i<dim[0];i++) {
				vtkF << setw(10) << setprecision(7) << clr_tree[2*(i*dim[1]*dim[2]+j*dim[2]+k)] << endl;
			}}} 
	vtkF << "SCALARS Region float 1" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int k=0;k<dim[2];k++) {
		for (int j=0;j<dim[1];j++) {
			for (int i=0;i<dim[0];i++) {
				vtkF << setw(10) << setprecision(7) << clr_tree[2*(i*dim[1]*dim[2]+j*dim[2]+k)+1] << endl;
			}}}
#else
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << clr_tree[2*i] << endl;
	}
	vtkF << "SCALARS Region float 1" << endl;
	vtkF << "LOOKUP_TABLE default" << endl;
	for (int i=0;i<size;i++) {
		vtkF << setw(10) << setprecision(7) << clr_tree[2*i+1] << endl;
	}
#endif
	vtkF.close();

	//printf(" Colored Tree created! \n");	
}



//private:

void		kdTree::branch(kdNode *Node) {
	int ext[3];
	ext[0] = Node->mask[0][1]+1-Node->mask[0][0];
	ext[1] = Node->mask[1][1]+1-Node->mask[1][0];
	ext[2] = Node->mask[2][1]+1-Node->mask[2][0];
	if ((Node->norm > this->Cryt)&&(ext[0]*ext[1]*ext[2]>1)) {
		split(Node);
		branch(Node->lt);
		branch(Node->rt);
		(this->leafNum)++;
	}
}

float		*kdTree::colorTree() {
	int	dim[3];
	dim[0] = this->root.mask[0][1]+1;
	dim[1] = this->root.mask[1][1]+1;
	dim[2] = this->root.mask[2][1]+1;
	int size = dim[0]*dim[1]*dim[2];

	float	*clr_tree = (float *) malloc(2*size*sizeof(float));
	int	*leaf_msk = leaves();

	for (int l=0;l<this->leafNum;l++) {

#ifdef H_Cstyle_Mesh
		for (int i=leaf_msk[6*l+0];i<=leaf_msk[6*l+1];i++) {
			for (int j=leaf_msk[6*l+2];j<=leaf_msk[6*l+3];j++) {
				for (int k=leaf_msk[6*l+4];k<=leaf_msk[6*l+5];k++) {
					clr_tree[2*(i*dim[1]*dim[2]+j*dim[2]+k)+0] = this->mesh[i*dim[1]*dim[2]+j*dim[2]+k];
					clr_tree[2*(i*dim[1]*dim[2]+j*dim[2]+k)+1] = l;
		}}} 
#else
		int maxLoad = 0;
		for (int i=leaf_msk[6*l+0];i<=leaf_msk[6*l+1];i++) {
			for (int j=leaf_msk[6*l+2];j<=leaf_msk[6*l+3];j++) {
				for (int k=leaf_msk[6*l+4];k<=leaf_msk[6*l+5];k++) {
					clr_tree[2*(k*dim[1]*dim[0]+j*dim[0]+i)+0] = this->mesh[k*dim[1]*dim[0]+j*dim[0]+i];
					clr_tree[2*(k*dim[1]*dim[0]+j*dim[0]+i)+1] = (float) (l%2);
		}}} 
#endif
	}
return clr_tree;
}

void		kdTree::fill_leaves(int **lf_ptr,kdNode Node) {
	if ((Node.lt == NULL)&&(Node.rt == NULL)) {
		int	*ptr = *lf_ptr;
		*ptr++ = Node.mask[0][0];
		*ptr++ = Node.mask[0][1];
		*ptr++ = Node.mask[1][0];
		*ptr++ = Node.mask[1][1];
		*ptr++ = Node.mask[2][0];
		*ptr++ = Node.mask[2][1];
		*lf_ptr = ptr;
		int vol = Node.mask[0][1] - Node.mask[0][0]+1;
			vol*= Node.mask[1][1] - Node.mask[1][0]+1;
			vol*= Node.mask[2][1] - Node.mask[2][0]+1;
		//printf("Leaf norm: %f / %f - volume: %i = %i-%ix%i-%ix%i-%i \n",Node.norm,this->Cryt,vol,Node.mask[0][1], Node.mask[0][0],Node.mask[1][1], Node.mask[1][0],Node.mask[2][1], Node.mask[2][0]);
	} else {
		fill_leaves(lf_ptr,*Node.lt);
		fill_leaves(lf_ptr,*Node.rt);
	}
}

kdTree::kdNode		*kdTree::make_kdNode(int Mask[3][2]) {
	kdNode *Node = (kdNode *) malloc(sizeof(kdNode));

	Node->mask[0][0] = Mask[0][0]; Node->mask[0][1] = Mask[0][1];
	Node->mask[1][0] = Mask[1][0]; Node->mask[1][1] = Mask[1][1];
	Node->mask[2][0] = Mask[2][0]; Node->mask[2][1] = Mask[2][1];

	Node->norm = 0;

	Node->lt =	NULL;
	Node->rt =	NULL;

return Node;
}

double		kdTree::integrate(kdNode *Node) {

	double	sum = 0.0;
	double	xc = 0.0;
	double	yc = 0.0;
	double	zc = 0.0;

	int		ext[3];
	ext[0] = this->root.mask[0][1]+1;
	ext[1] = this->root.mask[1][1]+1;
	ext[2] = this->root.mask[2][1]+1;

#ifdef H_Cstyle_Mesh
	for(int i = Node->mask[0][0]; i <= Node->mask[0][1]; i++)
		for(int j = Node->mask[1][0]; j <= Node->mask[1][1]; j++)
			for(int k = Node->mask[2][0]; k <= Node->mask[2][1]; k++) {
				sum += *(this->mesh + i*ext[2]*ext[1] + j*ext[2] + k);
				xc += *(this->mesh + i*ext[2]*ext[1] + j*ext[2] + k)*i;
				yc += *(this->mesh + i*ext[2]*ext[1] + j*ext[2] + k)*j;
				zc += *(this->mesh + i*ext[2]*ext[1] + j*ext[2] + k)*k;
			}
#else
	for(int k = Node->mask[2][0]; k <= Node->mask[2][1]; k++)
		for(int j = Node->mask[1][0]; j <= Node->mask[1][1]; j++)
			for(int i = Node->mask[0][0]; i <= Node->mask[0][1]; i++) {
				sum += *(this->mesh + i + j*ext[0] + k*ext[0]*ext[1]);
				xc += *(this->mesh + i + j*ext[0] + k*ext[0]*ext[1])*i;
				yc += *(this->mesh + i + j*ext[0] + k*ext[0]*ext[1])*j;
				zc += *(this->mesh + i + j*ext[0] + k*ext[0]*ext[1])*k;
			}
#endif
	xc /= sum;
	yc /= sum;
	zc /= sum;

	Node->norm = (float) sum;
	Node->cm[0] = (int) floor(xc+0.5f);
	Node->cm[1] = (int) floor(yc+0.5f);
	Node->cm[2] = (int) floor(zc+0.5f);

	return sum;
}

void		kdTree::split(kdNode *Node) {


	float	Min_diff = numeric_limits<float>::infinity();
	float	diff;
	int		minj;
	int		mindir;
	Node->lt = make_kdNode(Node->mask);
	Node->rt = make_kdNode(Node->mask);

#ifdef H_EQU_LOAD
	for(int i = 0;i<3;i++) {
		if (Node->mask[i][1] != Node->mask[i][0]) {
			for (int j = -1; j < 1; j++) {
				if ((Node->cm[i]+j>=Node->mask[i][0])&&(Node->cm[i]+j+1<=Node->mask[i][1])) {

					Node->lt->mask[i][1] = Node->cm[i]+j;
					Node->rt->mask[i][0] = Node->cm[i]+j+1;		
		
					// minimal load difference:
					diff = (float) abs(integrate(Node->lt)-integrate(Node->rt));
					if (diff<Min_diff) {
						Min_diff = diff;
						mindir = i;
						minj = j;
					}
					Node->lt->mask[i][1] = Node->mask[i][1];
					Node->rt->mask[i][0] = Node->mask[i][0];
				}
			}
		}
	}
#else
	int		ext[3];
	ext[0] = Node->mask[0][1]+1-Node->mask[0][0];
	ext[1] = Node->mask[1][1]+1-Node->mask[1][0];
	ext[2] = Node->mask[2][1]+1-Node->mask[2][0];
	mindir = (ext[0]>ext[1]?0:1);
	mindir = (ext[mindir]>ext[2]?mindir:2);
	if (Node->mask[mindir][1] != Node->mask[mindir][0]) {
		for (int j = -1; j < 1; j++) {
			if ((Node->cm[mindir]+j>=Node->mask[mindir][0])&&(Node->cm[mindir]+j+1<=Node->mask[mindir][1])) {
				Node->lt->mask[mindir][1] = Node->cm[mindir]+j;
				Node->rt->mask[mindir][0] = Node->cm[mindir]+j+1;		
		
				diff = abs(integrate(Node->lt)-integrate(Node->rt));
				if (diff<Min_diff) {
					Min_diff = diff;
					minj = j;
				}
				Node->lt->mask[mindir][1] = Node->mask[mindir][1];
				Node->rt->mask[mindir][0] = Node->mask[mindir][0];
			}
		}
	}
#endif

	Node->lt->mask[mindir][1] = Node->cm[mindir]+minj;
	Node->rt->mask[mindir][0] = Node->cm[mindir]+minj+1;

	integrate(Node->lt);
	integrate(Node->rt);

}
