// This file describes kdTree class to be used for optimal space partitioning
// both on mGPU- and grid-block levels
#pragma once

using namespace std;

//#define H_Cstyle_Mesh
#define H_EQU_LOAD

class kdTree {

public:

	kdTree(int norm, int ext[3], float *mesh);		// creates root node with given arguments
	kdTree(int norm, int ext[3], int *mesh);		// wrapper for int arrays;

	void	build(int maxNorm);						// creates nodes until node norm is no larger then maxNorm
	void	print(const char *base_name);			// writes vtk file to visualise division
	int		*leaves();								// returnes the array of leaf masks
	int		leafNum;								// number of leaf-nodes


private:

	typedef struct kdNode{
		kdNode	*lt;
		kdNode	*rt;
		
		float	norm;
		int	cm[3];
		int	mask[3][2];
	} kdNode;


	kdNode		root;
	float		*mesh;								// mesh data
	int			total;								// desired CNT number

	float		Cryt;								// = total mesh integral / by desired norm
	
	void	branch(kdNode *Node);					// recursively splits given subtree
	float	*colorTree();
	void	fill_leaves(int **lf_ptr,kdNode Node);// recursively traverses through tree leaves
	double	integrate(kdNode *Node);				// considers step to be 1
	kdNode	*make_kdNode(int mask[3][2]);
	void	split(kdNode *Node);
	

};
