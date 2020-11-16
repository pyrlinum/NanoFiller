/* CNT_web_cuda
This file contains entrance point to the application,
initiates CUDA ode and handles output 
*/
#include <cuda_runtime.h>
#include "CNT.h"
#include "IO.h"
#include "simPrms.h"

#ifdef WIN32
#include "windows.h"
#endif

int select_device();					// select device function
int cudaMain(simPrms *Simul);

//void save_state_TXT(const char* fname, simPrms Simul);
void save_state_BIN(const char* fname, simPrms Simul);
void read_state_BIN(const char* fname, simPrms *Simul);

int	plotSlab(const char* fname, float	Zlvl, float	Zhth, int pxlCntXY[2], simPrms *Simul, float	def_Tol);	// defined in plot_kernel.cu

//int collectStats(int type, int numBins, float sampleRange, char* fname, simPrms Simul);	// function to call statistics kernel:
int collect_PhiTheta_distr(const char *fname, int	numBinsX, int numBinsY,	simPrms	Simul);
int collect_1D_statDistr_weightedFloat( float range_lim, const char *fname,	int	bins1D,		simPrms	Simul);
//int collect_exclSurfArea(	float range_lim, simPrms	Simul);
int collect_exclSurfArea2(	float range_lim, simPrms	Simul);
//int collect_numContacts(	float range_lim, simPrms	Simul);
int collect_CntctPerInc(	float range_lim, simPrms	Simul);

int2 init_contacts_map(simPrms *Simul, char efield_dir);
int reduce(	simPrms	*Simul, int2 cond_grid, int min_cont );
int collect_cunduct_data(int dir, int internal_flag, simPrms *Simul);
int collect_vdw(	float range_lim, simPrms	Simul);
float get_nematic_order(	float3 	vec, simPrms	*Simul	);

int main(int argc, char** argv) {
int	Device;
	if (argc>=2) {
		Device = select_device();
		if (Device != -1) {

			string infile = string(argv[1]);
			simPrms Simul_set = simPrms();
			Simul_set.set_ParamsFromFile(argv[1]);

			int argP0 = 2;
			if(strcmp(argv[argP0],"generate")==0) { 
				int	cntNum = cudaMain(&Simul_set);
				printf("%i inclusions generated! \n",cntNum);
				float volume_fraction = Simul_set.get_VolFrac(cntNum);
				printf("Volume fraction of %5.2f percent is achieved! \n",volume_fraction);
				int new_ttlCNT = Simul_set.repack();
				if (Simul_set.wrt_bin_flag)  save_state_BIN("output.bin",Simul_set);
				argP0++;
			} else read_state_BIN("output.bin",&Simul_set);

			for (int argP=argP0;argP<argc;argP++) {
				if (strcmp(argv[argP],"plot")==0) {	
					float	Zlvl = Simul_set.physDim[2];
					float	Zhth = Simul_set.physDim[2];
					int		pxlSizeXY[2];
							pxlSizeXY[0] = 1280;
							pxlSizeXY[1] = 1280;
					plotSlab("output.tiff",Zlvl,Zhth,pxlSizeXY,&Simul_set,Simul_set.TOL);
				}
				if (strcmp(argv[argP],"expand")==0)	{
					int dir = atoi(argv[++argP]);
					Simul_set.expand(dir);
				}
				if (strcmp(argv[argP],"reduce")==0)	{
					int min_cont = 2;
					printf("Eliminating inclusions with less then %d contacts up to %f um: \n",min_cont,Simul_set.stat_dist_lim);
					//Simul_set.save_dmnOcc("expanded.dmn");
					float	range_lim = Simul_set.stat_dist_lim;
					char	dir = atoi(argv[++argP]);
					int		oldNum = Simul_set.numCNT;
					int		iter=0;
					int2 grid = init_contacts_map(&Simul_set, dir);
					while ((iter==0)||(Simul_set.numCNT!=oldNum)) {
						for(int i=0;i<80;i++) std::cout << "-";
						printf("\nIteration: %d\n",iter);
						oldNum = Simul_set.numCNT;
						int suc_flag = reduce(&Simul_set, grid, min_cont);
						if ( suc_flag == 0) printf("Error during reduction: %d \n",suc_flag);
						iter++;
					}
					Simul_set.repack();
					//Simul_set.save_dmnOcc("reduced.dmn");
					if (Simul_set.wrt_bin_flag)  save_state_BIN("reduced.bin",Simul_set);
				}
				if (strcmp(argv[argP],"stat_nematic")==0)	{
					//float	ortX = atof(argv[++argP]);
					//float	ortY = atof(argv[++argP]);
					//float	ortZ = atof(argv[++argP]);
					//float3	ort(ortX,ortY,ortZ);
					float3	ort = make_float3(0.0,0.0,1.0);
					float S1 = get_nematic_order(ort,&Simul_set);
					printf("Estimated nematic order parameter: %f\n",S1);
				}
				if (strcmp(argv[argP],"stat_phitheta")==0)	{
					int		binsX = 32;
					int		binsY = 32;
					collect_PhiTheta_distr("rho.dat", binsX, binsY,	Simul_set);;
				}
				if (strcmp(argv[argP],"stat_mutangle")==0)	{
					int		binsX = 32;
					float	range_lim = Simul_set.stat_dist_lim;
					collect_1D_statDistr_weightedFloat(range_lim,"gamma.dat",binsX,Simul_set);
				}
				if (strcmp(argv[argP],"stat_exclsurf")==0)	{
					float	range_lim = Simul_set.stat_dist_lim;
					//collect_exclSurfArea(range_lim, Simul_set);
					collect_exclSurfArea2(range_lim, Simul_set);
				}
				if (strcmp(argv[argP],"stat_contacts")==0)	{
					float	range_lim = Simul_set.stat_dist_lim;
					//printf("COLLECTING BODY-2-BODY contacts: \n");
					//collect_numContacts(range_lim, Simul_set);
					//printf("COLLECTING ALL contacts: \n");
					collect_CntctPerInc(range_lim, Simul_set);
				}
				if (strcmp(argv[argP],"stat_conduct")==0)	{
					printf("Building conductivity sparse matrix \n");
					float	range_lim = Simul_set.stat_dist_lim;
					int dir = atoi(argv[++argP]);
					int internal_flag = (++argP<argc ? (strcmp(argv[argP],"internal")==0) : 0);
					collect_cunduct_data(dir, internal_flag, &Simul_set);
					if (!internal_flag) --argP;
				}
				if (strcmp(argv[argP],"stat_vdw")==0)	{
					float	range_lim = Simul_set.stat_dist_lim;
					collect_vdw(range_lim, Simul_set);
				}
				if (strcmp(argv[argP],"write_vtk")==0)	{
					printf("CNTs created: %i \n",Simul_set.numCNT);
					CNT_t *h_arr = Simul_set.asmbl_cnt_data();
					write_CNTvtk("output", Simul_set.numCNT, h_arr);
				}
				if (strcmp(argv[argP],"write_ort")==0)	{
					printf("CNTs created: %i \n",Simul_set.numCNT);
					CNT_t *h_arr = Simul_set.asmbl_cnt_data();
					write_CNT_ORTvtk("orient", Simul_set.numCNT, h_arr);
				}
				if (strcmp(argv[argP],"write_lmp")==0)	{
					printf("CNTs created: %i \n",Simul_set.numCNT);
					CNT_t *h_arr = Simul_set.asmbl_cnt_data();
					write_CNTlmp("output", Simul_set.numCNT, h_arr, Simul_set.physDim, Simul_set.density);
				}
			}
		} else printf(" No CUDA-enabled GPU found!!! ");
	
	} else printf(" Error in command line arguments: at least filename and 1 command required!!! \n");
#ifndef BATCH
	//getchar();
#endif

#ifdef WIN32
	//while(!(GetAsyncKeyState(VK_RETURN))) {}
#endif
	return 0;
}
