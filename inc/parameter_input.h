#pragma once
#include <sstream>
#include <string>
#include <map>

const int	MAX_KEYW_LEN = 256;
//for file input:
enum	parValue  {	NOT_DEFINED,
					WRT_BIN,
					VOLUME_PERCENT,
					VOLUME_PERCENT_LIST,
					VOL_FRAC_PRECISION,
					SPAT_DISTR_FILE,
					PREF_ORT_PHI,
					PREF_ORT_THETA,
					DEF_THETA_MED,
					DEF_PHI_MED,
					DEF_THETA_DEV,
					DEF_PHI_DEV,
						SPAT_DIMENSIONS,
						ORNT_DISTR_FILE,
						IDIM_DISTR_FILE,
					INC_LEN_MED,
					INC_LEN_DEV,
					INC_RAD_MED,
					INC_RAD_DEV,
						INC_SEPARATION,
						INC_SOFTCORE,
						MATH_TOLERANCE,
						DMN_CELL_MARGE,
					CLUST_FLAG,
					CLUST_DIST_LIM,
					CLUST_ANGL_LIM,
					STAT_DIST_LIM,
					SELF_ALLIGN_STEP,
					REDUCE_FLAG
					};
static map<string,parValue> parMap;
int initParMap(void) {
	parMap["WRT_BIN"]				=	WRT_BIN;
	parMap["VOLUME_PERCENT"]		=	VOLUME_PERCENT;
	parMap["VOLUME_PERCENT_LIST"]		=	VOLUME_PERCENT_LIST;
	parMap["VOL_FRAC_PRECISION"]	=	VOL_FRAC_PRECISION;
	parMap["SPAT_DISTR_FILE"]		=	SPAT_DISTR_FILE;
	parMap["PREF_ORT_PHI"]			=	PREF_ORT_PHI;
	parMap["PREF_ORT_THETA"]		=	PREF_ORT_THETA;
	parMap["DEF_THETA_MED"]			=	DEF_THETA_MED;
	parMap["DEF_PHI_MED"]			=	DEF_PHI_MED;
	parMap["DEF_THETA_DEV"]			=	DEF_THETA_DEV;
	parMap["DEF_PHI_DEV"]			=	DEF_PHI_DEV;
	parMap["SPAT_DIMENSIONS"]		=	SPAT_DIMENSIONS;
	parMap["ORNT_DISTR_FILE"]		=	ORNT_DISTR_FILE;
	parMap["IDIM_DISTR_FILE"]		=	IDIM_DISTR_FILE;
	parMap["INC_LEN_MED"]			=	INC_LEN_MED;
	parMap["INC_LEN_DEV"]			=	INC_LEN_DEV;
	parMap["INC_RAD_MED"]			=	INC_RAD_MED;
	parMap["INC_RAD_DEV"]			=	INC_RAD_DEV;
	parMap["INC_SEPARATION"]		=	INC_SEPARATION;
	parMap["INC_SOFTCORE"]			=	INC_SOFTCORE;
	parMap["MATH_TOLERANCE"]		=	MATH_TOLERANCE;
	parMap["DMN_CELL_MARGE"]		=	DMN_CELL_MARGE;
	parMap["CLUST_FLAG"]			=	CLUST_FLAG;
	parMap["CLUST_DIST_LIM"]		=	CLUST_DIST_LIM;
	parMap["CLUST_ANGL_LIM"]		=	CLUST_ANGL_LIM;
	parMap["SELF_ALLIGN_STEP"]		=	SELF_ALLIGN_STEP;
	parMap["STAT_DIST_LIM"]			=	STAT_DIST_LIM;
	parMap["REDUCE_FLAG"]			=	REDUCE_FLAG;
return 1;
}

string	rm_spacers( string str) {
	str.erase(0,str.find_first_not_of(" \t"));
	return str;
}
