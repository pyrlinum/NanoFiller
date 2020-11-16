#!/bin/bash

MKL_ROOT=/opt/intel/mkl

all:
	g++ -I${MKL_ROOT}/include -fPIC -c -openmp -o solver_dss.o solver.cpp
	g++ -shared -o solver_dss.so solver_dss.o ${MKL_ROOT}/lib/intel64/libmkl_rt.so
