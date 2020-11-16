// This file contains interface functions to the core kernel describing cell interaction
#pragma once

// Check intersections:

// Count pair contacts:

// Record pair contacts:

// mark screened pair contacts:
// isnt_screened is an array of flags corresponding to whether a contact is not screened by an obstructing inclusion
//extern "C"
void	check_screened_Interactions(	thrust::device_vector<char>&			isnt_screened,
										thrust::device_vector<unsigned int>&	d_virtAddr,
										thrust::device_vector<unsigned int>&	d_virtOcc,
										thrust::device_vector<float>&			d_virt_inc,
										simPrms* Simul,
										int dir );
