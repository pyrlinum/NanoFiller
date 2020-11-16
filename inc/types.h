#pragma once

typedef thrust::device_vector<unsigned int>::iterator	uintIter;
typedef thrust::device_vector<unsigned char>::iterator	uchrIter;
typedef thrust::device_vector<unsigned short>::iterator	ushtIter;
typedef thrust::device_vector<float>::iterator			fltIter;
typedef thrust::tuple<uintIter,uintIter>				uint2Iter;
typedef thrust::tuple<uintIter,uchrIter>				uichIter;
typedef thrust::tuple<uintIter,ushtIter>				uishIter;
typedef thrust::tuple<uchrIter,uintIter>				uchiIter;
typedef thrust::tuple<ushtIter,uintIter>				ushiIter;
typedef thrust::tuple<uintIter,fltIter>					uifIter;
typedef thrust::tuple<fltIter,fltIter>					flt2Iter;
typedef thrust::tuple<uintIter,uintIter,fltIter>		ui2fIter;
typedef thrust::tuple<uintIter,fltIter,fltIter>			uif2Iter;
typedef thrust::zip_iterator<uint2Iter>					uint2ZipIter;
typedef thrust::zip_iterator<flt2Iter>					flt2ZipIter;
typedef thrust::zip_iterator<uichIter>					uichZipIter;
typedef thrust::zip_iterator<uchiIter>					uchiZipIter;
typedef thrust::zip_iterator<uishIter>					uishZipIter;
typedef thrust::zip_iterator<ushiIter>					ushiZipIter;
typedef thrust::zip_iterator<uifIter>					uifZipIter;
typedef thrust::zip_iterator<ui2fIter>					ui2fZipIter;
typedef thrust::zip_iterator<uif2Iter>					uif2ZipIter;
typedef thrust::tuple<	uintIter,
						uintIter,
						fltIter,
						fltIter>						uint2flt2Iter;
typedef thrust::zip_iterator<uint2flt2Iter>				ui2f2ZipIter;
typedef thrust::tuple<	uintIter,
						uintIter,
						uintIter,
						uintIter,
						fltIter,
						fltIter>						uint4flt2Iter;
typedef thrust::zip_iterator<uint4flt2Iter>				ui4f2ZipIter;


typedef thrust::tuple<fltIter,fltIter,fltIter,fltIter,fltIter,fltIter,fltIter,fltIter>	flt8Iter;
typedef thrust::zip_iterator<flt8Iter> 					flt8ZipIter;
typedef thrust::tuple<flt8ZipIter,uintIter>				flt8uInt_tup;
typedef thrust::zip_iterator<flt8uInt_tup>				flt8uInt_zIter;
