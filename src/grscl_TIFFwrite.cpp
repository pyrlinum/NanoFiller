#include "grscl_TIFFwrite.h"
#include <stdlib.h>
#include <tiffio.h>

int	grscl_TIFFwrite(const char *fname, grscl_img_dscr_t dscr) {
	TIFF	*img = TIFFOpen(fname,"w");

	// Image description info (set to grayscale)
	TIFFSetField(img,TIFFTAG_IMAGEWIDTH,		dscr.width);
	TIFFSetField(img,TIFFTAG_IMAGELENGTH,		dscr.length);
	TIFFSetField(img,TIFFTAG_BITSPERSAMPLE,		8);
	TIFFSetField(img,TIFFTAG_SAMPLESPERPIXEL,	1);
	TIFFSetField(img,TIFFTAG_ROWSPERSTRIP,		dscr.length);

	TIFFSetField(img,TIFFTAG_COMPRESSION,		COMPRESSION_NONE);
	TIFFSetField(img,TIFFTAG_PHOTOMETRIC,		PHOTOMETRIC_MINISWHITE);
	TIFFSetField(img,TIFFTAG_FILLORDER,			FILLORDER_LSB2MSB);
	TIFFSetField(img,TIFFTAG_PLANARCONFIG,		PLANARCONFIG_CONTIG);

	TIFFSetField(img,TIFFTAG_XRESOLUTION,		150);
	TIFFSetField(img,TIFFTAG_YRESOLUTION,		150);
	TIFFSetField(img,TIFFTAG_RESOLUTIONUNIT,	RESUNIT_INCH);

#ifdef SIGNED_IMG
	TIFFSetField(img,TIFFTAG_ARTIST,			"Sergey Pyrlin");
	TIFFSetField(img,TIFFTAG_COPYRIGHT,			"Marie Curie ITN CONTACT");
	TIFFSetField(img,TIFFTAG_SOFTWARE,			"CNTweb");
	TIFFSetField(img,TIFFTAG_IMAGEDESCRIPTION,	"Created by downsampling of B&W projection of quasirandom generated CNT-composite sample 3D model");
#endif
	// writing image as a single stripe:
	TIFFWriteEncodedStrip(img,0,dscr.data,dscr.width*dscr.length);

	TIFFClose(img);
	
	return 0;

}