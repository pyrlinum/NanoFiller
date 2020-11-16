#define SIGNED_IMG

// structure for image data:
struct grscl_img_dscr_t {
int	width;		// image width in pixels
int	length;		// image length in pixels
char	*data;	// data to plot - width x length array 
};

int	grscl_TIFFwrite(const char *fname, grscl_img_dscr_t dscr);	// writes data defined in DSCR into uncompressed grayscale tiff file.