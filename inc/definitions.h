// Macros to be used in calculations:
#pragma once

// float precision
#define DEV_PRECISION	0.000001f

#define BLOCK			1024

// core interaction kernel - operations:
#define	INTERSECT_ELIM		0	// Eliminating just-generated inclusions if contacts found
#define CONTACT_COUNT		1	// Counting pairs of contacting inclusions:
#define CONTACT_RECORD		2	// Recording pairs of contacting inclusions:
#define VIRTUAL_CHECK		3	// Marking virtual inclusions - pair interactions as redundant if crossed

// contact estimation - virtual inclusions:
#define	VIRT_INC_EMPTY		0
#define	VIRT_INC_PRESENT	1
#define	VOID_ELEC	UINT_MAX
#define	SRC_ELEC	UINT_MAX
#define	SNK_ELEC	SRC_ELEC-1

#define MAX_INC_FIELDS		9
#define DEF_EMPTY_VAL		-1.0f
#define EMPTY_SCALE			-2.0f
#define	MIN_SEPARATION		0.00034f

#define DT_CRIT_DEF			1.0f
