!     
! File:   Mod_Distribution.f90
! Author: sergey
!
! Created on March 3, 2011, 4:13 PM
!
! Contains distribution functions:  Uniform:    UniEL_3D
!                                   Gaussian:   Gauss_3D,Gauss_1D
!                                   Lognormal:  Log_Norm(to be)
!
MODULE Mod_Distribution

double precision,parameter :: PI=ACOS(-1.D0) !3.14159265359D0

CONTAINS

    double precision FUNCTION SIGM_1D(x,r,a)
        real    ::  x   ! probe point
        real    ::  r   ! center
        real    ::  a   ! distribution slope
        SIGM_1D=1/(1+exp(-a*(x-r)))
    END FUNCTION SIGM_1D

    double precision FUNCTION DSIGM_1D(x,r,w,a)
        real    ::  x   ! probe point
        real    ::  r   ! center
        real    ::  w   ! distribution half-width
        real    ::  a   ! distribution slope
        DSIGM_1D=SIGM_1D(x,r-w,a)-SIGM_1D(x,r+w,a)
    END FUNCTION DSIGM_1D
    
    double precision FUNCTION Gauss_1D(x,r,w)
        real    ::  x   ! probe point
        real    ::  r   ! center
        real    ::  w   ! distribution width
        Gauss_1D=exp(-(x-r)*(x-r)/2./w/w)
    END FUNCTION Gauss_1D

    double precision function  Gauss_3D(X,R,w)
        real,dimension(1:3) ::  X   ! array of probe point coordinates
        real,dimension(1:3) ::  R   ! array of aglomerate center coordinates
        real				::  w   ! array of distribution halfwidths
        Gauss_3D=Gauss_1D(X(1),R(1),w)*&
                 Gauss_1D(X(2),R(2),w)*&
                 Gauss_1D(X(3),R(3),w)
    END FUNCTION Gauss_3D
	
	double precision  FUNCTION LogNrm_1D(x,a,w)
		real    ::  x   ! probe point
        real    ::  a   ! center
        real    ::  w   ! distribution width
		if (x>=0.001) then
			LogNrm_1D=1.0/x/sqrt(2*PI)/w*exp(-((log(x/a))**2)/2./w/w)
		else
			LogNrm_1D=0.0
		endif
	END FUNCTION LogNrm_1D

END MODULE Mod_Distribution
