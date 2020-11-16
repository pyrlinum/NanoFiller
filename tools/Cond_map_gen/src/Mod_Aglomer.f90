!     
! File:   Mod_Aglomer.f90
! Author: sergey
!
! Created on March 3, 2011, 10:00 AM
!
! Describes aglomer type and subroutines to handle it
!
! Contains: t_aglo
!           function new_aglo
!           function are_merged
!           subroutine populate

MODULE Mod_Aglomer

USE Mod_Distribution
USE Mod_Descriptors
    implicit none
   TYPE :: t_aglo
        real,dimension(1:3) ::  c       ! Cartesian cordinates of aglomerate center (NORMALISED!!!)
        real				::  w       ! aglomerate distribition dispersion (NORMALISED!!!)
		real                ::  norm    ! Normalization constant related to averaged CNT concentration in aglomerate as norm=Veff*Cv

    END TYPE t_aglo

    CONTAINS

    ! Main subroutine:
    subroutine populate(A_pool,simul)
    type(t_simul),intent(inout)             :: simul
    type(t_aglo),dimension(:),intent(inout) :: A_pool
    type(t_aglo)                            :: probe
    integer                                 :: N,Aglo_count,I
    real,dimension(1:3)                     :: X
    real                                    :: w,a
    integer,dimension(1)                    :: seed
    real                                    :: coin,probT,vol,volT,maxvol
    logical                                 :: create_flag,merged_flag,fit_flag
        call system_clock(seed(1))
        call random_seed( )
        Aglo_count=1
        volT = 0
        maxvol = simul%box_size(1)*simul%box_size(2)*simul%box_size(3)&
                *simul%vf*simul%aglomer%ratio
                write(6,*) "aglomerates to be created:",simul%aglomer%max_num," to fill volume ",maxvol
        DO WHILE ((volT < maxvol).AND.(Aglo_count<=simul%aglomer%max_num))
            create_flag = .FALSE.
            DO WHILE ( .NOT. create_flag )  
                call random_number(w)
                probT  = LogNrm_1D(w*simul%box_size(1),simul%aglomer%avg_rad,simul%aglomer%rad_disp)
                call random_number(coin)
                if (coin <= probT) then
                    call random_number(X(1:3))
                    probe = new_aglo(X(1:3),w,simul%aglomer%density)
                    vol = 4.0/3.0*PI*(w*simul%box_size(1))**3
                    merged_flag = .FALSE.
                    I = 1
                    DO WHILE ( (I <= Aglo_count-1) .AND. (.NOT. merged_flag) )
                        merged_flag=are_merged(probe,A_pool(I))
                        I=I+1
                    ENDDO
                    create_flag = (.NOT.merged_flag).AND.(volT+vol<=maxvol*1.05)
                endif
            ENDDO
            A_pool(Aglo_count) = probe
            volT = volT + vol
            Aglo_count = Aglo_count + 1
        ENDDO
        write(6,*) (Aglo_count-1),' aglomerates created, total volume: ', volT
        simul%aglomer%num= Aglo_count-1
    end subroutine populate

    !-------------------------------------------------------------
    !                   AUXILARY SUBROUTINES:
    !-------------------------------------------------------------

    real function distance(P1,P2) ! mesure the distance using 3D periodic boundary conditions
    real,dimension(3)   :: P1,P2
    real,dimension(3)   :: box
    real                :: x,y,z
        x=MIN(abs(P1(1)-P2(1)),abs(P1(1)+1-P2(1)),abs(P1(1)-P2(1)-1))
        y=MIN(abs(P1(2)-P2(2)),abs(P1(2)+1-P2(2)),abs(P1(2)-P2(2)-1))
        z=MIN(abs(P1(3)-P2(3)),abs(P1(3)+1-P2(3)),abs(P1(3)-P2(3)-1))
        distance=sqrt(x*x+y*y+z*z)
    end function distance

	logical function are_merged(A1,A2)  ! check for the possibility to distinguish aglomerates
    type(t_aglo)        :: A1,A2
    type(t_simul)       :: S
    real                :: f1,f2
        !f1 = A1%norm*Gauss_1D(distance(A2%c,A1%c)*A1%w/(A1%w+A2%w),0.0,A1%w)
        !f2 = A2%norm*Gauss_1D(distance(A1%c,A2%c)*A2%w/(A1%w+A2%w),0.0,A2%w)
        are_merged = distance(A2%c,A1%c)<=A1%w+A2%w!(f1+f2)>=0.75*min(A1%norm,A2%norm)
    end function are_merged

    type(t_aglo) function new_aglo(X,W,A) ! constructor for t_aglo
    real,dimension(1:3) ::  X
	real                ::  W
    real                ::  A
        new_aglo%c = X
        new_aglo%w = W
        new_aglo%norm = A
    end function new_aglo

	subroutine print_aglo(Agl,Mesh,S)
	double precision,intent(inout)                  :: Mesh(:,:,:)
	type(t_aglo),intent(in)                         :: Agl
	type(t_simul)                                   :: S
	integer i,i_min,i_max,irel
	integer j,j_min,j_max,jrel
	integer k,k_min,k_max,krel
	real step,dist,slope
	    step = 1.0/S%grid_size(1)
	    slope = S%aglomer%slope

        i_min = 1; i_max = S%grid_size(1)
        j_min = 1; j_max = S%grid_size(2)
        k_min = 1; k_max = S%grid_size(3)
        
	    do k=k_min,k_max
	        do j=j_min,j_max
	            do i=i_min,i_max
	                dist = sqrt((Agl%c(1)-i*step)**2&
	                           +(Agl%c(2)-j*step)**2&
	                           +(Agl%c(3)-k*step)**2)
	                Mesh(i,j,k) = MAX(Agl%norm*DSIGM_1D(dist,0.0,Agl%w,slope),Mesh(i,j,k))
	            enddo
	        enddo
	    enddo
	end subroutine print_aglo
	
	! integer coordinates:
    integer function RelCoord(x1,gsize)
    integer    x1,gsize
        if ( x1 > gsize ) then
            RelCoord = x1-gsize
        else
            if ( x1 <= 0 ) then
                RelCoord = x1+gsize
            else
                RelCoord = x1
            endif
        endif
    end function RelCoord


END MODULE Mod_Aglomer
