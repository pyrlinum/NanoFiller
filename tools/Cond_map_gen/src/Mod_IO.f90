!     
! File:   Mod_IO.f90
! Author: sergey
!
! Created on March 3, 2011, 4:56 PM
!
! Contains subroutines to setup simulation from input file and output data
! and additional subroutines to be used for relative position determination
!


MODULE Mod_IO

USE Mod_Descriptors

CONTAINS

    type(t_simul) function setup(infile)
        character(255)     ::  infile      ! input file name
        logical            ::  f_exist     ! true if file exists
		
        inquire(file=infile,exist=f_exist)
        if (f_exist) then
            write(6,*) "Reading input from ",trim(adjustl(infile))
            open(1,file=infile)
            read(1,*)                                   ! BOX:
            read(1,*)   setup%box_size                  ! box size (micron)
            read(1,*)   setup%grid_size                 ! grid size
			read(1,*)   setup%vf                        ! filler volume fraction
			            setup%vf=setup%vf/100
            read(1,*)                                   ! AGLOMERATE DESCRIPTION:
			read(1,*)   setup%aglomer%ratio             ! ratio of aglomerate to dispersed volume fraction
			            setup%aglomer%ratio = setup%aglomer%ratio/100
			read(1,*)   setup%aglomer%avg_rad           ! average aglomerate raduis
			read(1,*)   setup%aglomer%rad_disp          ! dispersion of aglomerate size
			read(1,*)   setup%aglomer%slope           ! agglomerate slope
			read(1,*)   setup%aglomer%density           ! filler density (volume fraction) in agglomerate
			            setup%aglomer%num = 0           ! number of aglomerates in system
			read(1,*)								    ! CONDUCTIVITY DESCRIPTION
			read(1,*)   setup%conduct%sigma_m           ! martix conductivity
            read(1,*)   setup%conduct%sigma_c           ! filler conductivity
            read(1,*)   setup%conduct%power             ! critical exponent
            read(1,*)   setup%conduct%perc_threshold    ! percolation threshold filler volume fraction (%)
                        setup%conduct%perc_threshold = setup%conduct%perc_threshold/100
            close(1)
        else
            write(6,*) "No setup file found!"
        end if
        write(6,*) "Setup finished!"
    end function setup

    ! writing mesh dat-file
    subroutine write_mesh_data(mesh,pdim)
        double precision,dimension(:,:,:),intent(in)        ::  mesh
        real                                    ::  pdim(3)
        integer                                 ::  dm(3)
        write(6,*)  'Writing Data-file mesh.dat'
        open(3,file='mesh.dat',status = 'REPLACE')
        dm = (/ size(mesh,1),size(mesh,2),size(mesh,3) /)
        write(3,*) pdim
        write(3,*) dm
        write(3,*) mesh
        close(3)

    end subroutine write_mesh_data

    ! writing mesh vtk-file
    subroutine write_vtk_mesh(mesh,pdim)
        double precision,dimension(:,:,:),intent(in)        ::  mesh
        real,intent(in)                         ::  pdim(3)
        double precision,allocatable,dimension(:,:,:)       ::  mesh_p

        call mesh_append(mesh,mesh_p)

        write(6,*)  'Writing VTK-file mesh.vtk'
        open(3,file='mesh.vtk',status='REPLACE')
        write(3,'(A)') '# vtk DataFile Version 2.0'
        write(3,'(A)') 'Probability distribution to be used for CNT population'
        write(3,'(A)') 'ASCII'
        write(3,'(A)') 'DATASET STRUCTURED_POINTS'
        write(3,'(A,3I10)')   'DIMENSIONS ',size(mesh_p,1),size(mesh_p,2),size(mesh_p,3)
        write(3,'(A,3I10)')   'ORIGIN     ',0,0,0
        write(3,'(A,3F10.7)') 'SPACING    ',pdim(1)/(size(mesh_p,1)-1),pdim(2)/(size(mesh_p,2)-1),pdim(2)/(size(mesh_p,3)-1)
        write(3,'(A,3I10)')   'POINT_DATA ',size(mesh_p,1)*size(mesh_p,2)*size(mesh_p,3)
        write(3,'(A)') 'SCALARS probability float 1'
        write(3,'(A)') 'LOOKUP_TABLE default'
        write(3,'(6E15.6)') mesh_p
        close(3)
        deallocate(mesh_p)
    end subroutine write_vtk_mesh

    !-------------------------------------------------------------
    !                   AUXILARY SUBROUTINES:
    !-------------------------------------------------------------

    
    subroutine mesh_append(mesh_in,mesh_out)
    ! appending mesh for intuitive drawing
    double precision,dimension(:,:,:),intent(in)                    ::  mesh_in
    double precision,allocatable,dimension(:,:,:),intent(out)       ::  mesh_out
    integer,dimension(3)                                ::  d

        d = (/ size(mesh_in,1),size(mesh_in,2),size(mesh_in,3) /)
        allocate(mesh_out(d(1)+1,d(2)+1,d(3)+1))
        mesh_out = 100.
        mesh_out(  1:d(1),   1:d(2), 1:d(3))      = mesh_in
        mesh_out(  d(1)+1,   1:d(2), 1:d(3))      = mesh_out(1       , 1:d(2)  , 1:d(3))
        mesh_out(1:d(1)+1,   d(2)+1, 1:d(3))      = mesh_out(1:d(1)+1, 1       , 1:d(3))
        mesh_out(1:d(1)+1, 1:d(2)+1, d(3)+1)      = mesh_out(1:d(1)+1, 1:d(2)+1, 1     )

    end subroutine mesh_append


END MODULE Mod_IO
