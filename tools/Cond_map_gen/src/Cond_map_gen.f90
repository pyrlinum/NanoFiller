!     
! File:   Den_map_gen.f90
! Author: sergey
!
! Created on March 3, 2011, 9:48 AM
!
! This subprogram is for biulding probability density to be used
! for CNT populating of sample volume based on cluster distribution
!

PROGRAM Cond_map_gen

USE     Mod_Descriptors
USE     Mod_Aglomer
USE     Mod_IO


character(255)      ::  infile
! simulation main objects
type(t_simul)                           ::  simul       ! contains simulation parameters
type(t_aglo),allocatable,dimension(:)   ::  aglo_pool   ! contains aglomerates
double precision,allocatable,dimension(:,:,:)       ::  Mesh,CMesh  ! contains local volume fraction and conductivity meshes

! other vars
integer i

    call getarg(1,infile)
    simul = setup(infile)

    ! set maximum number of aglomerates as twice the average
    simul%aglomer%max_num = simul%box_size(1)*simul%box_size(2)*simul%box_size(3)&
                            *simul%vf*simul%aglomer%ratio&
                            /(4./3.*PI*simul%aglomer%avg_rad**3)*10

    allocate(aglo_pool(simul%aglomer%max_num))
    call populate(aglo_pool,simul)

    ! Creating density mesh
    allocate(Mesh(simul%grid_size(1),simul%grid_size(2),simul%grid_size(3)))
    call mesh_gen(aglo_pool,simul,Mesh)
    deallocate(aglo_pool)
    call write_mesh_data(Mesh,simul%box_size)
    call write_vtk_mesh(Mesh,simul%box_size)

    ! Convert density map to conductivity map:
    !allocate(CMesh(simul%grid_size(1),simul%grid_size(2),simul%grid_size(3)))
    !call build_conduct_map(Mesh,CMesh,simul%conduct,simul%grid_size)
    !deallocate(Mesh)
	
    !call write_mesh_data(CMesh,simul%box_size)
    !call write_vtk_mesh(CMesh,simul%box_size)
    !deallocate(CMesh)

CONTAINS

    subroutine build_conduct_map(Mesh,CMesh,cond_d,gsize)
    double precision,dimension(:,:,:),intent(in)    :: Mesh
    double precision,dimension(:,:,:),intent(out)   :: CMesh
    type(t_conductivity_dscr)                                   :: cond_d
    integer i,j,k,gsize(3)
        do k=1,gsize(3)
            do j=1,gsize(2)
                do i=1,gsize(1)
                    CMesh(i,j,k) =  conductivity(Mesh(i,j,k),cond_d)
                enddo
            enddo
        enddo
    end subroutine build_conduct_map

    double precision function conductivity(density,cond_d)
    double precision,intent(in)    :: density
    type(t_conductivity_dscr)      :: cond_d
    double precision    :: scaled_res_m,scaled_res_c
        if (density<cond_d%perc_threshold) then
            conductivity = cond_d%sigma_m
        else
            conductivity = cond_d%sigma_m+cond_d%sigma_c&
                            *((density-cond_d%perc_threshold)&
                            /(1-cond_d%perc_threshold))**cond_d%power
        endif
    end function conductivity

    subroutine mesh_gen(A_arr,simul,Mesh)
    double precision ,dimension(:,:,:),intent(inout)     ::  Mesh
    type(t_aglo),intent(in)                 ::  A_arr(*)
    type(t_simul),intent(in)                ::  simul
    integer                                 ::  m
        Mesh = 0.0D0
        do m=1,simul%aglomer%num
            call print_aglo(A_arr(m),Mesh,simul)
        enddo
    end subroutine

END PROGRAM Cond_map_gen
