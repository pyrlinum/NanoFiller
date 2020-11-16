MODULE Mod_Descriptors
    implicit none

    !double precision,parameter  ::  PI = 3.14159265359D0

    TYPE :: t_aglomerate_dscr
        real                ::  ratio           ! ratio of aglomerate to dispersed volume fraction
        real                ::  avg_rad         ! average aglomerate raduis
        real                ::  rad_disp        ! dispersion of aglomerate size
        real                ::  slope           ! dispersion of aglomerate slope
        real                ::  density         ! filler density (volume fraction) in agglomerate
        integer             ::  num             ! number of aglomerates in system
        integer             ::  max_num         ! maximum number of aglomerates
    END TYPE t_aglomerate_dscr

    TYPE :: t_conductivity_dscr
        real                ::  sigma_m         ! martix conductivity
        real                ::  sigma_c         ! filler conductivity
        real                ::  power           ! critical exponent
        real                ::  perc_threshold  ! percolation threshold filler volume fraction
    END TYPE t_conductivity_dscr

    TYPE :: t_simul
        real                        ::  box_size(3)
        integer                     ::  grid_size(3)
        real                        ::  vf          ! filler particle volume fraction
        type(t_aglomerate_dscr)     ::  aglomer
        type(t_conductivity_dscr)   ::  conduct     ! conductivity law descriptor
    END TYPE t_simul

END MODULE Mod_Descriptors
