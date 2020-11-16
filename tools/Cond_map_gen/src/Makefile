FC  = gfortran
FLAG= -g
#-nofor-main -g 
EXE = Cond_map_gen
MOD = Mod_Distribution.f90 Mod_Descriptors.f90 Mod_Aglomer.f90 Mod_IO.f90
OBJ = $(MOD:.f90=.o) $(EXE).o

all:	$(EXE)

$(EXE):	
	$(FC) $(FLAG) -o $@ $(MOD) $(EXE).f90

clean:
	rm *.mod
	rm $(EXE)

