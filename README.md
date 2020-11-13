# NanoFiller
NanoFiller is code to populate large numbers of rod-like particles according to predefined density profile and preferred orientation distribution..
It was designed by Sergey Pyrlin during PhD study in the group of Prof. Marta Ramos.
It was used within the Marie Curie Intial Training Network CONTACT to simulate agglomerates of CNT in polymer matrics.
The code uses NVIDIA CUDA GPU and von Neuman Monte Carlo method and checks for abcence of unfisical CNT intersections.

Can output:
  - the resulting density and angle distributions
  - generated result in vtk file for visualization
  - lists of partcle electric contacts to construct matrix of Kirchhoff's equations and compute impedance of simulated sample.
  
  
