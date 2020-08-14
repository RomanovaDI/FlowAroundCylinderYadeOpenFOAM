#Deepak Kunhappan, deepak.kunhappan@3sr-grenoble.fr
#Example script of Yade-OpenFOAM coupling.
#get the OpenFOAM solver at : https://github.com/dpkn31/Yade-OpenFOAM-coupling
#get the latest version of Yade with the FoamCoupling engine here : https://gitlab.com/yade-dev/trunk
#Both binary and compiled versions of OpenFOAM-6 can be used. (currently tested for OpenFOAM-6).
#Have MPI (preferably OpenMPI) installed. Note : Both OpenFOAM and Yade has to be compiled with the same MPI version.
#----------------------------------------------------------------------------------------------------------------#
# Usage :
#
# 1. Enable the OpenFOAM coupling engine : (Requires MPI installation)
#       cmake -DCMAKE_INSTALL_PREFIX= /path/to/install  /path/to/sources -DDEBUG=0 -DCHUNKSIZE=1 -DENABLE_MPI=ON
#       make
#       make install
#
# 2. Complile or install OpenFOAM (get binary from here : https://openfoam.org/download/6-ubuntu ) (ubuntu and debian)
#    Example installation for ubuntu (careful, it may be different on your system!):
#       apt install software-properties-common
#       sudo sh -c "wget -O - http://dl.openfoam.org/gpg.key | apt-key add -"
#       add-apt-repository http://dl.openfoam.org/ubuntu
#       apt-get update
#       apt-get install openfoam6
#       . /opt/openfoam6/etc/bashrc
#    Example installation from source (careful, it may be different on your system!):
#       sudo apt install build-essential flex bison git-core cmake zlib1g-dev libboost-system-dev libboost-thread-dev libopenmpi-dev openmpi-bin gnuplot libreadline-dev libncurses-dev libxt-dev libptscotch-dev
#       git clone https://github.com/OpenFOAM/OpenFOAM-6.git
#       source OpenFOAM-6/etc/bashrc
#       cd OpenFOAM-6
#       # export WM_COMPILE_OPTION=Debug
#       ./Allwmake -j 2
#
# 3. Get the Yade-OpenFOAM solver at : https://github.com/dpkn31/Yade-OpenFOAM-coupling
#       git clone https://github.com/dpkn31/Yade-OpenFOAM-coupling.git
#
# 4. Enter the dir : Yade-OpenFOAM-coupling
#     Adapt the file icoFoamYade/icoFoamYade.C on how shear flow velocity (or other parameters) is initialized up to your linking.
#     A commented out example is in icoFoamYade/icoFoamYade.C line 59
#     Compile the solvers and the libs
#       ./Allclean
#       ./Allmake
#
# 5. Once compilation is done, you can run the solver from any dir. Return to this example file directory.
#
# 6. Create a symbolic link to Yade Install
#       ln -s /path/to/yade/install/bin/yade-exec yadeimport.py
#
# 7. Yade side :
#
#    a) In the Yade side, create the scene/simulation similar to this script :
#       (typical yade script but recast in classes. See : https://yade-dev.gitlab.io/trunk/user.html#importing-yade-in-other-python-applications)
#        At present only spheres are supported.  The coupling module is called as "FoamCoupling", see lines 127-130 on how to
#       initialize this.
#
#    b) Set the ids of spheres involved in hydrodynamic interaction:
#        sphereIDs = [b.id for b in O.bodies if type(b.shape)==Sphere]
#        fluidCoupling.setNumParticles(len(sphereIDs))
#        fluidCoupling.setIdList(sphereIDs)
#
#    c) Type of coupling :icoFoamYade is based on simple point force coupling
#      fluidCoupling.isGaussianInterp=False;
#
#
# 8. OpenFOAM side :
#
#      Set up the OpenFOAM in the usual way. (If you're an experienced OpenFOAM user, skip this)
#      I will only highlight the steps to run this example. Note that you can use any type of mesh
#      that is supported by OpenFOAM. (dynamic mesh is not currently supported).
#      The coupling is set in the solver icoFoamYade.C;
#      (nothing has to be modified there, except for velocity initialization.)
#
#       a) create the mesh
#           blockMesh
#
#       b) decompose the mesh:
#           decomposePar
#
#       c) make dir for VTK dump for yade
#           mkdir yadep
#
#       d) run the example :
#           mpiexec -n 1 python3 scriptYade.py : -n 2 icoFoamYade -parallel
#
# 9. Notes (OpenFOAM side):
#     to configure the mesh, edit :  system/blockMeshDict
#     to change the number of subdomains, edit : system/decomposeParDict
#     to change solver settings, edit:   system/controlDict
#     to change solution settings, edit : system/fvSolution (linear solver settings and choice of linear solvers)
#     to change discretization schemes, edit : system/fvSceme (for gradient and divergence calculation schemes)
#     to change fluid properties and particle density , edit : constant/transportProperties
#     to set BCs : edit the files in 0 for each field variables.
#
#
#10. Post-Processing : Paraview or ParaFOAM can be used to visualize the results, you can also use the OpenFOAM
#    utilities to postprocess the fluid side.
#
#       DISCLAIMER : The settings provided in this example are not universal, depending on your problem, you
#       should change the solver settings such as timestep, under relaxation factors, linear solver settings etc.
#----------------------------------------------------------------------------------------------------------------#


from __future__ import print_function
import sys
from yadeimport import *
from yade.wrapper import *
from yade.utils import *

initMPI()                           #Initialize the mpi environment, always required.
fluidCoupling = yade.FoamCoupling();     #Initialize the engine
fluidCoupling.getRank();            #part of Initialization.


#example of spheres in shear flow : two-way point force coupling
class simulation():

	def __init__(self):
		O.periodic = True
		O.cell.setBox(0.4,0.4,0.4)

		numspheres=1000
		young = 5e6
		density = 1000

		mat1 = FrictMat(young=young,poisson=0.5,frictionAngle=radians(15),density=density,label='spheremat')
		O.materials.append(mat1)
		mat2 = FrictMat(young=young,poisson=0.5,frictionAngle=0,density=0,label='wallmat')
		O.materials.append(mat2)

		epsilon = 1e-08
		minval = 0 + epsilon
		maxval = 0.4 - epsilon
		#wall coords, use facets for wall BC:
		v0 = Vector3(minval, minval, minval)
		v1 = Vector3(minval,minval,maxval)
		v2 = Vector3(maxval,minval,minval)
		v3 = Vector3(maxval,minval,maxval)

		v4 = Vector3(minval,maxval,minval)
		v5 = Vector3(minval,maxval,maxval)
		v6 = Vector3(maxval,maxval,minval)
		v7 = Vector3(maxval, maxval, maxval)

		lf0 = facet(vertices=[v0,v1,v2], material='wallmat')
		O.bodies.append(lf0)
		lf1 = facet(vertices=[v0,v2,v3], material='wallmat')
		O.bodies.append(lf1)

		uf0 = facet(vertices=[v4,v5,v6], material='wallmat')
		O.bodies.append(uf0)
		uf1 = facet(vertices=[v4,v6,v7], material='wallmat')
		O.bodies.append(uf1)

		ff0 = facet(vertices=[v1,v2,v6], material='wallmat')
		O.bodies.append(ff0)
		ff1 = facet(vertices=[v1,v6,v5], material='wallmat')
		O.bodies.append(ff1)

		bf0 = facet(vertices=[v0,v3,v7], material='wallmat')
		O.bodies.append(bf0)
		bf1 = facet(vertices=[v0,v7,v4], material='wallmat')
		O.bodies.append(bf1)

		cyl = Cylinder(segment=[0,0,1],radius=0.05)
		b = Body()
		b.shape = cyl
		b.mat = O.materials['wallmat']
		O.bodies.append(b)

		#spheres
		#mn, mx= Vector3(minval + epsilon, minval + epsilon, minval + epsilon), Vector3(maxval - epsilon, maxval - epsilon, maxval - epsilon)
		mn, mx= Vector3(minval + epsilon, minval + epsilon, minval + epsilon), Vector3(-0.06, maxval - epsilon, maxval - epsilon)
		sp = pack.SpherePack();
		sp.makeCloud(mn,mx,rMean=0.00075,rRelFuzz=0.10, num=numspheres)
		O.bodies.append([sphere(center,rad,material='spheremat') for center,rad in sp])
		sphereIDs = [b.id for b in O.bodies if type(b.shape)==Sphere]

		#coupling engine settings
		fluidCoupling.setNumParticles(len(sphereIDs))
		fluidCoupling.setIdList(sphereIDs)
		fluidCoupling.isGaussianInterp=False;  #use pimpleFoamYade for gaussianInterp

		# Integrator
		newton=NewtonIntegrator(damping=0.0, gravity = (0.0 ,0.0, 0.0))
		# add small damping in case of stability issues.. ~ 0.1 max, also note : If gravity is needed, set it in constant/g dir.

		#O.dynDt = False
		#O.dt=1e-4

		O.engines=[
			PyRunner(command='sim.printState(0)', iterPeriod= 1, label='outputMessage'),
			ForceResetter(),
			PyRunner(command='sim.printState(1)', iterPeriod= 1, label='outputMessage'),
			InsertionSortCollider([Bo1_Sphere_Aabb(),Bo1_Facet_Aabb()], allowBiggerThanPeriod=True),
			PyRunner(command='sim.printState(2)', iterPeriod= 1, label='outputMessage'),
			InteractionLoop(
				[Ig2_Sphere_Sphere_ScGeom(),Ig2_Facet_Sphere_ScGeom()],
				[Ip2_FrictMat_FrictMat_FrictPhys()],
				[Law2_ScGeom_FrictPhys_CundallStrack()]
			),
			PyRunner(command='sim.printState(3)', iterPeriod= 1, label='outputMessage'),
			PyRunner(command='sim.printDt()', iterPeriod= 1, label='outputMessage'),
			GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.7, label = "ts"),
			PyRunner(command='sim.printState(4)', iterPeriod= 1, label='outputMessage'),
			PyRunner(command='sim.printDt()', iterPeriod= 1, label='outputMessage'),
			fluidCoupling, #to be called after timestepper
			PyRunner(command='sim.printState(5)', iterPeriod= 1, label='outputMessage'),
			PyRunner(command='sim.printDt()', iterPeriod= 1, label='outputMessage'),
			PyRunner(command='sim.printMessage()', iterPeriod= 1, label='outputMessage'),
			PyRunner(command='sim.printState(6)', iterPeriod= 1, label='outputMessage'),
			newton,
			PyRunner(command='sim.printState(7)', iterPeriod= 1, label='outputMessage'),
			VTKRecorder(fileName='yadep/3d-vtk-',recorders=['spheres'],iterPeriod=100),
			PyRunner(command='sim.printState(8)', iterPeriod= 1, label='outputMessage')
		]

	def printMessage(self):
		print("********************************YADE-ITER = " + str(O.iter) +" **********************************")
		print("********************************YADE-TIME = " + str(O.time) +" **********************************")

	def printState(self, n):
		print('blaaaaa ' + str(n))

	def printDt(self):
		print(O.dt)

	def irun(self,num):
		O.run(num,1)


if __name__=="__main__":
	sim = simulation()
	#sim.irun(5000)
	#sim.irun(10000)
	sim.irun(10)
	fluidCoupling.killMPI()

import builtins
builtins.sim=sim
