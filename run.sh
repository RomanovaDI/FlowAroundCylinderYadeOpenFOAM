#! /bin/bash

#ln -s /path/to/yade/install/bin/yade-exec yadeimport.py
source /opt/openfoam6/etc/bashrc
blockMesh
decomposePar
mpiexec -n 1 python scriptYade.py : -n 2 icoFoamYade -parallel
