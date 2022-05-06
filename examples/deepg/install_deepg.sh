#!/bin/bash

if [ ! -f "/root/gurobi.lic" ]; then
    echo "No Gurobi Licence found."
    echo "Please follow the instructions at https://www.gurobi.com/academia/academic-program-and-licenses to create an 'Web License Service for Container Environments'."
    echo "and place the resulting gurobi.lic at in the path /root/gurobi.lic."
    echo "If you have this file on your host machine run `docker cp gurobi.lic proof:/root/gurobi.lic`."
    echo "Press [Enter] to continue"
    read
fi


git clone https://github.com/eth-sri/deepg.git
cp deepg_constraints.cpp deepg/code
cp Makefile deepg/code
cd deepg/code
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz -O gurobi9.1.2_linux64.tar.gz
tar xvfz gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cd ../../../..
cp gurobi912/linux64/include/gurobi_c++.h .
cp gurobi912/linux64/include/gurobi_c.h .
cp gurobi912/linux64/lib/libgurobi91.so .
cp gurobi912/linux64/src/build/libgurobi_c++.a .
mkdir build
make deepg_constraints
