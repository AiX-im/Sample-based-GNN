rm -r ./build
mkdir ./build
cd ./build
cmake ..
make -j 12
cd ..