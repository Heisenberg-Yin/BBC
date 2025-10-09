
C=4096
data='c4-10m'
D=1024
B=1024
source='/mnt/hdd/yinziqi/yinziqi/large-heap/src/data/'

g++ -o ./bin/index_${data} ./src/index.cpp -I ./src/ -Ofast -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN

./bin/index_${data} -d $data -s "$source/$data/"    