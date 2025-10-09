
C=4096
data='deep100m'
D=96
source='/mnt/hdd/yinziqi/yinziqi/large-heap/src/data'

g++ -o ./index_${data} ./faiss_opq_index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

./index_${data} -d $data -s "$source/$data/"    

# g++ -o./index_${data} ./index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

# ./index_${data} -d $data -s "$source/$data/"    