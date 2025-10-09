
C=4096
data='c4-10m'
D=1024
source='/mnt/hdd/yinziqi/yinziqi/large_aknn/data'

g++ -o ./index_${data} ./faiss_opq_index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

./index_${data} -d $data -s "$source/$data/"    

# g++ -o./index_${data} ./index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

# ./index_${data} -d $data -s "$source/$data/"    