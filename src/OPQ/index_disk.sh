
C=4096
data='marco-30m'
D=1024
source='/yinziqi/'

g++ -o ./index_disk_${data} ./faiss_opq_disk_index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

./index_disk_${data} -d $data -s "$source/$data/"    

# g++ -o./index_${data} ./index.cpp -I ./ -Ofast -march=core-avx2 -lfaiss -fopenmp -lopenblas

# ./index_${data} -d $data -s "$source/$data/"    