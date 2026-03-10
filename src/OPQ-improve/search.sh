
C=4096
data='c4-10m'
D=1024
source='/mnt/hdd/yinziqi/yinziqi/large-heap/src/data'
K=40000

# -fsanitize=address,undefined -fno-omit-frame-pointer -g 
g++ -march=core-avx2 -Ofast -fno-omit-frame-pointer -g  -o ./search_${data} ./query.cpp -I ./

# ./search_${data} -d ${data} -k ${K} -s "$source/$data/"
sudo perf record -e l1_data_cache_fills_all ./search_${data} -d ${data} -k ${K} -s "$source/$data/"

# vtune -collect hotspots -result-dir vtune_hot ./search_${data} -d ${data} -k ${K} -s "$source/$data/"