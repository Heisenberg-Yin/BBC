
C=4096
data='marco-30m'
D=1024
source='/yinziqi/'
K=100000

# -fno-omit-frame-pointer -g -fsanitize=address
g++ -march=core-avx2 -Ofast -o ./search_${data} ./query_on_disk.cpp -I ./ -laio

./search_${data} -d ${data} -k ${K} -s "$source/$data/"
# sudo perf record -e l1_data_cache_fills_all ./search_${data} -d ${data} -k ${K} -s "$source/$data/"

# vtune -collect hotspots -result-dir vtune_hot ./search_${data} -d ${data} -k ${K} -s "$source/$data/"
