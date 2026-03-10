# source='./data'
source='/yinziqi/'
data='marco-30m'
C=4096
B=1024
D=1024
k=100000

# export ASAN_OPTIONS=allocator_may_return_null=1
# -fsanitize=address,undefined -fno-omit-frame-pointer -g
g++ -march=core-avx2 -Ofast  -o ./bin/search_on_disk_${data} ./src/search_on_disk.cpp -I ./src/  -I ../x86-simd-sort/lib -L ../x86-simd-sort/build -L /opt/intel/oneapi/vtune/latest/lib64 -lx86simdsortcpp -laio -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN 

result_path=./results
mkdir ${result_path}

res="${result_path}/${data}/"

mkdir "$result_path/${data}/"

./bin/search_on_disk_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"
# vtune -collect hotspots -result-dir vtune_hot ./bin/search_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"