# source='./data'
source='/mnt/hdd/yinziqi/yinziqi/large-heap/src/data/'
data='c4-10m'
C=4096
B=1024
D=1024
k=500

# export ASAN_OPTIONS=allocator_may_return_null=1
# -fsanitize=address,undefined -fno-omit-frame-pointer 

# g++ -march=core-avx2 -Ofast -o ./bin/search_${data} ./src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

# g++ -march=native -Ofast -o -fno-omit-frame-pointer -g ./bin/search_${data} ./src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

g++ -march=core-avx2 -Ofast -g -o ./bin/search_${data} ./src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

result_path=./results
mkdir ${result_path}

res="${result_path}/${data}/"

mkdir "$result_path/${data}/"

# ./bin/search_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"

vtune -collect hotspots -result-dir vtune_hot ./bin/search_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"



# vtune-backend --web-port 8080 --data-directory ./vtune_hot/
