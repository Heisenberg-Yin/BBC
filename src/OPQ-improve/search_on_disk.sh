C=4096
data='marco-30m'
D=1024
source='/yinziqi/'
K=5000

# -fsanitize=address,undefined -fno-omit-frame-pointer -g 
# g++ -march=core-avx2 -fsanitize=address,undefined -fno-omit-frame-pointer -Ofast  -o ./search_${data} ./query_on_disk.cpp -I ./ -I ../x86-simd-sort/lib -L ../x86-simd-sort/build -L /opt/intel/oneapi/vtune/latest/lib64 -lx86simdsortcpp -laio -luring

g++ -march=core-avx2  -Ofast  -o ./search_${data} ./query_on_disk.cpp -I ./ -I ../x86-simd-sort/lib -L ../x86-simd-sort/build -L /opt/intel/oneapi/vtune/latest/lib64 -lx86simdsortcpp -laio -luring


./search_${data} -d ${data} -k ${K} -s "$source/$data/"
# sudo perf record -e l1_data_cache_fills_all ./search_${data} -d ${data} -k ${K} -s "$source/$data/"

# vtune -collect hotspots -result-dir vtune_hot ./search_${data} -d ${data} -k ${K} -s "$source/$data/"