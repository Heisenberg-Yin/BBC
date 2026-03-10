./qadc-project/build/indexdb_create1 4096 ../../../large_aknn/data/c4-10m/base.fvecs \
    ./data/c4.4096.empty.index.db ./data/c4.4096.residuals.fvecs ../../../large_aknn/data/c4-10m/centroid_4096.fvecs

python index.py --dataset c4-10m

python ./qadc-project/quick-adc/convert-quantizer.py opq ./data/c4-10m/opq_256_4.pkl ./data/c4-10m/256x4.opq.data

./qadc-project/build/indexdb_create2 ./data/c4.4096.empty.index.db ./data/c4-10m/256x4.opq.data ./data/c4-10m/4096.256x4.opq.index.db

./qadc-project/build/db_add ./data/c4-10m/4096.256x4.opq.index.db ../../../large_aknn/data/c4-10m/base.fvecs

./qadc-project/build/db_query_4 -r100 -m100 -k500 -b1 \
    ./data/c4-10m/4096.256x4.opq.index.db \
    ../../../large_aknn/data/c4-10m/query.fvecs \
    ../../../large_aknn/data/c4-10m/top100_results.ivecs


