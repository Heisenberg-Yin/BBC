import numpy as np

def read_fvecs(filename):
    """
    读取 .fvecs 文件，返回 shape = (num_vectors, dim) 的 numpy 数组
    """
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype='int32')
    dim = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view('float32')

def check_inf(file_path):
    """
    检查 .fvecs 文件中有多少个向量包含 Inf
    """
    vectors = read_fvecs(file_path)
    inf_mask = np.isinf(vectors)
    vector_has_inf = np.any(inf_mask, axis=1)
    total_with_inf = np.sum(vector_has_inf)
    print(f"{file_path} 中包含 Inf 的向量数：{total_with_inf} / {len(vectors)}")
    return total_with_inf

# 调用示例
check_inf("/mnt/hdd/yinziqi/yinziqi/large-heap/src/data/biogen/base.fvecs")
check_inf("/mnt/hdd/yinziqi/yinziqi/large-heap/src/data/biogen/query.fvecs")
