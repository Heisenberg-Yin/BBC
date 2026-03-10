import numpy as np
import os
import struct


def get_fvecs_info(file_path):
    with open(file_path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0, 2)
        file_size = f.tell()
        n = file_size // (4 + dim * 4)
    return n, dim


def read_fvecs(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_flat_binary(filename, N, D):
    print(f"Reading flat binary from {filename}.")
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != N * D:
        raise RuntimeError(f"File size mismatch: expected {N * D}, got {data.size}")
    return data.reshape(N, D)


def verify_index_and_data():
    path = "/yinziqi/marco-30m/"
    
    # 读取原始数据
    data_path = os.path.join(path, "base.fvecs")
    N, D = get_fvecs_info(data_path)
    print(f"Original data: N={N}, D={D}")
    
    # 读取cluster_id
    cluster_id_path = os.path.join(path, "cluster_id_4096.ivecs")
    cluster_id = np.squeeze(read_ivecs(cluster_id_path))
    print(f"Cluster ID: shape={cluster_id.shape}")
    
    # 读取index文件中的id数组
    index_path = os.path.join(path, "ivfpq_marco-30m.index")
    with open(index_path, "rb") as f:
        N_idx = struct.unpack("I", f.read(4))[0]
        D_idx = struct.unpack("I", f.read(4))[0]
        C = struct.unpack("I", f.read(4))[0]
        M = struct.unpack("I", f.read(4))[0]
        centroid_per_sub = struct.unpack("I", f.read(4))[0]
        pq_code_size = struct.unpack("I", (f.read(4)))[0]
        
        print(f"Index: N={N_idx}, D={D_idx}, C={C}, M={M}, centroid_per_sub={centroid_per_sub}, pq_code_size={pq_code_size}")
        
        # 读取start数组
        start = np.fromfile(f, dtype=np.uint32, count=C)
        # 读取len数组
        len_arr = np.fromfile(f, dtype=np.uint32, count=C)
        # 读取id数组
        id_arr = np.fromfile(f, dtype=np.uint32, count=N_idx)
        
        print(f"start: {start[:10]}...")
        print(f"len: {len_arr[:10]}...")
        print(f"id: {id_arr[:10]}...")
    
    # 读取base_aligned_grouped.fvecs
    aligned_data_path = os.path.join(path, "base_aligned_grouped.fvecs")
    aligned_data = read_flat_binary(aligned_data_path, N, D)
    
    # 读取原始数据
    original_data = read_fvecs(data_path)
    
    # 验证：对于重排后的索引i，aligned_data[i]应该等于original_data[id_arr[i]]
    print("\n验证数据一致性...")
    num_check = 100
    for i in range(num_check):
        idx = i * (N // num_check)
        aligned_vec = aligned_data[idx]
        original_idx = id_arr[idx]
        original_vec = original_data[original_idx]
        
        if not np.allclose(aligned_vec, original_vec, rtol=1e-5, atol=1e-5):
            print(f"Error at idx {idx}: aligned_data[{idx}] != original_data[{original_idx}]")
            print(f"  aligned_vec[:10]: {aligned_vec[:10]}")
            print(f"  original_vec[:10]: {original_vec[:10]}")
            return False
    
    print(f"验证通过！检查了 {num_check} 个向量。")
    
    # 验证：重排后的向量应该按照簇分组
    print("\n验证簇分组...")
    current_cluster = 0
    current_cluster_start = 0
    for c in range(C):
        cluster_len = len_arr[c]
        cluster_start = start[c]
        
        # 检查这个簇的向量是否在正确的位置
        for i in range(cluster_len):
            idx = cluster_start + i
            original_idx = id_arr[idx]
            expected_cluster = cluster_id[original_idx]
            
            if expected_cluster != c:
                print(f"Error: 重排后索引 {idx} (原始索引 {original_idx}) 应该在簇 {c}，但实际在簇 {expected_cluster}")
                return False
    
    print(f"验证通过！所有向量都在正确的簇中。")
    
    return True


if __name__ == "__main__":
    verify_index_and_data()
    print("Done.")
