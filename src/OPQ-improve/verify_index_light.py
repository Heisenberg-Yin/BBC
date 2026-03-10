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


def verify_index_structure():
    path = "/yinziqi/marco-30m/"
    
    # 读取原始数据信息
    data_path = os.path.join(path, "base.fvecs")
    N, D = get_fvecs_info(data_path)
    print(f"Original data: N={N}, D={D}")
    
    # 读取cluster_id
    cluster_id_path = os.path.join(path, "cluster_id_4096.ivecs")
    cluster_id = np.squeeze(read_ivecs(cluster_id_path))
    print(f"Cluster ID: shape={cluster_id.shape}")
    
    # 读取index文件中的id数组（只读取前几个）
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
        # 读取id数组（只读取前1000个）
        id_arr = np.fromfile(f, dtype=np.uint32, count=min(1000, N_idx))
        
        print(f"start: {start[:10]}...")
        print(f"len: {len_arr[:10]}...")
        print(f"id: {id_arr[:10]}...")
    
    # 验证：对于重排后的索引i，id_arr[i]应该等于原始索引
    print("\n验证id数组结构...")
    for i in range(min(100, len(id_arr))):
        print(f"重排后索引 {i} -> 原始索引 {id_arr[i]}")
    
    # 验证簇分组
    print("\n验证簇分组（前几个簇）...")
    for c in range(min(5, C)):
        cluster_len = len_arr[c]
        cluster_start = start[c]
        print(f"簇 {c}: start={cluster_start}, len={cluster_len}")
        
        # 检查这个簇的前几个向量
        for i in range(min(3, cluster_len)):
            idx = cluster_start + i
            if idx < len(id_arr):
                original_idx = id_arr[idx]
                expected_cluster = cluster_id[original_idx]
                print(f"  重排后索引 {idx} (原始索引 {original_idx}) -> 簇 {expected_cluster}")
                
                if expected_cluster != c:
                    print(f"  ERROR: 应该在簇 {c}，但实际在簇 {expected_cluster}")
                    return False
    
    print("\n验证通过！")
    return True


def verify_data_consistency():
    path = "/yinziqi/marco-30m/"
    
    # 读取原始数据信息
    data_path = os.path.join(path, "base.fvecs")
    N, D = get_fvecs_info(data_path)
    print(f"Original data: N={N}, D={D}")
    
    # 读取index文件中的id数组（只读取前几个）
    index_path = os.path.join(path, "ivfpq_marco-30m.index")
    with open(index_path, "rb") as f:
        N_idx = struct.unpack("I", f.read(4))[0]
        D_idx = struct.unpack("I", f.read(4))[0]
        C = struct.unpack("I", f.read(4))[0]
        M = struct.unpack("I", f.read(4))[0]
        centroid_per_sub = struct.unpack("I", f.read(4))[0]
        pq_code_size = struct.unpack("I", (f.read(4)))[0]
        
        # 跳过start和len数组
        f.seek(C * 4, 1)  # start数组
        f.seek(C * 4, 1)  # len数组
        
        # 读取id数组（只读取前100个）
        id_arr = np.fromfile(f, dtype=np.uint32, count=min(100, N_idx))
    
    # 读取base_aligned_grouped.fvecs的前几个向量
    aligned_data_path = os.path.join(path, "base_aligned_grouped.fvecs")
    with open(aligned_data_path, "rb") as f_aligned:
        # 读取前几个向量
        num_vectors = min(10, N)
        aligned_data = np.fromfile(f_aligned, dtype=np.float32, count=num_vectors * D)
        aligned_data = aligned_data.reshape(num_vectors, D)
    
    # 读取原始数据的前几个向量
    with open(data_path, "rb") as f_original:
        # 跳过维度信息
        dim = struct.unpack("i", f_original.read(4))[0]
        
        # 读取前几个向量
        original_data = []
        for i in range(num_vectors):
            # 跳过维度信息
            f_original.seek(i * (4 + D * 4) + 4)
            vec = np.fromfile(f_original, dtype=np.float32, count=D)
            original_data.append(vec)
        original_data = np.array(original_data)
    
    # 验证：对于重排后的索引i，aligned_data[i]应该等于original_data[id_arr[i]]
    print("\n验证数据一致性...")
    for i in range(min(5, num_vectors)):
        aligned_vec = aligned_data[i]
        if i < len(id_arr):
            original_idx = id_arr[i]
            if original_idx < len(original_data):
                original_vec = original_data[original_idx]
                
                if not np.allclose(aligned_vec, original_vec, rtol=1e-5, atol=1e-5):
                    print(f"Error at idx {i}: aligned_data[{i}] != original_data[{original_idx}]")
                    print(f"  aligned_vec[:5]: {aligned_vec[:5]}")
                    print(f"  original_vec[:5]: {original_vec[:5]}")
                    return False
                else:
                    print(f"  ✓ 重排后索引 {i} (原始索引 {original_idx}) 数据一致")
    
    print("数据一致性验证通过！")
    return True


if __name__ == "__main__":
    print("=== 验证索引结构 ===")
    verify_index_structure()
    
    print("\n=== 验证数据一致性 ===")
    verify_data_consistency()
    
    print("\nDone.")