
#pragma once
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>
#include <Eigen/Dense>

template <typename T>
class Matrix
{
private:
public:
    T *data;
    uint32_t n;
    uint32_t d;

    // Construction
    Matrix();                                                         // Default
    Matrix(uint32_t n, uint32_t d);                                   // Fixed size
    Matrix(const Matrix<T> &X);                                       // Deep Copy
    Matrix(uint32_t n, uint32_t d, const Matrix<T> &X);               // Fixed size with a filling matrix.
    Matrix(uint32_t n, uint32_t d, const Matrix<T> &X, uint32_t *id); // Submatrix with given row numbers.
    Matrix(const Matrix<T> &X, const Matrix<uint32_t> &ID);           // Submatrix
    Matrix(const Matrix<T> &X, const uint32_t id);                    // row
    Matrix(char *data_file_path);                                     // IO
    Matrix(uint32_t n);                                               // ID

    // Deconstruction
    ~Matrix()
    {
        delete[] data;
    }

    // Serialization
    void serialize(FILE *fp);
    void deserialize(FILE *fp);

    Matrix &operator=(const Matrix &X)
    {
        delete[] data;
        n = X.n;
        d = X.d;
        size_t total_size = static_cast<size_t>(n) * d;
        data = new T[total_size];
        memcpy(data, X.data, sizeof(T) * total_size);
        return *this;
    }

    // Linear Algebra
    void add(uint32_t a, const Matrix<T> &B, uint32_t b);
    void div(uint32_t a, T c);
    void mul(const Matrix<T> &A, Matrix<T> &result) const;
    void copy(uint32_t r, const Matrix<T> &A, uint32_t ra); // copy the ra th row of A to r th row
    float dist(const Matrix<T> &A, uint32_t a, const Matrix<T> &B, uint32_t b) const;
    float dist(uint32_t a, const Matrix<T> &B, uint32_t b) const;

    uint32_t scalar()
    {
        return data[0];
    }

    bool empty()
    {
        if (n == 0)
            return 1;
        return 0;
    }

    // Experiment and Debug
    void print();
    void reset();
};

template <typename T>
Matrix<T>::Matrix()
{
    n = 0;
    d = 0;
    data = NULL;
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &X)
{
    n = X.n;
    d = X.d;
    size_t total_size = static_cast<size_t>(n) * d;
    data = new T[total_size];
    memcpy(data, X.data, sizeof(T) * total_size);
}

template <typename T>
Matrix<T>::Matrix(uint32_t _n, uint32_t _d)
{
    n = _n;
    d = _d;
    size_t total_size = static_cast<size_t>(n) * d;
    data = new T[total_size + 10];
    memset(data, 0, (total_size + 10) * sizeof(T));
}

template <typename T>
Matrix<T>::Matrix(uint32_t _n, uint32_t _d, const Matrix<T> &X)
{
    n = _n;
    d = _d;
    size_t total_size = static_cast<size_t>(n) * d;
    data = new T[total_size + 10];
    memset(data, 0, (total_size + 10) * sizeof(T));
    for (uint32_t i = 0; i < n; i++)
    {
        // memcpy(data + i * d, X.data + i * X.d, sizeof(T) * X.d);
        memcpy(data + static_cast<size_t>(i) * d,
               X.data + static_cast<size_t>(i) * X.d,
               sizeof(T) * X.d);
    }
}

template <typename T>
Matrix<T>::Matrix(uint32_t _n, uint32_t _d, const Matrix<T> &X, uint32_t *id)
{
    n = _n;
    d = _d;
    data = new T[static_cast<size_t>(n) * d];
    for (uint32_t i = 0; i < n; i++)
    {
        // memcpy(data + i * d, X.data + id[i] * d, sizeof(T) * d);
        memcpy(data + static_cast<size_t>(i) * d, X.data + static_cast<size_t>(id[i]) * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &X, const Matrix<uint32_t> &id)
{
    n = id.n;
    d = X.d;
    data = new T[static_cast<size_t>(n) * d];
    for (uint32_t i = 0; i < n; i++)
    {
        memcpy(data + static_cast<size_t>(i) * d, X.data + static_cast<size_t>(id.data[i]) * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &X, const uint32_t id)
{
    n = 1;
    d = X.d;
    data = new T[static_cast<size_t>(n) * d];
    for (uint32_t i = 0; i < n; i++)
    {
        memcpy(data + static_cast<size_t>(i) * d, X.data + static_cast<size_t>(id) * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path)
{
    n = 0;
    d = 0;
    data = NULL;
    printf("%s\n", data_file_path);
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&d, 4);

    std::cerr << "Dimensionality - " << d << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = static_cast<size_t>(ss);
    n = (uint32_t)(fsize / (sizeof(T) * d + 4));
    // n = (uint32_t)(fsize / (d + 1) / 4);
    data = new T[static_cast<size_t>(n) * d + 10];
    std::cerr << "Cardinality - " << n << std::endl;
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * d), d * sizeof(T));
    }
    in.close();
}

template <typename T>
Matrix<T>::Matrix(uint32_t _n)
{
    n = _n;
    d = 1;
    data = new T[n];
    for (uint32_t i = 0; i < n; i++)
        data[i] = i;
}

template <typename T>
void Matrix<T>::print()
{
    for (uint32_t i = 0; i < 2; i++)
    {
        std::cout << "(";
        for (uint32_t j = 0; j < d; j++)
        {
            std::cout << data[static_cast<size_t>(i) * d + j] << (j == d - 1 ? ")" : ", ");
        }
        std::cout << std::endl;
    }
}

template <typename T>
void Matrix<T>::reset()
{
    memset(data, 0, sizeof(T) * static_cast<size_t>(n) * d);
}

template <typename T>
float Matrix<T>::dist(const Matrix<T> &A, uint32_t a, const Matrix<T> &B, uint32_t b) const
{
    float dist = 0;
    float *ptra = A.data + static_cast<size_t>(a) * d;
    float *ptrb = B.data + static_cast<size_t>(b) * d;

    for (int i = 0; i < d; i++)
    {
        float t = *ptra - *ptrb;
        dist += t * t;
        ptra++;
        ptrb++;
    }

    return dist;
}

template <typename T>
float Matrix<T>::dist(uint32_t a, const Matrix<T> &B, uint32_t b) const
{
    float dist = 0;
    size_t a_offset = static_cast<size_t>(a) * d;
    size_t b_offset = static_cast<size_t>(b) * d;
    for (uint32_t i = 0; i < d; i++)
    {
        dist += (data[a_offset + i] - B.data[b_offset + i]) * (data[a_offset + i] - B.data[b_offset + i]);
    }
    return dist;
}

template <typename T>
void Matrix<T>::add(uint32_t a, const Matrix<T> &B, uint32_t b)
{
    size_t a_offset = static_cast<size_t>(a) * d;
    size_t b_offset = static_cast<size_t>(b) * d;
    for (uint32_t i = 0; i < d; i++)
        data[a_offset + i] += B.data[b_offset + i];
}

template <typename T>
void Matrix<T>::div(uint32_t a, T c)
{
    size_t a_offset = static_cast<size_t>(a) * d;
    for (uint32_t i = 0; i < d; i++)
        data[a_offset + i] /= c;
}

template <typename T>
void Matrix<T>::mul(const Matrix<T> &A, Matrix<T> &result) const
{
    // result.reset();
    result.n = n;
    result.d = A.d;
    for (uint32_t i = 0; i < n; i++)
    {
        size_t this_row_offset = static_cast<size_t>(i) * d;
        size_t result_row_offset = static_cast<size_t>(i) * A.d;
        for (uint32_t k = 0; k < A.d; k++)
        {
            T p = 0;
            for (uint32_t j = 0; j < d; j++)
            {
                size_t A_col_offset = static_cast<size_t>(j) * A.d;
                p += data[this_row_offset + j] * A.data[A_col_offset + k];
            }
            result.data[result_row_offset + k] = p;
        }
    }
    // result = data \times A
}

template <typename T>
Matrix<T> mul(const Matrix<T> &A, const Matrix<T> &B)
{

    std::cerr << "Matrix Multiplication - " << A.n << " " << A.d << " " << B.d << std::endl;
    Eigen::MatrixXd _A(A.n, A.d);
    Eigen::MatrixXd _B(B.n, B.d);
    Eigen::MatrixXd _C(A.n, B.d);

    for (int i = 0; i < A.n; i++)
    {
        size_t row_offset = static_cast<size_t>(i) * A.d;
        for (int j = 0; j < A.d; j++)
            _A(i, j) = A.data[row_offset + j];
    }

    for (int i = 0; i < B.n; i++)
    {
        size_t row_offset = static_cast<size_t>(i) * B.d;
        for (int j = 0; j < B.d; j++)
            _B(i, j) = B.data[i * B.d + j];
    }
    _C = _A * _B;

    Matrix<T> result(A.n, B.d);

    for (int i = 0; i < A.n; i++)
    {
        size_t row_offset = static_cast<size_t>(i) * B.d;
        for (int j = 0; j < B.d; j++)
            result.data[row_offset + j] = _C(i, j);
    }
    // result = data \times A

    return result;
}

template <typename T>
void Matrix<T>::serialize(FILE *fp)
{
    fwrite(&n, sizeof(uint32_t), 1, fp);
    fwrite(&d, sizeof(uint32_t), 1, fp);
    uint32_t size = sizeof(T);
    fwrite(&size, sizeof(uint32_t), 1, fp);
    fwrite(data, size, static_cast<size_t>(n) * d, fp);
}

template <typename T>
void Matrix<T>::deserialize(FILE *fp)
{
    fread(&n, sizeof(uint32_t), 1, fp);
    fread(&d, sizeof(uint32_t), 1, fp);
    // std::cerr << n << " " << d << std::endl;
    assert(n <= 1000000000);
    assert(d <= 2000);

    uint32_t size = sizeof(T);
    fread(&size, sizeof(uint32_t), 1, fp);
    size_t total_size = static_cast<size_t>(n) * d;
    data = new T[total_size];
    fread(data, size, total_size, fp);
}

double normalize(float *x, unsigned D)
{
    Eigen::VectorXd v(D);
    for (int i = 0; i < D; i++)
        v(i) = x[i];

    double norm = v.norm();
    v.normalize();
    for (int i = 0; i < D; i++)
        x[i] = v(i);
    return norm;
}
