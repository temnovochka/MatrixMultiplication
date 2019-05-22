#include <iostream>
#include <exception>
#include <vector>
#include <pthread.h>
#include <random>
#include <omp.h>
#include <functional>
#include <chrono>

template<typename T>
class Matrix {
private:
    int n;
    int m;
    std::vector<T> mat;

public:
    Matrix(int n, int m) : n(n), m(m), mat(m * n) {}

    ~Matrix() = default;

    int getN() const {
        return n;
    }

    int getM() const {
        return m;
    }

    int getMat(int i, int j) const {
        return mat[i * m + j];
    }

    void putMat(int i, int j, T res) {
        mat[i * m + j] = res;
    }

    bool compareTo(Matrix<T> z) {
        if (n != z.getN() || m != z.getM())
            return false;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (getMat(i, j) != z.getMat(i, j))
                    return false;
            }
        }
        return true;
    }

    void randInit() {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(1, 100);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                putMat(i, j, dist(mt));
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &s, const Matrix<T> &m) {
    for (int i = 0; i < m.getN(); i++) {
        for (int j = 0; j < m.getM(); j++) {
            s << m.getMat(i, j) << " ";
        }
        s << std::endl;
    }
    return s;
}

template<typename T>
void defaultMultiplication(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
    if (a.getM() != b.getN() || c.getN() != a.getN() || c.getM() != b.getM())
        throw std::invalid_argument("Size mismatch");

    for (int i = 0; i < a.getN(); i++) {
        for (int j = 0; j < b.getM(); j++) {
            T result = 0;
            for (int k = 0; k < a.getM(); k++) {
                result += a.getMat(i, k) * b.getMat(k, j);
            }
            c.putMat(i, j, result);
        }
    }
}

struct Indexes {
    int iFrom, iTo, jFrom, jTo;
};

template <typename T>
struct Arguments {
    const Matrix<T>* a = nullptr;
    const Matrix<T>* b = nullptr;
    Matrix<T>* c = nullptr;
    Indexes* ind = nullptr;
};

template<typename T>
void threadsMultiplication(const Matrix<T>* a, const Matrix<T>* b, Matrix<T>* c, Indexes* ind) {
    if (a->getM() != b->getN() || c->getN() != a->getN() || c->getM() != b->getM())
        throw std::invalid_argument("Size mismatch");

    for (int i = ind->iFrom; i < ind->iTo; i++) {
        for (int j = ind->jFrom; j < ind->jTo; j++) {
            T result = 0;
            for (int k = 0; k < a->getM(); k++) {
                result += a->getMat(i, k) * b->getMat(k, j);
            }
            c->putMat(i, j, result);
        }
    }
}

template <typename T>
void* task(void* arg) {
    auto args =  reinterpret_cast<Arguments<T>*>(arg);
    threadsMultiplication(args->a, args->b, args->c, args->ind);
    return nullptr;
}

template <typename T>
void foo(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c, int numThreads) {
    int indForI = a.getN() / numThreads;

    std::vector<pthread_t> thr(numThreads);

    std::vector<Indexes> ind(numThreads);
    for (int k = 0; k < numThreads; k++) {
        if (k == numThreads - 1) {
            ind[k] = Indexes{k*indForI, a.getN(), 0, b.getM()};
            continue;
        }
        ind[k] = Indexes{k*indForI, (k + 1)*indForI, 0, b.getM()};
    }

    std::vector<Arguments<T>> args(numThreads);
    for (int k = 0; k < numThreads; k++) {
        args[k] = Arguments<T> { &a, &b, &c,  &ind[k]};
    }

    for(int k = 0; k < numThreads; k++) {
        pthread_create(&thr[k], nullptr, task<T>, &args[k]);
    }

    for(int k = 0; k < numThreads; k++) {
        pthread_join(thr[k], nullptr);
    }
}

template<typename T>
void openMPMultiplication(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c, int numThreads) {
    if (a.getM() != b.getN() || c.getN() != a.getN() || c.getM() != b.getM())
        throw std::invalid_argument("Size mismatch");

    omp_set_num_threads(numThreads);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < a.getN(); i++) {
        for (int j = 0; j < b.getM(); j++) {
            T result = 0;
            for (int k = 0; k < a.getM(); k++) {
                result += a.getMat(i, k) * b.getMat(k, j);
            }
            c.putMat(i, j, result);
        }
    }
}

long testRun(std::function<void (Matrix<int>&, Matrix<int>&, Matrix<int>&)>& f) {
    Matrix<int> a(200, 200);
    Matrix<int> b(200, 200);
    Matrix<int> c(200, 200);

    a.randInit();
    b.randInit();

    auto start = std::chrono::steady_clock::now();
    f(a, b, c);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start);

    return duration.count();
}

void test(std::string methodName, std::function<void (Matrix<int>&, Matrix<int>&, Matrix<int>&)> f) {
    std::vector<long> time(50);

    int size = time.size();
    for (auto i = 0; i < size; i ++) {
        time[i] = testRun(f);
    }

    double avg = 0;
    for (auto& n : time)
        avg += n;
    avg /= size;

    double d = 0;
    for (auto& n : time)
        d += pow(avg - n, 2);
    d /= size == 1 ? 1 : size - 1;

    double maxError = 2.6778 * pow(d / size, 0.5);

    std::cout << methodName << " : m = " << avg << " ms, d = " << d << std::endl;
    std::cout << "99% interval: " << avg << " +- " << maxError << " ms" << std::endl << std::endl;
}


int main() {
    auto num = {1, 2, 4, 8, 16};

    test("default", [] (Matrix<int>& a, Matrix<int>& b, Matrix<int>& c) {
        defaultMultiplication(a, b, c);
    });

    for (auto& n : num)
        test("p_threads " + std::to_string(n), [n] (Matrix<int>& a, Matrix<int>& b, Matrix<int>& c) {
            foo(a, b, c, n);
        });

    for (auto& n : num)
        test("openMP " + std::to_string(n), [n] (Matrix<int>& a, Matrix<int>& b, Matrix<int>& c) {
            openMPMultiplication(a, b, c, n);
        });

    return 0;
}
