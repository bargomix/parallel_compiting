#include <iostream>
#include <cmath>

#ifndef ARRAY_TYPE
#define ARRAY_TYPE double
#endif

const int N = 10000000; // 10^7 элементов

int main() {
    ARRAY_TYPE* arr = new ARRAY_TYPE[N];

    // Заполняем массив синусом (один период на всю длину)
    for (int i = 0; i < N; i++) {
        arr[i] = sin(2 * M_PI * i / N);
    }

    // Считаем сумму
    ARRAY_TYPE sum = 0;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    std::cout << "Тип массива: ";
    #ifdef USE_FLOAT
    std::cout << "float" << std::endl;
    #else
    std::cout << "double" << std::endl;
    #endif

    std::cout << "Количество элементов: " << N << std::endl;
    std::cout << "Сумма элементов: " << sum << std::endl;

    delete[] arr;

    return 0;
}