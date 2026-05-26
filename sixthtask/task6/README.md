# Task 6

Решение уравнения теплопроводности в двумерной области методом Якоби.

## Что реализовано

- сетки `128x128`, `256x256`, `512x512`, `1024x1024` и любые другие через `--size`;
- граничные условия линейной интерполяцией между углами `10, 20, 30, 20`;
- внутренняя область заполняется нулями;
- тип значений `double`;
- точность задается через `--eps`;
- максимальное число итераций задается через `--iters`;
- параметры командной строки через `boost::program_options`;
- OpenACC-директивы для переноса вычислений на GPU/CPU;
- вывод: размер сетки, число итераций, достигнутая ошибка и время.

Для размеров `10x10` и `13x13` программа дополнительно печатает всю сетку.

## Сборка

GPU:

```bash
cmake -S . -B build-gpu -DACC_TARGET=gpu
cmake --build build-gpu
```

CPU host:

```bash
cmake -S . -B build-host -DACC_TARGET=host
cmake --build build-host
```

CPU multicore:

```bash
cmake -S . -B build-multicore -DACC_TARGET=multicore
cmake --build build-multicore
```

## Запуск

```bash
./build-gpu/task6 --size 128 --eps 1e-6 --iters 1000000
```

## Замеры

```bash
bash benchmark.sh
```

Скрипт соберет варианты `host`, `multicore`, `gpu`, прогонит размеры `128`, `256`, `512`, `1024` и сохранит таблицу в `benchmark.csv`.

Проверка печати сетки:

```bash
./build-gpu/task6 --size 10 --eps 1e-6 --iters 1000000
./build-gpu/task6 --size 13 --eps 1e-6 --iters 1000000
```
