# Mnożenie macierzy

Wymagane: CUDA Toolkit (`nvcc`), kompilator C++17, `make` i karta NVIDIA.

Kompilacja i test poprawności:

```bash
cd src
make
./matmul test
```

Wszystkie scenariusze pomiarowe:

```bash
./matmul all
```

Pojedyncze pomiary dla macierzy 256 × 256:

```bash
./matmul seq 256    # CPU sekwencyjny
./matmul par 256 4  # CPU, 4 wątki
./matmul gpu 256    # GPU
```

20 powtórzeń, rozgrzewka 1000 ms i zapis wyników do CSV:

```bash
./matmul all --repeats 20 --warmup-ms 1000 > wyniki.csv
```
