#!/bin/bash
set -e # Termina lo script immediatamente se un comando fallisce

echo "=== 1. CLONE DEL REPOSITORY ==="
if [ ! -d ".git" ]; then
    git clone -b rep_for_CUDA https://github.com/Jacopo99-br/MIDTERM_PATTERN_RECOGNITION_2.git .
else
    echo "Repository già presente, eseguo git pull..."
    git pull origin rep_for_CUDA
fi

echo "=== 2. DOWNLOAD E ESTRAZIONE DEL DATASET ==="
if [ ! -d "FaultDetectionA" ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/FaultDetectionA.zip" -O dataset.zip
    unzip -q dataset.zip -d FaultDetectionA
    rm dataset.zip
else
    echo "Cartella FaultDetectionA già presente."
fi

echo "=== 3. INSTALLAZIONE DIPENDENZE ==="
apt-get update -qq && apt-get install -y -qq libomp-dev libgomp1

echo "=== 4. CONFIGURAZIONE E COMPILAZIONE (CMAKE & MAKE) ==="
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ..
make -j$(nproc)

echo "=== 5. ESECUZIONE DEL BENCHMARK ==="
# Se passi un argomento allo script lo usa (es: ./setup_and_run.sh --gpu-only),
# altrimenti su Colab di default imposta --gpu-only per non andare in timeout
FLAG="${1:---gpu-only}"

echo "Avvio con flag: $FLAG"
./Project_TSPR $FLAG