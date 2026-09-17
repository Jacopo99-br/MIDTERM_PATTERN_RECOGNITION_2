cd /content
rm -rf MIDTERM_PATTERN_RECOGNITION_2 build FaultDetectionA

# 1. Clona dentro una cartella dedicata (MIDTERM_PATTERN_RECOGNITION_2)
git clone -b rep_for_CUDA https://github.com/Jacopo99-br/MIDTERM_PATTERN_RECOGNITION_2.git MIDTERM_PATTERN_RECOGNITION_2
cd MIDTERM_PATTERN_RECOGNITION_2

# 2. Scarica ed estrai il dataset
wget -q "https://www.timeseriesclassification.com/aeon-toolkit/FaultDetectionA.zip" -O dataset.zip
unzip -q dataset.zip -d FaultDetectionA
rm dataset.zip

# 3. Installa dipendenze
apt-get update -qq && apt-get install -y -qq libomp-dev libgomp1

# 4. Compila
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ..
make -j$(nproc)

# 5. Esegui
./Project_TSPR "$@"