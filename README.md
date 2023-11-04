# Run

mkdir build
cd build
cmake ..
make -j
./main --benchmark

# Benchmark

Sur notre version on a benchmark sur une rtx 2070 et un i9 intel, cependant le resultat n'est pas deterministe, la pluspart du temps le gpu avait de meilleur performance que le cpu, et parfois le cpu avait presque les mêmes prefromances, il se peut que le intel i9 soit puissant et égal la rtx2070, sur des quelques millions de pixels. Il faudrait tester avec des gpus plus performants aussi. 
