NVCC=nvcc

all: travelling_salesman

travelling_salesman: clean
	$(NVCC) $(NVCCFLAGS) --std c++17 -O2 -o travelling_salesman src/travelling_salesman.cu
	chmod +x travelling_salesman

debug: clean
	$(NVCC) $(NVCCFLAGS) --std c++17 -g -o travelling_salesman src/travelling_salesman.cu
	chmod +x travelling_salesman

clean:
	rm -f ./travelling_salesman
