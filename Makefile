NVCC=nvcc
MPICXX=mpicxx
SRC_PATH="./src"
BIN_PATH="./bin"

all: 
	mkdir -p ${BIN_PATH}	
	${NVCC} -arch=sm_35 -ccbin ${MPICXX} -O3 -o ${BIN_PATH}/miog_mpi ${SRC_PATH}/main.cu

run:
	echo "Running with ${NUM_NODES} nodes"
	mpirun -n ${NUM_NODES} ${BIN_PATH}/miog_mpi "data/miog.bin" "result.txt"
clean:
	rm -rf ${BIN_PATH}
	rm -f result.txt