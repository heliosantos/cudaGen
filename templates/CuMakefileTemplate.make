CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++ -arch x86_64
CC := gcc -arch x86_64
LINK := g++ -fPIC 
NVCC  := nvcc #--compiler-options -fno-strict-aliasing -m64

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

OBJS = $!FILENAME!$.cu.o
TARGET = $!FILENAME!$
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .c .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

clean:
	rm -f $(BIN) *.o *.cu_o ${TARGET}



