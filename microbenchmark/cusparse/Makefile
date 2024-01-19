# Makefile for cuSPARSE example

# Specify the compiler
CC=nvcc

# Specify the flags
CFLAGS=-lcusparse -I${CUDA_HOME}/include

# Specify the target
TARGET=main

# Specify the source files
SRCS=main.cu

# Specify the object files
OBJS=$(SRCS:.cu=.o)

# The build target executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(CFLAGS)

# This is a suffix replacement rule for building .o's from .cu's
# It uses automatic variables $<: the name of the prerequisite of
# the rule(a .cu file) and $@: the name of the target of the rule
# (a .o file) (see the gnu make manual section about automatic variables)
%.o: %.cu
	$(CC) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)