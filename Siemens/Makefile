# Makefile for SIEMENS Project

# Paths
SRC_DIR := src
COMMON_DIR := $(SRC_DIR)/common
CPU_DIR := $(SRC_DIR)/cpu
GPU_DIR := $(SRC_DIR)/gpu
BUILD_DIR := build

# Compilers
CC := gcc
CFLAGS := -I$(COMMON_DIR) -O0
OMPFLAGS := -fopenmp
NVCC := nvcc
NVFLAGS := -I$(COMMON_DIR) -O0

# Get all common object files
COMMON_SRC := $(wildcard $(COMMON_DIR)/*.c)
COMMON_OBJS := $(patsubst $(COMMON_DIR)/%.c, $(BUILD_DIR)/%.o, $(COMMON_SRC))

# Create build directory if it doesn't exist
$(shell mkdir -p $(BUILD_DIR))

# Generic build command
.PHONY: all cpu openMp gpu clean

# Build entry point: make filename type
%:
	@echo "Specify type: make <filename> <type>"
	@echo "Types: cpu | openMp | gpu"

# CPU build
cpu: FILENAME ?= main
cpu:
	$(CC) $(CFLAGS) $(COMMON_SRC) $(CPU_DIR)/$(FILENAME).c -o $(BUILD_DIR)/$(FILENAME)

# OpenMP build
openMp: FILENAME ?= main
openMp:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(COMMON_SRC) $(CPU_DIR)/$(FILENAME).c -o $(BUILD_DIR)/$(FILENAME).exe

# GPU build
gpu: FILENAME ?= main
gpu:
	$(NVCC) $(NVFLAGS) $(COMMON_SRC) $(GPU_DIR)/$(FILENAME).cu -o $(BUILD_DIR)/$(FILENAME)

# Clean
clean:
	rm -rf $(BUILD_DIR)/*
