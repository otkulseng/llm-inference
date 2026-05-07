# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS 
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
NVCCFLAGS := -std=c++17
LDFLAGS := -lcudart
KERNEL_DIR := kernel
BUILD ?= release

ifeq ($(BUILD),debug)
  CXXFLAGS += -g -O0
else
  CXXFLAGS += -O2
endif

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TARGET := llm
INCLUDES := -I$(INC_DIR) -I. -I$(KERNEL_DIR)

# Auto-discover every .cu in kernel/. Drop a new file in there and it joins the
# build with no Makefile edits.
KERNEL_SRCS := $(wildcard $(KERNEL_DIR)/*.cu)
KERNEL_OBJS := $(patsubst $(KERNEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(KERNEL_SRCS))

SOURCES := main.cpp $(SRC_DIR)/tokenizer_bpe.cpp $(SRC_DIR)/loader.cpp $(SRC_DIR)/model.cpp
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(BUILD_DIR)/model.o $(KERNEL_OBJS)
DEPS := $(OBJECTS:.o=.d)


all: $(BIN_DIR)/$(TARGET)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)
$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/tokenizer_bpe.o: $(SRC_DIR)/tokenizer_bpe.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/loader.o: $(SRC_DIR)/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/model.o: $(SRC_DIR)/model.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f a.out

.PHONY: run

run: all
	./$(BIN_DIR)/$(TARGET)
-include $(DEPS)

# ------------------------------------------------------------
# Tests build
# All kernel/*.cu objects are pulled in automatically via $(KERNEL_OBJS).

.PHONY: tests

TEST_OBJECTS := $(BUILD_DIR)/test.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(BUILD_DIR)/model.o $(KERNEL_OBJS)

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test.o: tests/test.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_api.o: tests/test_api.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@


.PHONY: my_tests

MY_TEST_OBJECTS := $(BUILD_DIR)/my_tests.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(BUILD_DIR)/model.o $(KERNEL_OBJS)

my_tests: $(BIN_DIR)/my_tests

$(BIN_DIR)/my_tests: $(MY_TEST_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(MY_TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/my_tests.o: tests/my_tests.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
