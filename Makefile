# Fallback build for machines without CMake. CMakeLists.txt is the primary
# build; this exists so a checkout compiles with nothing but a C++17 compiler.
#
#   make            build the simulator and the tests
#   make test       build and run the tests
#   make run        build and run the short smoke configuration
#   make OPENMP=1   build with OpenMP (needs a compiler that supports it)
#   make OFFLOAD=1 CXX=nvc++ OFFLOAD_FLAGS="-mp=gpu -gpu=cc90"
#                   build the astrocyte update as a GPU target region.
#                   OFFLOAD_FLAGS carries the OpenMP flag too; see below.
#   make clean

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2
INCLUDES  = -Iinclude
BUILD     = build

# NVHPC does not accept every GCC warning flag, so the warning set depends on
# the compiler rather than being fixed.
ifneq (,$(findstring nvc++,$(CXX)))
  WARNFLAGS ?= -Wall
else
  WARNFLAGS ?= -Wall -Wextra -Wpedantic
endif

OPENMP ?= 0
# OFFLOAD=1 compiles the astrocyte update as an OpenMP target region instead
# of a host parallel loop, and is off by default: the default build has no
# device code at all.
#
# OFFLOAD_FLAGS must carry the compiler's own OpenMP *and* offload flags,
# because they differ between compilers and cannot be guessed:
#
#   nvc++ : -mp=gpu -gpu=cc90
#   g++   : -fopenmp -foffload=nvptx-none
#   clang : -fopenmp -fopenmp-targets=nvptx64
#
# -fopenmp is deliberately not added here. NVHPC spells it -mp, and passing
# both produces a command line no compiler accepts.
OFFLOAD ?= 0
ifeq ($(OFFLOAD),1)
  CXXFLAGS += -DASTROSIMGPU_OFFLOAD $(OFFLOAD_FLAGS)
  LDFLAGS  += $(OFFLOAD_FLAGS)
else ifeq ($(OPENMP),1)
  CXXFLAGS += -fopenmp
  LDFLAGS  += -fopenmp
else
  # The OpenMP pragmas are inert without the flag; do not warn about them.
  CXXFLAGS += -Wno-unknown-pragmas
endif

CORE_SRC = src/astrocyte.cpp src/neuron.cpp src/network.cpp src/recorder.cpp \
           src/analysis.cpp src/parameters.cpp src/json.cpp
CORE_OBJ = $(CORE_SRC:%.cpp=$(BUILD)/%.o)

.PHONY: all test run clean

all: $(BUILD)/astrosimgpu $(BUILD)/astrosimgpu_tests

$(BUILD)/astrosimgpu: $(CORE_OBJ) $(BUILD)/src/main.o
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD)/astrosimgpu_tests: $(CORE_OBJ) $(BUILD)/tests/test_astrosimgpu.o
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

test: $(BUILD)/astrosimgpu_tests
	./$(BUILD)/astrosimgpu_tests

run: $(BUILD)/astrosimgpu
	./$(BUILD)/astrosimgpu --config config/quick.json

clean:
	rm -rf $(BUILD)

-include $(CORE_OBJ:.o=.d) $(BUILD)/src/main.d $(BUILD)/tests/test_astrosimgpu.d
