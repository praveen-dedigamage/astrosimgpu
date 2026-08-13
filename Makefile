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
#   make clean       remove the directory named by BUILD
#   make distclean   remove every build directory

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2
INCLUDES  = -Iinclude
BUILD     = build

# NVHPC spells the OpenMP flag -mp and does not accept every GCC warning flag,
# so both depend on the compiler. This matters on machines where the module
# system will not let gcc and nvhpc be loaded at once: nvc++ can then build
# both the host and the offload variant, and there is no way to end up linking
# objects from two different toolchains.
ifneq (,$(findstring nvc++,$(CXX)))
  OMPFLAG   ?= -mp
  WARNFLAGS ?= -Wall
else
  OMPFLAG   ?= -fopenmp
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
  CXXFLAGS += $(OMPFLAG)
  LDFLAGS  += $(OMPFLAG)
else
  # The OpenMP pragmas are inert without the flag; do not warn about them.
  CXXFLAGS += -Wno-unknown-pragmas
endif

CORE_SRC = src/astrocyte.cpp src/neuron.cpp src/network.cpp src/recorder.cpp \
           src/analysis.cpp src/parameters.cpp src/json.cpp
CORE_OBJ = $(CORE_SRC:%.cpp=$(BUILD)/%.o)

.PHONY: all test run clean distclean

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

# clean only removes the directory named by BUILD, so a side-by-side host and
# offload build needs both named. distclean removes every build directory,
# which is what you want after changing compiler: objects from two toolchains
# link into an unhelpful pile of undefined references.
distclean:
	rm -rf build build-cpu build-gpu

-include $(CORE_OBJ:.o=.d) $(BUILD)/src/main.d $(BUILD)/tests/test_astrosimgpu.d
