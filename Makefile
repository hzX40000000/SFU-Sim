# SFU-Sim build and verification entry points (Linux). Run `make help` for the target list.
#
# The MUFU model is linked from lib/libmufu_model.a and is not rebuilt by these targets.
# ARCH selects the GPU target (sm_80 A100, sm_89 RTX 4060, sm_120 RTX PRO 5000); sm_120 adds the BF16 paths.

# This Makefile targets Linux. lib/libmufu_model.a is an ELF archive, so a native Windows toolchain
# cannot link it (you would get a wall of "undefined reference" errors). Use lib/mufu_model.lib with
# test/gpu/run_all.ps1 instead, or run the Linux flow under WSL2 - see docs/windows.md.
UNAME_S := $(shell uname -s 2>/dev/null)
ifneq (,$(or $(findstring MINGW,$(UNAME_S)),$(findstring MSYS,$(UNAME_S)),$(findstring CYGWIN,$(UNAME_S))))
ifneq ($(MAKECMDGOALS),help)
$(warning This looks like a native Windows/MSYS shell: lib/libmufu_model.a is an ELF archive and cannot be linked here. Use test/gpu/run_all.ps1 with lib/mufu_model.lib, or WSL2 - see docs/windows.md.)
endif
endif

NVCC     ?= nvcc
CXX      ?= g++
AR       ?= ar
ARCH     ?= sm_80
SM       := $(patsubst sm_%,%,$(ARCH))
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fno-fast-math -ffp-contract=off
INC      := -I. -Iinclude
LIB      := lib/libmufu_model.a
LINK     := -Llib -lmufu_model
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -DSFU_TARGET_SM=$(SM) $(INC)
ifeq ($(ARCH),sm_120)
GPU_FEATURE_FLAGS := -DSFU_ENABLE_BF16=1
endif

B := build
HDR := $(wildcard ptx/*.hpp include/*.h include/*.hpp)
OFFLINE_TSV ?= test/offline/a100_sm80.tsv

.PHONY: all help test selftest offline eval eval-sm120 lib \
        gpu-minimal-smoke gpu-minimal gpu-negative gpu-extended-smoke gpu-extended gpu-f16x2-smoke gpu-f16x2 \
        gpu-round-smoke gpu-round gpu-all clean

all: selftest offline

help:
	@echo "SFU-Sim build and verification targets"
	@echo "  make test                      selftest + offline (no GPU needed)"
	@echo "  make selftest | offline        model checks"
	@echo "  make eval                      build/sfu_eval <ptx-path> <hex-input>"
	@echo "  make gpu-all TAG=<name>               all GPU checks; ARCH auto-detected if omitted"
	@echo "  make gpu-minimal-smoke         quick GPU sanity check of the 12 minimal paths (ARCH=sm_XX)"
	@echo "  make gpu-minimal | gpu-extended | gpu-f16x2 | gpu-round   single exhaustive GPU check (ARCH=sm_XX)"
	@echo "  environment: SFU_THREADS=<n> CPU threads used for the model (default: all cores)"
	@echo "  make lib                       check that lib/libmufu_model.a is present"
	@echo "  make clean                     remove build/"

# CPU-only checks (what CI runs)
test: selftest offline

# ------------------------------------------------------------------ model library
# Targets use the published archive as-is and only check that it is present.
lib: $(LIB)

$(LIB):
	@test -f $@ || { echo "ERROR: $@ is missing. Re-clone the repository to restore it."; exit 1; }


# ------------------------------------------------------------------ model checks (no GPU)
selftest: $(B)/model_selftest
	./$(B)/model_selftest

offline: $(B)/offline_verify
	./$(B)/offline_verify $(OFFLINE_TSV)

eval: $(B)/sfu_eval
eval-sm120: $(B)/sfu_eval_sm120

$(B)/model_selftest: test/unit/model_selftest.cpp $(HDR) $(LIB) | $(B)
	$(CXX) $(CXXFLAGS) $(INC) $< $(LINK) -o $@
$(B)/offline_verify: test/offline/offline_verify.cpp $(HDR) $(LIB) | $(B)
	$(CXX) $(CXXFLAGS) $(INC) $< $(LINK) -o $@
$(B)/sfu_eval: tools/sfu_eval.cpp $(HDR) $(LIB) | $(B)
	$(CXX) $(CXXFLAGS) $(INC) $< $(LINK) -o $@
$(B)/sfu_eval_sm120: tools/sfu_eval.cpp $(HDR) $(LIB) | $(B)
	$(CXX) $(CXXFLAGS) -DSFU_ENABLE_BF16=1 $(INC) $< $(LINK) -o $@

# ------------------------------------------------------------------ GPU hardware vs model
GPU_MIN   := $(B)/gpu_verify_$(ARCH)
GPU_EXT   := $(B)/gpu_verify_extended_f32_$(ARCH)
GPU_F16X2 := $(B)/gpu_verify_f16x2_$(ARCH)
GPU_ROUND := $(B)/gpu_verify_round_f32_$(ARCH)

gpu-minimal-smoke: $(GPU_MIN)
	./$(GPU_MIN) smoke
gpu-minimal: $(GPU_MIN)
	./$(GPU_MIN) full
gpu-negative: $(GPU_MIN)
	./$(GPU_MIN) negative
gpu-extended-smoke: $(GPU_EXT)
	./$(GPU_EXT) smoke
gpu-extended: $(GPU_EXT)
	./$(GPU_EXT) full
gpu-f16x2-smoke: $(GPU_F16X2)
	./$(GPU_F16X2) smoke
gpu-f16x2: $(GPU_F16X2)
	./$(GPU_F16X2) full
gpu-round-smoke: $(GPU_ROUND)
	./$(GPU_ROUND) smoke $(PATH_NAME)
gpu-round: $(GPU_ROUND)
	./$(GPU_ROUND) full $(PATH_NAME)

# One GPU, all steps, logs to build/results/$(TAG)_$(ARCH)/.
# If ARCH was not given on the command line, pass an empty argument so run_all.sh detects the
# compute capability of the installed GPU. Defaulting everything to sm_80 would reject most cards.
GPU_ARCH_ARG := $(if $(filter command line,$(origin ARCH)),$(ARCH),)
gpu-all:
	@test -n "$(TAG)" || { echo "usage: make gpu-all TAG=<name> [ARCH=sm_XX]"; exit 2; }
	bash test/gpu/run_all.sh $(TAG) $(GPU_ARCH_ARG)

$(GPU_MIN): test/gpu/gpu_verify.cu test/gpu/gpu_common.hpp $(HDR) $(LIB) | $(B)
	$(NVCC) $(NVFLAGS) $(GPU_FEATURE_FLAGS) $< -o $@ $(LINK) -lpthread
$(GPU_EXT): test/gpu/gpu_verify_extended_f32.cu test/gpu/gpu_common.hpp $(HDR) $(LIB) | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ $(LINK) -lpthread
$(GPU_F16X2): test/gpu/gpu_verify_f16x2.cu test/gpu/gpu_common.hpp $(HDR) $(LIB) | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ $(LINK) -lpthread
$(GPU_ROUND): test/gpu/gpu_verify_round_f32.cu test/gpu/gpu_common.hpp $(HDR) $(LIB) | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ $(LINK) -lpthread

# ------------------------------------------------------------------
$(B):
	@mkdir -p $(B)

clean:
	rm -rf $(B)
