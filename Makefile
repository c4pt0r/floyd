UNAME_S := $(shell uname -s)
PYTHON  ?= python3

ifeq ($(UNAME_S),Darwin)
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
else
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
endif

METAL    ?= 0
METAL_OBJ =
DEEPSEEK_V4_GGML_OBJ =
DEEPSEEK_V4_DS4_OBJ =
FLOYD_LINK = $(CC)
LLAMA_CPP_DIR ?= .deps/llama.cpp
LLAMA_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build-floyd
LLAMA_CPP_REPO ?= https://github.com/cchuter/llama.cpp.git
LLAMA_CPP_REV ?= 19b63dc368dfef6db6783e5ba3143927b7ed1c96
LLAMA_PATCHES = patches/llama.cpp/deepseek-v4-native-mxfp4-converter.patch \
	patches/llama.cpp/deepseek-v4-context-reserve.patch
LLAMA_REV_STAMP = $(LLAMA_CPP_DIR)/.floyd-revision-$(LLAMA_CPP_REV)
LLAMA_INCLUDES = -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include
LLAMA_STATIC_LIBS = $(LLAMA_BUILD_DIR)/src/libllama.a \
	$(LLAMA_BUILD_DIR)/ggml/src/libggml.a \
	$(LLAMA_BUILD_DIR)/ggml/src/ggml-metal/libggml-metal.a \
	$(LLAMA_BUILD_DIR)/ggml/src/libggml-cpu.a \
	$(LLAMA_BUILD_DIR)/ggml/src/ggml-blas/libggml-blas.a \
	$(LLAMA_BUILD_DIR)/ggml/src/libggml-base.a
DS4_DIR ?= .deps/ds4
DS4_REPO ?= https://github.com/antirez/ds4.git
DS4_REV ?= 80ebbc396aee40eedc1d829222f3362d10fa4c6c
DS4_REV_STAMP = $(DS4_DIR)/.floyd-revision-$(DS4_REV)
DS4_CORE_OBJS = $(DS4_DIR)/ds4.o $(DS4_DIR)/ds4_distributed.o \
	$(DS4_DIR)/ds4_ssd.o $(DS4_DIR)/ds4_metal.o
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 e' supportato solo su macOS)
endif
CFLAGS  += -DFLOYD_METAL
METAL_OBJ = backend_metal.o
CFLAGS  += -DFLOYD_DEEPSEEK_V4_GGML
CFLAGS  += -DFLOYD_DEEPSEEK_V4_DS4
DEEPSEEK_V4_GGML_OBJ = deepseek_v4_ggml.o
DEEPSEEK_V4_DS4_OBJ = deepseek_v4_ds4.o
FLOYD_LINK = $(CXX)
LDFLAGS += -pthread -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
endif

TEST_BINS = tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_deepseek_v4_hc tests/test_deepseek_v4_quant tests/test_st_probe

all: floyd

DEEPSEEK_V4_CHAT_DEPS = deepseek_v4_chat.h deepseek_v4_runtime.h deepseek_v4_chat_format.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h deepseek_v4_compress.h deepseek_v4_indexer.h moe_route.h st.h json.h tok.h tok_unicode.h compat.h

floyd: floyd.o deepseek_v4_chat.o $(METAL_OBJ) $(DEEPSEEK_V4_GGML_OBJ) $(DEEPSEEK_V4_DS4_OBJ) $(if $(DEEPSEEK_V4_DS4_OBJ),$(DS4_CORE_OBJS))
	$(FLOYD_LINK) floyd.o deepseek_v4_chat.o $(METAL_OBJ) \
		$(DEEPSEEK_V4_GGML_OBJ) $(DEEPSEEK_V4_DS4_OBJ) \
		$(if $(DEEPSEEK_V4_GGML_OBJ),$(LLAMA_STATIC_LIBS)) \
		$(if $(DEEPSEEK_V4_DS4_OBJ),$(DS4_CORE_OBJS)) \
		-o floyd $(LDFLAGS)

floyd.o: floyd.c $(DEEPSEEK_V4_CHAT_DEPS) st.h json.h tok.h tok_unicode.h tok_moon.h moe_route.h compat.h
	$(CC) $(CFLAGS) -c $< -o $@

deepseek_v4_chat.o: deepseek_v4_chat.c $(DEEPSEEK_V4_CHAT_DEPS) deepseek_v4_ggml.h
	$(CC) $(CFLAGS) -c $< -o $@

$(LLAMA_REV_STAMP): $(LLAMA_PATCHES)
	@test -f "$(LLAMA_CPP_DIR)/CMakeLists.txt" || \
		git clone --filter=blob:none "$(LLAMA_CPP_REPO)" "$(LLAMA_CPP_DIR)"
	@current=$$(git -C "$(LLAMA_CPP_DIR)" rev-parse HEAD); \
		if test "$$current" != "$(LLAMA_CPP_REV)"; then \
			git -C "$(LLAMA_CPP_DIR)" fetch --depth 1 "$(LLAMA_CPP_REPO)" "$(LLAMA_CPP_REV)"; \
			git -C "$(LLAMA_CPP_DIR)" checkout --detach "$(LLAMA_CPP_REV)"; \
		fi
	@for patch in $(LLAMA_PATCHES); do \
		if git -C "$(LLAMA_CPP_DIR)" apply --reverse --check "$(CURDIR)/$$patch" 2>/dev/null; then \
			:; \
		else \
			git -C "$(LLAMA_CPP_DIR)" apply "$(CURDIR)/$$patch"; \
		fi; \
	done
	@touch "$@"

$(LLAMA_BUILD_DIR)/src/libllama.a: $(LLAMA_REV_STAMP)
	cmake -S "$(LLAMA_CPP_DIR)" -B "$(LLAMA_BUILD_DIR)" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
		-DGGML_METAL=ON -DGGML_ACCELERATE=ON -DLLAMA_CURL=OFF \
		-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
	cmake --build "$(LLAMA_BUILD_DIR)" --target llama -j 16

deepseek_v4_ggml.o: deepseek_v4_ggml.cpp deepseek_v4_ggml.h deepseek_v4_chat_format.h $(LLAMA_BUILD_DIR)/src/libllama.a
	$(CXX) -O3 -std=c++17 -DFLOYD_DEEPSEEK_V4_GGML $(LLAMA_INCLUDES) -c $< -o $@

$(DS4_REV_STAMP):
	@test -f "$(DS4_DIR)/Makefile" || \
		git clone --filter=blob:none "$(DS4_REPO)" "$(DS4_DIR)"
	@current=$$(git -C "$(DS4_DIR)" rev-parse HEAD); \
		if test "$$current" != "$(DS4_REV)"; then \
			git -C "$(DS4_DIR)" fetch --depth 1 "$(DS4_REPO)" "$(DS4_REV)"; \
			git -C "$(DS4_DIR)" checkout --detach "$(DS4_REV)"; \
		fi
	@touch "$@"

$(DS4_CORE_OBJS): $(DS4_REV_STAMP)
	$(MAKE) -C "$(DS4_DIR)" ds4.o ds4_distributed.o ds4_ssd.o ds4_metal.o

deepseek_v4_ds4.o: deepseek_v4_ds4.c deepseek_v4_ds4.h $(DS4_CORE_OBJS)
	$(CC) -O3 -std=c99 -DFLOYD_DEEPSEEK_V4_DS4 -I$(DS4_DIR) -c $< -o $@

backend_metal.o: backend_metal.m backend_metal.h kernels_metal.h
	clang -O2 -fobjc-arc -c backend_metal.m -o $@

kernels_metal.h: kernels.metal
	xxd -i kernels.metal > kernels_metal.h

metal-test: backend_metal.o tests/test_backend_metal.c backend_metal.h
	clang -O2 $(OMPC) tests/test_backend_metal.c backend_metal.o -o tests/test_backend_metal \
		-framework Metal -framework Foundation -lm $(OMPL)
	./tests/test_backend_metal

tests/test_json: tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st: tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_moe_route: tests/test_moe_route.c moe_route.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_moe_exec: tests/test_moe_exec.c moe_exec.h moe_route.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_moe_fixture: tests/test_deepseek_v4_moe_fixture.c moe_exec.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-moe: tests/test_deepseek_v4_moe_fixture
	./tests/test_deepseek_v4_moe_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_hc: tests/test_deepseek_v4_hc.c deepseek_v4_hc.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_hc_fixture: tests/test_deepseek_v4_hc_fixture.c deepseek_v4_hc.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-hc: tests/test_deepseek_v4_hc_fixture
	./tests/test_deepseek_v4_hc_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_attention_fixture: tests/test_deepseek_v4_attention_fixture.c deepseek_v4_attention.h deepseek_v4_kv_cache.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-attention: tests/test_deepseek_v4_attention_fixture
	./tests/test_deepseek_v4_attention_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_compress_fixture: tests/test_deepseek_v4_compress_fixture.c deepseek_v4_compress.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-compress: tests/test_deepseek_v4_compress_fixture
	./tests/test_deepseek_v4_compress_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_indexer_fixture: tests/test_deepseek_v4_indexer_fixture.c deepseek_v4_indexer.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-indexer: tests/test_deepseek_v4_indexer_fixture
	./tests/test_deepseek_v4_indexer_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_kv_cache_fixture: tests/test_deepseek_v4_kv_cache_fixture.c deepseek_v4_kv_cache.h deepseek_v4_attention.h deepseek_v4_indexer.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-kv-cache: tests/test_deepseek_v4_kv_cache_fixture
	./tests/test_deepseek_v4_kv_cache_fixture fixture_tiny_deepseek_v4

tests/test_deepseek_v4_quant: tests/test_deepseek_v4_quant.c deepseek_v4_quant.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

tests/test_deepseek_v4_native_quant: tests/test_deepseek_v4_native_quant.c deepseek_v4_quant.h st.h json.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

test-deepseek-v4-native-quant: tests/test_deepseek_v4_native_quant
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_deepseek_v4_native_quant "$(DSPARK)"

tests/test_deepseek_v4_native_quant_metal: tests/test_deepseek_v4_native_quant.c deepseek_v4_quant.h st.h json.h compat.h backend_metal.o
	$(CC) $(CFLAGS) -DFLOYD_METAL $< backend_metal.o -o $@ $(LDFLAGS) \
		-framework Metal -framework Foundation

test-deepseek-v4-native-quant-metal: tests/test_deepseek_v4_native_quant_metal
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_deepseek_v4_native_quant_metal "$(DSPARK)"

tests/test_deepseek_v4_model_manifest: tests/test_deepseek_v4_model_manifest.c deepseek_v4_model_manifest.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-model-manifest: tests/test_deepseek_v4_model_manifest
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_deepseek_v4_model_manifest "$(DSPARK)"

tests/test_st_probe: tests/test_st_probe.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t || exit 1; done

tests/test_tok_moon: tests/test_tok_moon.c tok_moon.h tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
test-tok: tests/test_tok_moon
	./tests/test_tok_moon models/Moonlight-16B-A3B-Instruct tok_cases.json

test-cli-default-chat: floyd
	@test -n "$(MOONLIGHT)" || (echo "set MOONLIGHT=/path/to/moonlight_i8"; exit 2)
	sh tests/test_cli_default_chat.sh "$(MOONLIGHT)"

test-deepseek-v4-naming:
	sh tests/test_deepseek_v4_naming.sh

test-prepare-deepseek-v4-gguf:
	sh tests/test_prepare_deepseek_v4_gguf.sh

tests/test_deepseek_v4_ggml: tests/test_deepseek_v4_ggml.c deepseek_v4_ggml.cpp deepseek_v4_ggml.h
	$(CXX) -O2 -std=c++17 -x c++ tests/test_deepseek_v4_ggml.c deepseek_v4_ggml.cpp -o $@

test-deepseek-v4-ggml: tests/test_deepseek_v4_ggml
	./tests/test_deepseek_v4_ggml

tests/test_deepseek_v4_ds4: tests/test_deepseek_v4_ds4.c deepseek_v4_ds4.c deepseek_v4_ds4.h
	$(CC) -O2 tests/test_deepseek_v4_ds4.c deepseek_v4_ds4.c -o $@

test-deepseek-v4-ds4: tests/test_deepseek_v4_ds4
	./tests/test_deepseek_v4_ds4

tests/test_deepseek_v4_ggml_official: tests/test_deepseek_v4_ggml_official.c deepseek_v4_ggml.cpp deepseek_v4_ggml.h $(LLAMA_BUILD_DIR)/src/libllama.a
	$(CXX) -O2 -std=c++17 -DFLOYD_DEEPSEEK_V4_GGML $(LLAMA_INCLUDES) \
		-x c++ $< deepseek_v4_ggml.cpp -x none $(LLAMA_STATIC_LIBS) -o $@ \
		-framework Accelerate -framework Metal -framework MetalKit -framework Foundation

test-deepseek-v4-ggml-official: tests/test_deepseek_v4_ggml_official
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_deepseek_v4_ggml_official "$(DSPARK)"

prepare-deepseek-v4-gguf:
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	PYTHON="$(PYTHON)" tools/prepare_deepseek_v4_gguf.sh "$(DSPARK)" \
		"$(if $(DEEPSEEK_V4_GGUF),$(DEEPSEEK_V4_GGUF),$(DSPARK)-GGUF)"

test-deepseek-v4-chat-dispatch: floyd
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	sh tests/test_deepseek_v4_chat_dispatch.sh "$(DSPARK)"

tests/test_deepseek_v4_chat_format: tests/test_deepseek_v4_chat_format.c tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

fixture_dspark_chat/chat_oracle.json: tools/make_deepseek_v4_chat_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_chat_oracle.py --model "$(DSPARK)" --output $@

fixture_dspark_chat/forward.safetensors: tools/make_deepseek_v4_chat_oracle.py tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_chat_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_chat/chat_oracle.json --forward-output $@

test-deepseek-v4-chat-format: fixture_dspark_chat/chat_oracle.json tests/test_deepseek_v4_chat_format
	./tests/test_deepseek_v4_chat_format "$(DSPARK)" fixture_dspark_chat/chat_oracle.json

test-deepseek-v4-chat: fixture_dspark_chat/forward.safetensors floyd
	PYTHON="$(PYTHON)" sh tests/test_deepseek_v4_chat_cli.sh "$(DSPARK)" $<

test-deepseek-v4-chat-metal: fixture_dspark_chat/forward.safetensors floyd
	FM_MIN_S=4 EXPECT_BACKEND=metal PYTHON="$(PYTHON)" \
		sh tests/test_deepseek_v4_chat_cli.sh "$(DSPARK)" $<

test-deepseek-v4-chat-backend: floyd
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	sh tests/test_deepseek_v4_chat_backend.sh "$(DSPARK)" cpu

test-deepseek-v4-chat-backend-metal: floyd
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	sh tests/test_deepseek_v4_chat_backend.sh "$(DSPARK)" metal-ggml

test-deepseek-v4-chat-spec: fixture_dspark_dspark_decode/oracle.safetensors floyd
	sh tests/test_deepseek_v4_chat_spec.sh "$(DSPARK)"

tools/probe_safetensors: tools/probe_safetensors.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-oracle:
	PYTHONPATH=. $(PYTHON) tests/test_make_deepseek_v4_oracle.py

test-deepseek-v4-forward-oracle:
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	PYTHONPATH=. DSPARK="$(DSPARK)" $(PYTHON) tests/test_deepseek_v4_forward.py

fixture_dspark_layer0/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" --output $@

fixture_dspark_layers_0_2/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layers-0-2-output $@

fixture_dspark_layer3_hca/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layer3-hca-output $@

fixture_dspark_layers_3_4/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layers-3-4-output $@

fixture_dspark_base_forward/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --base-forward-output $@

fixture_dspark_base_decode/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --base-decode-output $@

fixture_dspark_dspark/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --dspark-output $@

fixture_dspark_dspark_decode/oracle.safetensors: tools/make_deepseek_v4_forward_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_deepseek_v4_forward_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --dspark-decode-output $@

tests/test_deepseek_v4_forward: tests/test_deepseek_v4_forward.c deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

tests/test_deepseek_v4_decode: tests/test_deepseek_v4_decode.c deepseek_v4_runtime.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

test-deepseek-v4-decode: fixture_dspark_base_decode/oracle.safetensors tests/test_deepseek_v4_decode
	./tests/test_deepseek_v4_decode "$(DSPARK)" fixture_dspark_base_decode

tests/test_deepseek_v4_dspark_decode: tests/test_deepseek_v4_dspark_decode.c deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

test-deepseek-v4-dspark-decode: fixture_dspark_dspark_decode/oracle.safetensors tests/test_deepseek_v4_dspark_decode
	./tests/test_deepseek_v4_dspark_decode "$(DSPARK)" fixture_dspark_dspark_decode

tests/test_deepseek_v4_spec_runtime: tests/test_deepseek_v4_spec_runtime.c deepseek_v4_runtime.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(METAL_OBJ) -o $@ $(LDFLAGS)

test-deepseek-v4-spec-runtime: fixture_dspark_dspark_decode/oracle.safetensors tests/test_deepseek_v4_spec_runtime
	./tests/test_deepseek_v4_spec_runtime "$(DSPARK)" fixture_dspark_dspark_decode

test-deepseek-v4-forward: fixture_dspark_layer0/oracle.safetensors fixture_dspark_layers_0_2/oracle.safetensors fixture_dspark_layer3_hca/oracle.safetensors fixture_dspark_layers_3_4/oracle.safetensors fixture_dspark_base_forward/oracle.safetensors fixture_dspark_dspark/oracle.safetensors tests/test_deepseek_v4_forward
	./tests/test_deepseek_v4_forward "$(DSPARK)" fixture_dspark_layer0 fixture_dspark_layers_0_2 fixture_dspark_layer3_hca fixture_dspark_layers_3_4 fixture_dspark_base_forward fixture_dspark_dspark

clean:
	rm -f floyd *.o kernels_metal.h tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_deepseek_v4_moe_fixture tests/test_deepseek_v4_hc tests/test_deepseek_v4_hc_fixture tests/test_deepseek_v4_attention_fixture tests/test_deepseek_v4_compress_fixture tests/test_deepseek_v4_indexer_fixture tests/test_deepseek_v4_kv_cache_fixture tests/test_deepseek_v4_quant tests/test_deepseek_v4_native_quant tests/test_deepseek_v4_native_quant_metal tests/test_deepseek_v4_model_manifest tests/test_deepseek_v4_forward tests/test_deepseek_v4_decode tests/test_deepseek_v4_dspark_decode tests/test_deepseek_v4_spec_runtime tests/test_deepseek_v4_ggml tests/test_deepseek_v4_ggml_official tests/test_st_probe tests/test_backend_metal tests/test_tok_moon tests/test_deepseek_v4_chat_format tools/probe_safetensors

.PHONY: all floyd prepare-deepseek-v4-gguf test-c test-tok test-cli-default-chat test-deepseek-v4-naming test-deepseek-v4-chat-dispatch test-deepseek-v4-attention test-deepseek-v4-chat test-deepseek-v4-chat-metal test-deepseek-v4-chat-backend test-deepseek-v4-chat-backend-metal test-deepseek-v4-chat-format test-deepseek-v4-chat-spec test-deepseek-v4-compress test-deepseek-v4-dspark-decode test-deepseek-v4-ggml test-deepseek-v4-ggml-official test-deepseek-v4-hc test-deepseek-v4-indexer test-deepseek-v4-kv-cache test-deepseek-v4-model-manifest test-deepseek-v4-moe test-deepseek-v4-native-quant test-deepseek-v4-native-quant-metal test-deepseek-v4-oracle test-deepseek-v4-decode test-deepseek-v4-forward test-deepseek-v4-forward-oracle test-deepseek-v4-spec-runtime test-prepare-deepseek-v4-gguf metal-test clean portable
