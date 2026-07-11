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
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 e' supportato solo su macOS)
endif
CFLAGS  += -DFLOYD_METAL
METAL_OBJ = backend_metal.o
LDFLAGS += -framework Metal -framework Foundation
endif

TEST_BINS = tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_v4_hc tests/test_v4_quant tests/test_st_probe

all: floyd

floyd: floyd.c st.h json.h tok.h tok_unicode.h moe_route.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) floyd.c $(METAL_OBJ) -o floyd $(LDFLAGS)

v4_chat: v4_chat.c v4_runtime.h v4_chat_format.h v4_real_layer0.h v4_quant.h v4_hc.h v4_kv_cache.h v4_compress.h v4_indexer.h moe_route.h st.h json.h tok.h tok_unicode.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) v4_chat.c $(METAL_OBJ) -o v4_chat $(LDFLAGS)

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

tests/test_v4_moe_fixture: tests/test_v4_moe_fixture.c moe_exec.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-moe: tests/test_v4_moe_fixture
	./tests/test_v4_moe_fixture fixture_tiny_v4

tests/test_v4_hc: tests/test_v4_hc.c v4_hc.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_hc_fixture: tests/test_v4_hc_fixture.c v4_hc.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-hc: tests/test_v4_hc_fixture
	./tests/test_v4_hc_fixture fixture_tiny_v4

tests/test_v4_attention_fixture: tests/test_v4_attention_fixture.c v4_attention.h v4_kv_cache.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-attention: tests/test_v4_attention_fixture
	./tests/test_v4_attention_fixture fixture_tiny_v4

tests/test_v4_compress_fixture: tests/test_v4_compress_fixture.c v4_compress.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-compress: tests/test_v4_compress_fixture
	./tests/test_v4_compress_fixture fixture_tiny_v4

tests/test_v4_indexer_fixture: tests/test_v4_indexer_fixture.c v4_indexer.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-indexer: tests/test_v4_indexer_fixture
	./tests/test_v4_indexer_fixture fixture_tiny_v4

tests/test_v4_kv_cache_fixture: tests/test_v4_kv_cache_fixture.c v4_kv_cache.h v4_attention.h v4_indexer.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-kv-cache: tests/test_v4_kv_cache_fixture
	./tests/test_v4_kv_cache_fixture fixture_tiny_v4

tests/test_v4_quant: tests/test_v4_quant.c v4_quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_native_quant: tests/test_v4_native_quant.c v4_quant.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-native-quant: tests/test_v4_native_quant
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_v4_native_quant "$(DSPARK)"

tests/test_v4_native_quant_metal: tests/test_v4_native_quant.c v4_quant.h st.h json.h compat.h backend_metal.o
	$(CC) $(CFLAGS) -DFLOYD_METAL $< backend_metal.o -o $@ $(LDFLAGS) \
		-framework Metal -framework Foundation

test-v4-native-quant-metal: tests/test_v4_native_quant_metal
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_v4_native_quant_metal "$(DSPARK)"

tests/test_v4_model_manifest: tests/test_v4_model_manifest.c v4_model_manifest.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-model-manifest: tests/test_v4_model_manifest
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	./tests/test_v4_model_manifest "$(DSPARK)"

tests/test_st_probe: tests/test_st_probe.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t || exit 1; done

tests/test_tok_moon: tests/test_tok_moon.c tok_moon.h tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
test-tok: tests/test_tok_moon
	./tests/test_tok_moon models/Moonlight-16B-A3B-Instruct tok_cases.json

tests/test_v4_chat_format: tests/test_v4_chat_format.c tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

fixture_dspark_chat/chat_oracle.json: tools/make_v4_chat_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_chat_oracle.py --model "$(DSPARK)" --output $@

fixture_dspark_chat/forward.safetensors: tools/make_v4_chat_oracle.py tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_chat_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_chat/chat_oracle.json --forward-output $@

test-v4-chat-format: fixture_dspark_chat/chat_oracle.json tests/test_v4_chat_format
	./tests/test_v4_chat_format "$(DSPARK)" fixture_dspark_chat/chat_oracle.json

test-v4-chat: fixture_dspark_chat/forward.safetensors v4_chat
	PYTHON="$(PYTHON)" sh tests/test_v4_chat_cli.sh "$(DSPARK)" $<

test-v4-chat-backend: v4_chat
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	sh tests/test_v4_chat_backend.sh "$(DSPARK)" cpu

test-v4-chat-backend-metal: v4_chat
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	FLOYD_METAL=1 FM_MIN_S=4 sh tests/test_v4_chat_backend.sh "$(DSPARK)" metal

test-v4-chat-spec: fixture_dspark_dspark_decode/oracle.safetensors v4_chat
	sh tests/test_v4_chat_spec.sh "$(DSPARK)"

tools/probe_safetensors: tools/probe_safetensors.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-oracle:
	PYTHONPATH=. $(PYTHON) tests/test_make_v4_oracle.py

test-v4-real-layer0-oracle:
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	PYTHONPATH=. DSPARK="$(DSPARK)" $(PYTHON) tests/test_v4_real_layer0.py

fixture_dspark_layer0/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" --output $@

fixture_dspark_layers_0_2/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layers-0-2-output $@

fixture_dspark_layer3_hca/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layer3-hca-output $@

fixture_dspark_layers_3_4/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --layers-3-4-output $@

fixture_dspark_base_forward/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --base-forward-output $@

fixture_dspark_base_decode/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --base-decode-output $@

fixture_dspark_dspark/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --dspark-output $@

fixture_dspark_dspark_decode/oracle.safetensors: tools/make_v4_real_layer0_oracle.py
	@test -n "$(DSPARK)" || (echo "set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark"; exit 2)
	$(PYTHON) tools/make_v4_real_layer0_oracle.py --model "$(DSPARK)" \
	  --output fixture_dspark_layer0/oracle.safetensors --dspark-decode-output $@

tests/test_v4_real_layer0: tests/test_v4_real_layer0.c v4_real_layer0.h v4_quant.h v4_hc.h v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_real_decode: tests/test_v4_real_decode.c v4_runtime.h v4_real_layer0.h v4_quant.h v4_hc.h v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-real-decode: fixture_dspark_base_decode/oracle.safetensors tests/test_v4_real_decode
	./tests/test_v4_real_decode "$(DSPARK)" fixture_dspark_base_decode

tests/test_v4_dspark_decode: tests/test_v4_dspark_decode.c v4_real_layer0.h v4_quant.h v4_hc.h v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-dspark-decode: fixture_dspark_dspark_decode/oracle.safetensors tests/test_v4_dspark_decode
	./tests/test_v4_dspark_decode "$(DSPARK)" fixture_dspark_dspark_decode

tests/test_v4_spec_runtime: tests/test_v4_spec_runtime.c v4_runtime.h v4_real_layer0.h v4_quant.h v4_hc.h v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-spec-runtime: fixture_dspark_dspark_decode/oracle.safetensors tests/test_v4_spec_runtime
	./tests/test_v4_spec_runtime "$(DSPARK)" fixture_dspark_dspark_decode

test-v4-real-layer0: fixture_dspark_layer0/oracle.safetensors fixture_dspark_layers_0_2/oracle.safetensors fixture_dspark_layer3_hca/oracle.safetensors fixture_dspark_layers_3_4/oracle.safetensors fixture_dspark_base_forward/oracle.safetensors fixture_dspark_dspark/oracle.safetensors tests/test_v4_real_layer0
	./tests/test_v4_real_layer0 "$(DSPARK)" fixture_dspark_layer0 fixture_dspark_layers_0_2 fixture_dspark_layer3_hca fixture_dspark_layers_3_4 fixture_dspark_base_forward fixture_dspark_dspark

clean:
	rm -f floyd v4_chat *.o kernels_metal.h tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_v4_moe_fixture tests/test_v4_hc tests/test_v4_hc_fixture tests/test_v4_attention_fixture tests/test_v4_compress_fixture tests/test_v4_indexer_fixture tests/test_v4_kv_cache_fixture tests/test_v4_quant tests/test_v4_native_quant tests/test_v4_native_quant_metal tests/test_v4_model_manifest tests/test_v4_real_layer0 tests/test_v4_real_decode tests/test_v4_dspark_decode tests/test_v4_spec_runtime tests/test_st_probe tests/test_backend_metal tests/test_tok_moon tests/test_v4_chat_format tools/probe_safetensors

.PHONY: all test-c test-tok test-v4-attention test-v4-chat test-v4-chat-backend test-v4-chat-backend-metal test-v4-chat-format test-v4-chat-spec test-v4-compress test-v4-dspark-decode test-v4-hc test-v4-indexer test-v4-kv-cache test-v4-model-manifest test-v4-moe test-v4-native-quant test-v4-native-quant-metal test-v4-oracle test-v4-real-decode test-v4-real-layer0 test-v4-real-layer0-oracle test-v4-spec-runtime metal-test clean portable
