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

TEST_BINS = tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_deepseek_v4_hc tests/test_deepseek_v4_quant tests/test_st_probe

all: floyd

DEEPSEEK_V4_CHAT_DEPS = deepseek_v4_chat.h deepseek_v4_runtime.h deepseek_v4_chat_format.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h deepseek_v4_compress.h deepseek_v4_indexer.h moe_route.h st.h json.h tok.h tok_unicode.h compat.h

floyd: floyd.c deepseek_v4_chat.c $(DEEPSEEK_V4_CHAT_DEPS) st.h json.h tok.h tok_unicode.h tok_moon.h moe_route.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) floyd.c deepseek_v4_chat.c $(METAL_OBJ) -o floyd $(LDFLAGS)

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

tests/test_deepseek_v4_quant: tests/test_deepseek_v4_quant.c deepseek_v4_quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_native_quant: tests/test_deepseek_v4_native_quant.c deepseek_v4_quant.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

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
	FM_MIN_S=4 sh tests/test_deepseek_v4_chat_backend.sh "$(DSPARK)" metal

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

tests/test_deepseek_v4_forward: tests/test_deepseek_v4_forward.c deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_decode: tests/test_deepseek_v4_decode.c deepseek_v4_runtime.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-decode: fixture_dspark_base_decode/oracle.safetensors tests/test_deepseek_v4_decode
	./tests/test_deepseek_v4_decode "$(DSPARK)" fixture_dspark_base_decode

tests/test_deepseek_v4_dspark_decode: tests/test_deepseek_v4_dspark_decode.c deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-dspark-decode: fixture_dspark_dspark_decode/oracle.safetensors tests/test_deepseek_v4_dspark_decode
	./tests/test_deepseek_v4_dspark_decode "$(DSPARK)" fixture_dspark_dspark_decode

tests/test_deepseek_v4_spec_runtime: tests/test_deepseek_v4_spec_runtime.c deepseek_v4_runtime.h deepseek_v4_forward.h deepseek_v4_quant.h deepseek_v4_hc.h deepseek_v4_kv_cache.h moe_route.h st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-deepseek-v4-spec-runtime: fixture_dspark_dspark_decode/oracle.safetensors tests/test_deepseek_v4_spec_runtime
	./tests/test_deepseek_v4_spec_runtime "$(DSPARK)" fixture_dspark_dspark_decode

test-deepseek-v4-forward: fixture_dspark_layer0/oracle.safetensors fixture_dspark_layers_0_2/oracle.safetensors fixture_dspark_layer3_hca/oracle.safetensors fixture_dspark_layers_3_4/oracle.safetensors fixture_dspark_base_forward/oracle.safetensors fixture_dspark_dspark/oracle.safetensors tests/test_deepseek_v4_forward
	./tests/test_deepseek_v4_forward "$(DSPARK)" fixture_dspark_layer0 fixture_dspark_layers_0_2 fixture_dspark_layer3_hca fixture_dspark_layers_3_4 fixture_dspark_base_forward fixture_dspark_dspark

clean:
	rm -f floyd *.o kernels_metal.h tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_deepseek_v4_moe_fixture tests/test_deepseek_v4_hc tests/test_deepseek_v4_hc_fixture tests/test_deepseek_v4_attention_fixture tests/test_deepseek_v4_compress_fixture tests/test_deepseek_v4_indexer_fixture tests/test_deepseek_v4_kv_cache_fixture tests/test_deepseek_v4_quant tests/test_deepseek_v4_native_quant tests/test_deepseek_v4_native_quant_metal tests/test_deepseek_v4_model_manifest tests/test_deepseek_v4_forward tests/test_deepseek_v4_decode tests/test_deepseek_v4_dspark_decode tests/test_deepseek_v4_spec_runtime tests/test_st_probe tests/test_backend_metal tests/test_tok_moon tests/test_deepseek_v4_chat_format tools/probe_safetensors

.PHONY: all floyd test-c test-tok test-cli-default-chat test-deepseek-v4-naming test-deepseek-v4-chat-dispatch test-deepseek-v4-attention test-deepseek-v4-chat test-deepseek-v4-chat-metal test-deepseek-v4-chat-backend test-deepseek-v4-chat-backend-metal test-deepseek-v4-chat-format test-deepseek-v4-chat-spec test-deepseek-v4-compress test-deepseek-v4-dspark-decode test-deepseek-v4-hc test-deepseek-v4-indexer test-deepseek-v4-kv-cache test-deepseek-v4-model-manifest test-deepseek-v4-moe test-deepseek-v4-native-quant test-deepseek-v4-native-quant-metal test-deepseek-v4-oracle test-deepseek-v4-decode test-deepseek-v4-forward test-deepseek-v4-forward-oracle test-deepseek-v4-spec-runtime metal-test clean portable
