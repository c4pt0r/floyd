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

TEST_BINS = tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_v4_quant tests/test_st_probe

all: floyd

floyd: floyd.c st.h json.h tok.h tok_unicode.h moe_route.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) floyd.c $(METAL_OBJ) -o floyd $(LDFLAGS)

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

tests/test_v4_quant: tests/test_v4_quant.c v4_quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_probe: tests/test_st_probe.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t || exit 1; done

tests/test_tok_moon: tests/test_tok_moon.c tok_moon.h tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
test-tok: tests/test_tok_moon
	./tests/test_tok_moon models/Moonlight-16B-A3B-Instruct tok_cases.json

tools/probe_safetensors: tools/probe_safetensors.c st.h st_probe.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-v4-oracle:
	PYTHONPATH=. $(PYTHON) tests/test_make_v4_oracle.py

clean:
	rm -f floyd *.o kernels_metal.h tests/test_json tests/test_st tests/test_moe_route tests/test_moe_exec tests/test_v4_quant tests/test_st_probe tests/test_backend_metal tests/test_tok_moon tools/probe_safetensors

.PHONY: all test-c test-tok test-v4-oracle metal-test clean portable
