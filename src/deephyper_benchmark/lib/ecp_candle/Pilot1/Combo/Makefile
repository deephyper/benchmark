
VERSION=0.5.1

build:
	mkdir -p build
	wget https://github.com/ECP-CANDLE/Benchmarks/archive/refs/tags/v$(VERSION).tar.gz -O build/ecp-candle-benchmarks.tar.gz
	tar -xzvf build/ecp-candle-benchmarks.tar.gz -C build
	mv build/Benchmarks-$(VERSION) build/Benchmarks

clean:
	rm -rf build/