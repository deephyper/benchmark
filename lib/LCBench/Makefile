
COMMIT=8de69dc0e7f7b4baee7129d599e74cc3e09b6d06

build:
	mkdir -p build
	wget https://github.com/automl/LCBench/archive/$(COMMIT).zip -O build/LCBench.zip
	unzip build/LCBench.zip -d build/
	mv build/LCBench-$(COMMIT) build/LCBench
	mkdir -p build/LCBench/data
	wget https://figshare.com/ndownloader/files/21188607 -O build/LCBench/data/data_2k.zip
	unzip build/LCBench/data/data_2k.zip -d build/LCBench/data/

clean:
	rm -rf build/