#include <iostream>
#include "gpu_info.h"
#include "config.h"

int main() {
	gpu_info::print();
	std::cout << "Data folder: " << DATA_DIR << std::endl;
	return 0;
}