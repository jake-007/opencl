// OpenClTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "opencl.hpp"
#include <chrono>
#include <algorithm>
#include <vector>
using namespace std::chrono;

int run_on_cpu() {
	const uint n = 1e8; // size of vectors
	std::vector<float> a(n), b(n);

	std::fill(a.begin(), a.end(), 2.0f);
	std::fill(b.begin(), b.end(), 1.01f);

	printf("Value before CPU execution: a[n-1] = %f\n", a[n-1]);
	auto start = high_resolution_clock::now();
	for (uint i = 0; i < 1000; ++i) {
		for (uint x = 0; x < n; ++x) {
			a[x] *= b[x];
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	printf("Value after CPU execution: a[n-1] = %f\n", a[n-1]);
	printf("Duration: %lld seconds\n", duration.count());
	return 0;
}

int run() {
	Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	const uint n = 1e8; // size of vectors
	Memory<float> A(device, n); // allocate memory on both host and device
	Memory<float> B(device, n);

	Kernel multiply_kernel(device, n, "multiply_kernel", A, B); // kernel that runs on the device

	for (uint i = 0u; i < n; i++) {
		A[i] = 2.0f; // initialize memory
		B[i] = 1.01f;
	}

	print_info("Value before kernel execution: A[n-1] = " + to_string(A[n-1]));

	A.write_to_device(); // copy data from host memory to device memory
	B.write_to_device();
	auto start = high_resolution_clock::now();

	multiply_kernel.run(); // run add_kernel on the device

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	A.read_from_device(); // copy data from device memory to host memory

	print_info("Value after kernel execution: A[n-1] = " + to_string(A[n-1]));
	printf("Duration: %lld seconds\n", duration.count());

	wait();
	return 0;
}

int main() {
	run_on_cpu();
	printf("---------------------------------\n");
	run();
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
