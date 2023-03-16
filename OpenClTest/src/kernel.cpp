#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(

string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################

										
										
// equivalent to "for(uint n = 0u; n < N; n++) {", but executed in parallel
kernel void add_kernel(global float* A, global float* B, global float* C) {
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}

kernel void multiply_kernel(global float* A, global float* B) {
	const uint n = get_global_id(0);
	//A[n] = A[n] * C[n];
	//A[n] = A[n] * C[n];

	for (uint i = 0; i < 1000; ++i) {
		A[n] *= B[n];
	}
}



);} // ############################################################### end of OpenCL C code #####################################################################
