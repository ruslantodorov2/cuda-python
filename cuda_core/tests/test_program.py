# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._program import Program, ProgramOptions
from cuda.core.experimental._module import ObjectCode, Kernel
from cuda.core.experimental._device import Device
import pytest

def test_program_with_various_options(init_cuda):
    code = "extern \"C\" __global__ void my_kernel() {}"
    
    options_list = [
        ProgramOptions(device_w=True, dopt=True, ptxas_options="-v"),
        ProgramOptions(relocatable_device_code=True, maxrregcount=32),
        ProgramOptions(ftz=True, prec_sqrt=False, prec_div=False),
        ProgramOptions(fmad=False, use_fast_math=True),
        ProgramOptions(extra_device_vectorization=True, modify_stack_limit=False),
        ProgramOptions(dlink_time_opt=True, gen_opt_lto=True),
        ProgramOptions(optix_ir=True, jump_table_density=50),
        ProgramOptions(device_stack_protector=True, define_macro="MY_MACRO"),
        ProgramOptions(undefine_macro="MY_MACRO", include_path="/usr/local/include"),
        ProgramOptions(pre_include="my_header.h", no_source_include=True),
        ProgramOptions(builtin_initializer_list=False, disable_warnings=True),
        ProgramOptions(restrict=True, device_as_default_execution_space=True),
        ProgramOptions(device_int128=True, optimization_info="inline"),
        ProgramOptions(display_error_number=False, no_display_error_number=True),
        ProgramOptions(diag_error="1234", diag_suppress="5678"),
        ProgramOptions(diag_warn="91011", brief_diagnostics=True),
        ProgramOptions(time="compile_time.csv", split_compile=4),
        ProgramOptions(fdevice_syntax_only=True, minimal=True)
    ]
    
    #TODO compile the program once the CI is set up
    for options in options_list:
        program = Program(code, "c++", options)
        assert program.backend == "nvrtc"
        program.close()
        assert program.handle is None

def test_program_init_valid_code_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.backend == "nvrtc"
    assert program.handle is not None

def test_program_init_invalid_code_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    with pytest.raises(NotImplementedError):
        Program(code, "python")

def test_program_init_invalid_code_format():
    code = 12345
    with pytest.raises(TypeError):
        Program(code, "c++")

def test_program_compile_valid_target_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    object_code = program.compile("ptx")
    kernel = object_code.get_kernel("my_kernel")
    assert isinstance(object_code, ObjectCode)
    assert isinstance(kernel, Kernel)

def test_program_compile_invalid_target_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    with pytest.raises(NotImplementedError):
        program.compile("invalid_target")

def test_program_backend_property():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.backend == "nvrtc"

def test_program_handle_property():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.handle is not None

def test_program_close():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    program.close()
    assert program.handle is None
