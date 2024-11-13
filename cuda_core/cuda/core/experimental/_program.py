# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda import nvrtc
from cuda.core.experimental._utils import handle_return
from cuda.core.experimental._module import ObjectCode

from typing import Optional, Tuple, Union
from dataclasses import dataclass
    
@dataclass
class ProgramOptions:
    gpu_architecture: Optional[str] = None  # /**< --gpu-architecture=<arch> (-arch) Specify the name of the class of GPU architectures for which the input must be compiled. Default: compute_52 */
    device_c: Optional[bool] = None  # /**< --device-c (-dc) Generate relocatable code that can be linked with other relocatable device code. Equivalent to --relocatable-device-code=true. */
    device_w: Optional[bool] = None  # /**< --device-w (-dw) Generate non-relocatable code. Equivalent to --relocatable-device-code=false. */
    relocatable_device_code: Optional[bool] = None  # /**< --relocatable-device-code={true|false} (-rdc) Enable (disable) the generation of relocatable device code. Default: false */
    extensible_whole_program: Optional[bool] = None  # /**< --extensible-whole-program (-ewp) Do extensible whole program compilation of device code. Default: false */
    device_debug: Optional[bool] = None  # /**< --device-debug (-G) Generate debug information. If --dopt is not specified, then turns off all optimizations. */
    generate_line_info: Optional[bool] = None  # /**< --generate-line-info (-lineinfo) Generate line-number information. */
    dopt: Optional[bool] = None  # /**< --dopt on (-dopt) Enable device code optimization. When specified along with ‘-G’, enables limited debug information generation for optimized device code. */
    ptxas_options: Optional[str] = None  # /**< --ptxas-options <options> (-Xptxas) Specify options directly to ptxas, the PTX optimizing assembler. */
    maxrregcount: Optional[int] = None  # /**< --maxrregcount=<N> (-maxrregcount) Specify the maximum amount of registers that GPU functions can use. */
    ftz: Optional[bool] = None  # /**< --ftz={true|false} (-ftz) When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal values. Default: false */
    prec_sqrt: Optional[bool] = None  # /**< --prec-sqrt={true|false} (-prec-sqrt) For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation. Default: true */
    prec_div: Optional[bool] = None  # /**< --prec-div={true|false} (-prec-div) For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation. Default: true */
    fmad: Optional[bool] = None  # /**< --fmad={true|false} (-fmad) Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations. Default: true */
    use_fast_math: Optional[bool] = None  # /**< --use_fast_math (-use_fast_math) Make use of fast math operations. */
    extra_device_vectorization: Optional[bool] = None  # /**< --extra-device-vectorization (-extra-device-vectorization) Enables more aggressive device code vectorization in the NVVM optimizer. */
    modify_stack_limit: Optional[bool] = None  # /**< --modify-stack-limit={true|false} (-modify-stack-limit) On Linux, during compilation, use setrlimit() to increase stack size to maximum allowed. Default: true */
    dlink_time_opt: Optional[bool] = None  # /**< --dlink-time-opt (-dlto) Generate intermediate code for later link-time optimization. Implies -rdc=true. */
    gen_opt_lto: Optional[bool] = None  # /**< --gen-opt-lto (-gen-opt-lto) Run the optimizer passes before generating the LTO IR. */
    optix_ir: Optional[bool] = None  # /**< --optix-ir (-optix-ir) Generate OptiX IR. Only intended for consumption by OptiX through appropriate APIs. */
    jump_table_density: Optional[int] = None  # /**< --jump-table-density=[0-101] (-jtd) Specify the case density percentage in switch statements. Default: 101 */
    device_stack_protector: Optional[bool] = None  # /**< --device-stack-protector={true|false} (-device-stack-protector) Enable (disable) the generation of stack canaries in device code. Default: false */
    define_macro: Optional[Union[str, Tuple[str, str]]] = None  # /**< --define-macro=<def> (-D) Predefine a macro. Can be either a string, in which case that macro will be set to 1, or a tuple of strings, in which case the first element is defined as the second */
    undefine_macro: Optional[str] = None  # /**< --undefine-macro=<def> (-U) Cancel any previous definition of a macro. */
    include_path: Optional[str] = None  # /**< --include-path=<dir> (-I) Add the directory to the list of directories to be searched for headers. */
    pre_include: Optional[str] = None  # /**< --pre-include=<header> (-include) Preinclude a header during preprocessing. */
    no_source_include: Optional[bool] = None  # /**< --no-source-include (-no-source-include) Disable the default behavior of adding the directory of each input source to the include path. */
    std: Optional[str] = None  # /**< --std={c++03|c++11|c++14|c++17|c++20} (-std) Set language dialect to C++03, C++11, C++14, C++17 or C++20. Default: c++17 */
    builtin_move_forward: Optional[bool] = None  # /**< --builtin-move-forward={true|false} (-builtin-move-forward) Provide builtin definitions of std::move and std::forward. Default: true */
    builtin_initializer_list: Optional[bool] = None  # /**< --builtin-initializer-list={true|false} (-builtin-initializer-list) Provide builtin definitions of std::initializer_list class and member functions. Default: true */
    disable_warnings: Optional[bool] = None  # /**< --disable-warnings (-w) Inhibit all warning messages. */
    restrict: Optional[bool] = None  # /**< --restrict (-restrict) Programmer assertion that all kernel pointer parameters are restrict pointers. */
    device_as_default_execution_space: Optional[bool] = None  # /**< --device-as-default-execution-space (-default-device) Treat entities with no execution space annotation as __device__ entities. */
    device_int128: Optional[bool] = None  # /**< --device-int128 (-device-int128) Allow the __int128 type in device code. */
    optimization_info: Optional[str] = None  # /**< --optimization-info=<kind> (-opt-info) Provide optimization reports for the specified kind of optimization. */
    display_error_number: Optional[bool] = None  # /**< --display-error-number (-err-no) Display diagnostic number for warning messages. Default: true */
    no_display_error_number: Optional[bool] = None  # /**< --no-display-error-number (-no-err-no) Disable the display of a diagnostic number for warning messages. */
    diag_error: Optional[str] = None  # /**< --diag-error=<error-number>,… (-diag-error) Emit error for specified diagnostic message number(s). */
    diag_suppress: Optional[str] = None  # /**< --diag-suppress=<error-number>,… (-diag-suppress) Suppress specified diagnostic message number(s). */
    diag_warn: Optional[str] = None  # /**< --diag-warn=<error-number>,… (-diag-warn) Emit warning for specified diagnostic message number(s). */
    brief_diagnostics: Optional[bool] = None  # /**< --brief-diagnostics={true|false} (-brief-diag) Disable or enable showing source line and column info in a diagnostic. Default: false */
    time: Optional[str] = None  # /**< --time=<file-name> (-time) Generate a CSV table with the time taken by each compilation phase. */
    split_compile: Optional[int] = None  # /**< --split-compile= <number of threads> (-split-compile) Perform compiler optimizations in parallel. */
    fdevice_syntax_only: Optional[bool] = None  # /**< --fdevice-syntax-only (-fdevice-syntax-only) Ends device compilation after front-end syntax checking. */
    minimal: Optional[bool] = None  # /**< --minimal (-minimal) Omit certain language features to reduce compile time for small programs. */
    device_stack_protector: Optional[bool] = None  # /**< --device-stack-protector (-device-stack-protector) Enable stack canaries in device code. */

    def __post_init__(self):
        # Format options into a list of strings
        self.formatted_options = []
        if self.gpu_architecture is not None:
            self.formatted_options.append(f"--gpu-architecture={self.gpu_architecture}".encode())
        if self.device_c is not None:
            self.formatted_options.append("--device-c".encode())
        if self.device_w is not None:
            self.formatted_options.append("--device-w".encode())
        if self.relocatable_device_code is not None:
            self.formatted_options.append(f"--relocatable-device-code={'true' if self.relocatable_device_code else 'false'}".encode())
        if self.extensible_whole_program is not None:
            self.formatted_options.append("--extensible-whole-program".encode())
        if self.device_debug is not None:
            self.formatted_options.append("--device-debug".encode())
        if self.generate_line_info is not None:
            self.formatted_options.append("--generate-line-info".encode())
        if self.dopt is not None:
            self.formatted_options.append(f"--dopt={'on' if self.dopt else 'off'}".encode())
        if self.ptxas_options is not None:
            self.formatted_options.append(f"--ptxas-options={self.ptxas_options}".encode())
        if self.maxrregcount is not None:
            self.formatted_options.append(f"--maxrregcount={self.maxrregcount}".encode())
        if self.ftz is not None:
            self.formatted_options.append(f"--ftz={'true' if self.ftz else 'false'}".encode())
        if self.prec_sqrt is not None:
            self.formatted_options.append(f"--prec-sqrt={'true' if self.prec_sqrt else 'false'}".encode())
        if self.prec_div is not None:
            self.formatted_options.append(f"--prec-div={'true' if self.prec_div else 'false'}".encode())
        if self.fmad is not None:
            self.formatted_options.append(f"--fmad={'true' if self.fmad else 'false'}".encode())
        if self.use_fast_math is not None:
            self.formatted_options.append("--use_fast_math".encode())
        if self.extra_device_vectorization is not None:
            self.formatted_options.append("--extra-device-vectorization".encode())
        if self.modify_stack_limit is not None:
            self.formatted_options.append(f"--modify-stack-limit={'true' if self.modify_stack_limit else 'false'}".encode())
        if self.dlink_time_opt is not None:
            self.formatted_options.append("--dlink-time-opt".encode())
        if self.gen_opt_lto is not None:
            self.formatted_options.append("--gen-opt-lto".encode())
        if self.optix_ir is not None:
            self.formatted_options.append("--optix-ir".encode())
        if self.jump_table_density is not None:
            self.formatted_options.append(f"--jump-table-density={self.jump_table_density}".encode())
        if self.device_stack_protector is not None:
            self.formatted_options.append(f"--device-stack-protector={'true' if self.device_stack_protector else 'false'}".encode())
        if self.define_macro is not None:
            if isinstance(self.define_macro, tuple):
                self.formatted_options.append(f"--define-macro={self.define_macro[0]}={self.define_macro[1]}".encode())
            else:
                self.formatted_options.append(f"--define-macro={self.define_macro}".encode())
        if self.undefine_macro is not None:
            self.formatted_options.append(f"--undefine-macro={self.undefine_macro}".encode())
        if self.include_path is not None:
            self.formatted_options.append(f"--include-path={self.include_path}".encode())
        if self.pre_include is not None:
            self.formatted_options.append(f"--pre-include={self.pre_include}".encode())
        if self.no_source_include is not None:
            self.formatted_options.append("--no-source-include".encode())
        if self.std is not None:
            self.formatted_options.append(f"--std={self.std}".encode())
        if self.builtin_move_forward is not None:
            self.formatted_options.append(f"--builtin-move-forward={'true' if self.builtin_move_forward else 'false'}".encode())
        if self.builtin_initializer_list is not None:
            self.formatted_options.append(f"--builtin-initializer-list={'true' if self.builtin_initializer_list else 'false'}".encode())
        if self.disable_warnings is not None:
            self.formatted_options.append("--disable-warnings".encode())
        if self.restrict is not None:
            self.formatted_options.append("--restrict".encode())
        if self.device_as_default_execution_space is not None:
            self.formatted_options.append("--device-as-default-execution-space".encode())
        if self.device_int128 is not None:
            self.formatted_options.append("--device-int128".encode())
        if self.optimization_info is not None:
            self.formatted_options.append(f"--optimization-info={self.optimization_info}".encode())
        if self.display_error_number is not None:
            self.formatted_options.append("--display-error-number".encode())
        if self.no_display_error_number is not None:
            self.formatted_options.append("--no-display-error-number".encode())
        if self.diag_error is not None:
            self.formatted_options.append(f"--diag-error={self.diag_error}".encode())
        if self.diag_suppress is not None:
            self.formatted_options.append(f"--diag-suppress={self.diag_suppress}".encode())
        if self.diag_warn is not None:
            self.formatted_options.append(f"--diag-warn={self.diag_warn}".encode())
        if self.brief_diagnostics is not None:
            self.formatted_options.append(f"--brief-diagnostics={'true' if self.brief_diagnostics else 'false'}".encode())
        if self.time is not None:
            self.formatted_options.append(f"--time={self.time}".encode())
        if self.split_compile is not None:
            self.formatted_options.append(f"--split-compile={self.split_compile}".encode())
        if self.fdevice_syntax_only is not None:
            self.formatted_options.append("--fdevice-syntax-only".encode())
        if self.minimal is not None:
            self.formatted_options.append("--minimal".encode())
        if self.device_stack_protector is not None:
            self.formatted_options.append("--device-stack-protector".encode())


class Program:
    """Represent a compilation machinery to process programs into
    :obj:`~cuda.core.experimental._module.ObjectCode`.

    This object provides a unified interface to multiple underlying
    compiler libraries. Compilation support is enabled for a wide
    range of code types and compilation types.

    Parameters
    ----------
    code : Any
        String of the CUDA Runtime Compilation program.
    code_type : Any
        String of the code type. Currently only ``"c++"`` is supported.

    """

    __slots__ = ("_handle", "_backend", "_options" )
    _supported_code_type = ("c++", )
    _supported_target_type = ("ptx", "cubin", "ltoir", )

    def __init__(self, code, code_type, options : ProgramOptions = ProgramOptions()):
        self._handle = None
        if code_type not in self._supported_code_type:
            raise NotImplementedError

        if code_type.lower() == "c++":
            if not isinstance(code, str):
                raise TypeError
            # TODO: support pre-loaded headers & include names
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            self._handle = handle_return(
                nvrtc.nvrtcCreateProgram(code.encode(), b"", 0, [], []))
            self._backend = "nvrtc"
        else:
            raise NotImplementedError
        
        self._options = options.formatted_options

    def __del__(self):
        """Return close(self)."""
        self.close()

    def close(self):
        """Destroy this program."""
        if self._handle is not None:
            handle_return(nvrtc.nvrtcDestroyProgram(self._handle))
            self._handle = None

    def compile(self, target_type, name_expressions=(), logs=None):
        """Compile the program with a specific compilation type.

        Parameters
        ----------
        target_type : Any
            String of the targeted compilation type.
            Supported options are "ptx", "cubin" and "ltoir".
        options : Union[List, Tuple], optional
            List of compilation options associated with the backend
            of this :obj:`Program`. (Default to no options)
        name_expressions : Union[List, Tuple], optional
            List of explicit name expressions to become accessible.
            (Default to no expressions)
        logs : Any, optional
            Object with a write method to receive the logs generated
            from compilation.
            (Default to no logs)

        Returns
        -------
        :obj:`~cuda.core.experimental._module.ObjectCode`
            Newly created code object.

        """
        if target_type not in self._supported_target_type:
            raise NotImplementedError

        if self._backend == "nvrtc":
            if name_expressions:
                for n in name_expressions:
                    handle_return(
                        nvrtc.nvrtcAddNameExpression(self._handle, n.encode()),
                        handle=self._handle)
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            handle_return(
                nvrtc.nvrtcCompileProgram(self._handle, len(self._options), self._options),
                handle=self._handle)

            size_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}Size")
            comp_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}")
            size = handle_return(size_func(self._handle), handle=self._handle)
            data = b" " * size
            handle_return(comp_func(self._handle, data), handle=self._handle)

            symbol_mapping = {}
            if name_expressions:
                for n in name_expressions:
                    symbol_mapping[n] = handle_return(nvrtc.nvrtcGetLoweredName(
                        self._handle, n.encode()), handle=self._handle)

            if logs is not None:
                logsize = handle_return(nvrtc.nvrtcGetProgramLogSize(self._handle),
                                        handle=self._handle)
                if logsize > 1:
                    log = b" " * logsize
                    handle_return(nvrtc.nvrtcGetProgramLog(self._handle, log),
                                  handle=self._handle)
                    logs.write(log.decode())

            # TODO: handle jit_options for ptx?

            return ObjectCode(data, target_type, symbol_mapping=symbol_mapping)

    @property
    def backend(self):
        """Return the backend type string associated with this program."""
        return self._backend

    @property
    def handle(self):
        """Return the program handle object."""
        return self._handle
