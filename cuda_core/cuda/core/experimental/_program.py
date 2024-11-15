# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda import nvrtc
from cuda.core.experimental._utils import handle_return, _handle_boolean_option, check_or_create_options
from cuda.core.experimental._module import ObjectCode

from typing import Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ProgramOptions:
    """Customizable :obj:`ProgramOptions` for NVRTC.

    Attributes
    ----------
    gpu_architecture : str, optional
        Specify the name of the class of GPU architectures for which the input must be compiled.
        Valid values: compute_50, compute_52, compute_53, compute_60, compute_61, compute_62, compute_70, compute_72, compute_75, compute_80, compute_87, compute_89, compute_90, compute_90a, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80, sm_87, sm_89, sm_90, sm_90a.
        Default: compute_52
        Maps to: --gpu-architecture=<arch> (-arch)
    device_c : bool, optional
        Generate relocatable code that can be linked with other relocatable device code.
        Equivalent to --relocatable-device-code=true.
        Default: False
        Maps to: --device-c (-dc)
    device_w : bool, optional
        Generate non-relocatable code.
        Equivalent to --relocatable-device-code=false.
        Default: False
        Maps to: --device-w (-dw)
    relocatable_device_code : bool, optional
        Enable (disable) the generation of relocatable device code.
        Default: False
        Maps to: --relocatable-device-code={true|false} (-rdc)
    extensible_whole_program : bool, optional
        Do extensible whole program compilation of device code.
        Default: False
        Maps to: --extensible-whole-program (-ewp)
    device_debug : bool, optional
        Generate debug information. If --dopt is not specified, then turns off all optimizations.
        Default: False
        Maps to: --device-debug (-G)
    generate_line_info : bool, optional
        Generate line-number information.
        Default: False
        Maps to: --generate-line-info (-lineinfo)
    dopt : bool, optional
        Enable device code optimization. When specified along with ‘-G’, enables limited debug information generation for optimized device code.
        Default: None
        Maps to: --dopt on (-dopt)
    ptxas_options : str, optional
        Specify options directly to ptxas, the PTX optimizing assembler.
        Default: None
        Maps to: --ptxas-options <options> (-Xptxas)
    maxrregcount : int, optional
        Specify the maximum amount of registers that GPU functions can use.
        Default: None
        Maps to: --maxrregcount=<N> (-maxrregcount)
    ftz : bool, optional
        When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal values.
        Default: False
        Maps to: --ftz={true|false} (-ftz)
    prec_sqrt : bool, optional
        For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
        Default: True
        Maps to: --prec-sqrt={true|false} (-prec-sqrt)
    prec_div : bool, optional
        For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation.
        Default: True
        Maps to: --prec-div={true|false} (-prec-div)
    fmad : bool, optional
        Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations.
        Default: True
        Maps to: --fmad={true|false} (-fmad)
    use_fast_math : bool, optional
        Make use of fast math operations.
        Default: False
        Maps to: --use_fast_math (-use_fast_math)
    extra_device_vectorization : bool, optional
        Enables more aggressive device code vectorization in the NVVM optimizer.
        Default: False
        Maps to: --extra-device-vectorization (-extra-device-vectorization)
    modify_stack_limit : bool, optional
        On Linux, during compilation, use setrlimit() to increase stack size to maximum allowed.
        Default: True
        Maps to: --modify-stack-limit={true|false} (-modify-stack-limit)
    dlink_time_opt : bool, optional
        Generate intermediate code for later link-time optimization.
        Default: False
        Maps to: --dlink-time-opt (-dlto)
    gen_opt_lto : bool, optional
        Run the optimizer passes before generating the LTO IR.
        Default: False
        Maps to: --gen-opt-lto (-gen-opt-lto)
    optix_ir : bool, optional
        Generate OptiX IR. Only intended for consumption by OptiX through appropriate APIs.
        Default: False
        Maps to: --optix-ir (-optix-ir)
    jump_table_density : int, optional
        Specify the case density percentage in switch statements, and use it as a minimal threshold to determine whether jump table (brx.idx instruction) will be used to implement a switch statement.
        Default: 101
        Maps to: --jump-table-density=[0-101] (-jtd)
    device_stack_protector : bool, optional
        Enable (disable) the generation of stack canaries in device code.
        Default: False
        Maps to: --device-stack-protector={true|false} (-device-stack-protector)
    define_macro : Union[str, Tuple[str, str]], optional
        Predefine a macro. Can be either a string, in which case that macro will be set to 1, or a 2 element tuple of strings, in which case the first element is defined as the second.
        Default: None
        Maps to: --define-macro=<def> (-D)
    undefine_macro : str, optional
        Cancel any previous definition of a macro.
        Default: None
        Maps to: --undefine-macro=<def> (-U)
    include_path : str, optional
        Add the directory to the list of directories to be searched for headers.
        Default: None
        Maps to: --include-path=<dir> (-I)
    pre_include : str, optional
        Preinclude a header during preprocessing.
        Default: None
        Maps to: --pre-include=<header> (-include)
    no_source_include : bool, optional
        Disable the default behavior of adding the directory of each input source to the include path.
        Default: False
        Maps to: --no-source-include (-no-source-include)
    std : str, optional
        Set language dialect to C++03, C++11, C++14, C++17 or C++20.
        Default: c++17
        Maps to: --std={c++03|c++11|c++14|c++17|c++20} (-std)
    builtin_move_forward : bool, optional
        Provide builtin definitions of std::move and std::forward.
        Default: True
        Maps to: --builtin-move-forward={true|false} (-builtin-move-forward)
    builtin_initializer_list : bool, optional
        Provide builtin definitions of std::initializer_list class and member functions.
        Default: True
        Maps to: --builtin-initializer-list={true|false} (-builtin-initializer-list)
    disable_warnings : bool, optional
        Inhibit all warning messages.
        Default: False
        Maps to: --disable-warnings (-w)
    restrict : bool, optional
        Programmer assertion that all kernel pointer parameters are restrict pointers.
        Default: False
        Maps to: --restrict (-restrict)
    device_as_default_execution_space : bool, optional
        Treat entities with no execution space annotation as __device__ entities.
        Default: False
        Maps to: --device-as-default-execution-space (-default-device)
    device_int128 : bool, optional
        Allow the __int128 type in device code.
        Default: False
        Maps to: --device-int128 (-device-int128)
    optimization_info : str, optional
        Provide optimization reports for the specified kind of optimization.
        Default: None
        Maps to: --optimization-info=<kind> (-opt-info)
    display_error_number : bool, optional
        Display diagnostic number for warning messages.
        Default: True
        Maps to: --display-error-number (-err-no)
    no_display_error_number : bool, optional
        Disable the display of a diagnostic number for warning messages.
        Default: False
        Maps to: --no-display-error-number (-no-err-no)
    diag_error : str, optional
        Emit error for specified diagnostic message number(s).
        Default: None
        Maps to: --diag-error=<error-number>,… (-diag-error)
    diag_suppress : str, optional
        Suppress specified diagnostic message number(s).
        Default: None
        Maps to: --diag-suppress=<error-number>,… (-diag-suppress)
    diag_warn : str, optional
        Emit warning for specified diagnostic message number(s).
        Default: None
        Maps to: --diag-warn=<error-number>,… (-diag-warn)
    brief_diagnostics : bool, optional
        Disable or enable showing source line and column info in a diagnostic.
        Default: False
        Maps to: --brief-diagnostics={true|false} (-brief-diag)
    time : str, optional
        Generate a CSV table with the time taken by each compilation phase.
        Default: None
        Maps to: --time=<file-name> (-time)
    split_compile : int, optional
        Perform compiler optimizations in parallel.
        Default: 1
        Maps to: --split-compile= <number of threads> (-split-compile)
    fdevice_syntax_only : bool, optional
        Ends device compilation after front-end syntax checking.
        Default: False
        Maps to: --fdevice-syntax-only (-fdevice-syntax-only)
    minimal : bool, optional
        Omit certain language features to reduce compile time for small programs.
        Default: False
        Maps to: --minimal (-minimal)
    device_stack_protector : bool, optional
        Enable stack canaries in device code.
        Default: False
        Maps to: --device-stack-protector (-device-stack-protector)
    """
    gpu_architecture: Optional[str] = None
    device_c: Optional[bool] = None
    device_w: Optional[bool] = None
    relocatable_device_code: Optional[bool] = None
    extensible_whole_program: Optional[bool] = None
    device_debug: Optional[bool] = None
    generate_line_info: Optional[bool] = None
    dopt: Optional[bool] = None
    ptxas_options: Optional[str] = None
    maxrregcount: Optional[int] = None
    ftz: Optional[bool] = None
    prec_sqrt: Optional[bool] = None
    prec_div: Optional[bool] = None
    fmad: Optional[bool] = None
    use_fast_math: Optional[bool] = None
    extra_device_vectorization: Optional[bool] = None
    modify_stack_limit: Optional[bool] = None
    dlink_time_opt: Optional[bool] = None
    gen_opt_lto: Optional[bool] = None
    optix_ir: Optional[bool] = None
    jump_table_density: Optional[int] = None
    device_stack_protector: Optional[bool] = None
    define_macro: Optional[Union[str, Tuple[str, str]]] = None
    undefine_macro: Optional[str] = None
    include_path: Optional[str] = None
    pre_include: Optional[str] = None
    no_source_include: Optional[bool] = None
    std: Optional[str] = None
    builtin_move_forward: Optional[bool] = None
    builtin_initializer_list: Optional[bool] = None
    disable_warnings: Optional[bool] = None
    restrict: Optional[bool] = None
    device_as_default_execution_space: Optional[bool] = None
    device_int128: Optional[bool] = None
    optimization_info: Optional[str] = None
    display_error_number: Optional[bool] = None
    no_display_error_number: Optional[bool] = None
    diag_error: Optional[str] = None
    diag_suppress: Optional[str] = None
    diag_warn: Optional[str] = None
    brief_diagnostics: Optional[bool] = None
    time: Optional[str] = None
    split_compile: Optional[int] = None
    fdevice_syntax_only: Optional[bool] = None
    minimal: Optional[bool] = None
    device_stack_protector: Optional[bool] = None
    
    def __post_init__(self):
        self._formatted_options = []
        if self.gpu_architecture is not None:
            self._formatted_options.append(f"--gpu-architecture={self.gpu_architecture}")
        if self.device_c is not None and self.device_c:
            self._formatted_options.append("--device-c")
        if self.device_w is not None and self.device_w:
            self._formatted_options.append("--device-w")
        if self.relocatable_device_code is not None:
            self._formatted_options.append(f"--relocatable-device-code={_handle_boolean_option(self.relocatable_device_code)}")
        if self.extensible_whole_program is not None and self.extensible_whole_program:
            self._formatted_options.append("--extensible-whole-program")
        if self.device_debug is not None and self.device_debug:
            self._formatted_options.append("--device-debug")
        if self.generate_line_info is not None and self.generate_line_info:
            self._formatted_options.append("--generate-line-info")
        if self.dopt is not None:
            self._formatted_options.append(f"--dopt={'on' if self.dopt else 'off'}")
        if self.ptxas_options is not None:
            self._formatted_options.append(f"--ptxas-options={self.ptxas_options}")
        if self.maxrregcount is not None:
            self._formatted_options.append(f"--maxrregcount={self.maxrregcount}")
        if self.ftz is not None:
            self._formatted_options.append(f"--ftz={_handle_boolean_option(self.ftz)}")
        if self.prec_sqrt is not None:
            self._formatted_options.append(f"--prec-sqrt={_handle_boolean_option(self.prec_sqrt)}")
        if self.prec_div is not None:
            self._formatted_options.append(f"--prec-div={_handle_boolean_option(self.prec_div)}")
        if self.fmad is not None:
            self._formatted_options.append(f"--fmad={_handle_boolean_option(self.fmad)}")
        if self.use_fast_math is not None and self.use_fast_math:
            self._formatted_options.append("--use_fast_math")
        if self.extra_device_vectorization is not None and self.extra_device_vectorization:
            self._formatted_options.append("--extra-device-vectorization")
        if self.modify_stack_limit is not None:
            self._formatted_options.append(f"--modify-stack-limit={_handle_boolean_option(self.modify_stack_limit)}")
        if self.dlink_time_opt is not None and self.dlink_time_opt:
            self._formatted_options.append("--dlink-time-opt")
        if self.gen_opt_lto is not None and self.gen_opt_lto:
            self._formatted_options.append("--gen-opt-lto")
        if self.optix_ir is not None and self.optix_ir:
            self._formatted_options.append("--optix-ir")
        if self.jump_table_density is not None:
            self._formatted_options.append(f"--jump-table-density={self.jump_table_density}")
        if self.device_stack_protector is not None:
            self._formatted_options.append(f"--device-stack-protector={_handle_boolean_option(self.device_stack_protector)}")
        if self.define_macro is not None:
            if isinstance(self.define_macro, tuple):
                assert len(self.define_macro) == 2
                self._formatted_options.append(f"--define-macro={self.define_macro[0]}={self.define_macro[1]}")
            else:
                self._formatted_options.append(f"--define-macro={self.define_macro}")
        if self.undefine_macro is not None:
            self._formatted_options.append(f"--undefine-macro={self.undefine_macro}")
        if self.include_path is not None:
            self._formatted_options.append(f"--include-path={self.include_path}")
        if self.pre_include is not None:
            self._formatted_options.append(f"--pre-include={self.pre_include}")
        if self.no_source_include is not None and self.no_source_include:
            self._formatted_options.append("--no-source-include")
        if self.std is not None:
            self._formatted_options.append(f"--std={self.std}")
        if self.builtin_move_forward is not None:
            self._formatted_options.append(f"--builtin-move-forward={_handle_boolean_option(self.builtin_move_forward)}")
        if self.builtin_initializer_list is not None:
            self._formatted_options.append(f"--builtin-initializer-list={_handle_boolean_option(self.builtin_initializer_list)}")
        if self.disable_warnings is not None and self.disable_warnings:
            self._formatted_options.append("--disable-warnings")
        if self.restrict is not None and self.restrict:
            self._formatted_options.append("--restrict")
        if self.device_as_default_execution_space is not None and self.device_as_default_execution_space:
            self._formatted_options.append("--device-as-default-execution-space")
        if self.device_int128 is not None and self.device_int128:
            self._formatted_options.append("--device-int128")
        if self.optimization_info is not None:
            self._formatted_options.append(f"--optimization-info={self.optimization_info}")
        if self.display_error_number is not None and self.display_error_number:
            self._formatted_options.append("--display-error-number")
        if self.no_display_error_number is not None and self.no_display_error_number:
            self._formatted_options.append("--no-display-error-number")
        if self.diag_error is not None:
            self._formatted_options.append(f"--diag-error={self.diag_error}")
        if self.diag_suppress is not None:
            self._formatted_options.append(f"--diag-suppress={self.diag_suppress}")
        if self.diag_warn is not None:
            self._formatted_options.append(f"--diag-warn={self.diag_warn}")
        if self.brief_diagnostics is not None:
            self._formatted_options.append(f"--brief-diagnostics={_handle_boolean_option(self.brief_diagnostics)}")
        if self.time is not None:
            self._formatted_options.append(f"--time={self.time}")
        if self.split_compile is not None:
            self._formatted_options.append(f"--split-compile={self.split_compile}")
        if self.fdevice_syntax_only is not None and self.fdevice_syntax_only:
            self._formatted_options.append("--fdevice-syntax-only")
        if self.minimal is not None and self.minimal:
            self._formatted_options.append("--minimal")
        if self.device_stack_protector is not None and self.device_stack_protector:
            self._formatted_options.append("--device-stack-protector")

    def _as_bytes(self):
        result = []
        for option in self._formatted_options:
            result.append(option.encode())
        return result
    
    def __repr__(self):
        #__TODO__ improve this
        return self._formatted_options

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
        
        self._options = options._as_bytes()

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
