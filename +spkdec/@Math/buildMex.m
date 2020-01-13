function buildMex(varargin)
% Build the MEX files, which may accelerate performance
%   buildMex(...)
%
% Optional parameters (key/value pairs) [default]:
%   files       MEX files to build          [ all ]
%
% This requires a C complier (e.g. gcc) and the NVidia CUDA compiler (nvcc).
% Any existing MEX files will be overwritten by this operation.

% Source code that needs to be compiled
cpp_func_list = {'pbsolve_mex'};
cuda_func_list = {};

% Source directory containing the C++/CUDA source
class_dir = fileparts(mfilename('fullpath'));
src_dir = fullfile(class_dir,'mex_src');

% Compile the C++ code
fprintf('Compiling C++ code...\n');
% Compiler/linker options
mex_opts = {
    'CXXOPTIMFLAGS="-O3 -DNDEBUG"'  % Set optimization level to O3
    '-largeArrayDims'               % Use 64-bit API
    '-lmwlapack'                    % Link to LAPACK
    '-lmwblas'                      % Link to BLAS
    };
% Compile, link, and move to the MoDT class directory
for ii = 1:length(cpp_func_list)
    func_name = cpp_func_list{ii};
    fprintf('    %s\n',func_name);
    % Compile+link
    src_file = fullfile(src_dir, [func_name '.cpp']);
    mex(mex_opts{:}, src_file)
    % Move to the MoDT class directory
    if ~strcmp(pwd, class_dir)
        mex_file = fullfile(pwd, [func_name '.' mexext]);
        movefile(mex_file, class_dir);
    end
end
fprintf('Finish compliing C++ code\n\n');

% Compile the CUDA code
fprintf('Compiling CUDA code...\n');
% Compiler/linker options
nvcc_opts = [...
    '-gencode=arch=compute_30,code=sm_30 ' ...
    '-gencode=arch=compute_50,code=sm_50 ' ...
    '-gencode=arch=compute_60,code=sm_60 ' ...
    '-std=c++11' ...
    ];
mexcuda_opts = {
    '-lcublas'                      % Link to cuBLAS
    ['NVCCFLAGS="' nvcc_opts '"']
    '-L/usr/local/cuda/lib64'       % Location of CUDA libraries
    };
% Compile, link, and move to the MoDT class directory
for ii = 1:length(cuda_func_list)
    func_name = cuda_func_list{ii};
    fprintf('    %s\n',func_name);
    % Compile+link
    src_file = fullfile(src_dir, [func_name '.cu']);
    mexcuda(mexcuda_opts{:}, src_file)
    % Move to the MoDT class directory
    if ~strcmp(pwd, class_dir)
        mex_file = fullfile(pwd, [func_name '.' mexext]);
        movefile(mex_file, class_dir);
    end
end
fprintf('Finished compiling CUDA code\n');

end
