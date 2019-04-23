function x = typeconv(x, flags)
% Convert a numeric array to the desired datatype and memory location
%   x = typeconv(x, flags)
%
% Returns:
%   x           Output array
% Required arguments:
%   x           Input array
%   flags       Struct with fields:
%     use_f32     Convert x into single-precision (32-bit float)
%     use_gpu     Convert x into a gpuArray (located in GPU vs. host memory)

if flags.use_gpu
    x = gpuArray(x);
else
    x = gather(x);
end

if flags.use_f32
    x = single(x);
else
    x = double(x);
end

end
