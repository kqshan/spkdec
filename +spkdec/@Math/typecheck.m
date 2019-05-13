function tf = typecheck(x, flags)
% Test whether the given array's datatype matches the given flags
%   tf = typecheck(x, flags)
%
% Returns:
%   tf          Logical scalar (true or false)
% Required arguments:
%   x           Input array
%   flags       Struct with fields:
%     use_f32     Data should be single-precision (32-bit float)
%     use_gpu     Data should be a gpuArray

if flags.use_f32
    desired_class = 'single';
else
    desired_class = 'double';
end

if flags.use_gpu
    % Check that x is a gpuArray, then use isaUnderlying() to get its datatype
    tf = isa(x,'gpuArray') && isaUnderlying(x,desired_class);
else
    % isa(x,'double') returns false if x is any type of gpuArray
    tf = isa(x,desired_class);
end

end
