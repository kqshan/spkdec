function [outputs, sinks] = make_detection_outputs(basis, varargin)
% Create a struct of DataSinks for the util.detect_spikes() function
%   [outputs, sinks] = make_detection_outputs(basis, ...)
%
% Returns:
%   outputs     Struct of DataSink objects compatible with util.detect_spikes()
%   sinks       Nested struct of underlying DataSinks and attributes (see below)
% Required arguments:
%   basis       SpikeBasisCS that will be used for spike detection
% Optional parameters (key/value pairs):
%   filename    HDF5 output filename, or empty to use DataSinkMatrix objects
%               instead. Default = [].
%   *_datatype  Storage datatype. Defaults: index = uint32, subidx = uint8,
%               {feature,spknorm,resid,residnorm} = single.
%   *_scaling   Dataset scaling ({feature,resid} datasets only). This is useful
%               if you wish to store these as integer types. Default = 1.
%   det_thresh  Detection threshold as specified to util.detect_spikes()
%   det_refrac  Detection refractory period as specified to util.detect_spikes()
% HDF5 parameters (key/value pairs) [default]:
%   chunk_size  HDF5 chunk size in extensible dimension     [ 4096 ]
%   deflate     gzip compression level (0..9)               [ 1 ]
%   shuffle     Enable Shuffle filter                       [ true ]
%   
% The <sinks> struct is a nested struct that emulates the HDF5 file structure:
%   version     File version string (currently '1.0')
%   feature     Spike features
%     dataset     [D x N] DataSink
%     scaling     Scaling coefficient to apply when reading this dataset
%   index       Spike source index (1..T) where detected
%     dataset     [N] DataSink
%   subidx      Spike sub-sample shift index (1..R)
%     dataset     [N] DataSink
%     R           Overall number of sub-sample shifts
%   spknorm     Spike norm (in whitened space)
%     dataset     [N] DataSink
%   resid       Spike residuals (unwhitened)
%     dataset     [L x C x N] DataSink
%     scaling     Scaling coefficient to apply when reading this dataset
%   residnorm   Residual norms (in whitened space)
%     dataset     [N] DataSink
%   basis       Spike basis parameters
%     waveforms   [L x C x D] spike basis waveforms
%     t0          Basis sample index (1..L) corresponding to t=0
%     subshift    [L x L x R] sub-sample shift operators
%   whitener    Whitening parameters
%     wh_filt     [W x C] whitening filters
%     wh_ch       [C x C] cross-channel whitening operator
%     delay       Whitening filter delay (#samples)
%   detparams   Detection parameters (only if det_thresh,det_refrac are given)
%     thresh      Detection threshold (see spkdec.Solver.det_thresh)
%     norm        Minimum detectable spknorm, equal to sqrt(D*det_thresh)
%     refrac      Detection refractory period (#samples)
%
% If you call this with <file>=[], then after running util.detect_spikes(), you
% can extract the data matrices out of their DataSinkMatrix wrappers using:
%   for fn = {'feature','index','subidx','spknorm','resid','residnorm'}
%       sinks.(fn{1}).dataset = sinks.(fn{1}).dataset.data;
%   end
%
% See also: spkdec.util.detect_spikes

errid_pfx = 'spkdec:io:make_detection_outputs';

% Parse optional parameters
ip = inputParser();
ip.addParameter('filename', [], @(x) isempty(x) || ischar(x));
ip.addParameter('feature_datatype', 'single', @ischar);
ip.addParameter('index_datatype', 'uint32', @ischar);
ip.addParameter('subidx_datatype', 'uint8', @ischar);
ip.addParameter('spknorm_datatype', 'single', @ischar);
ip.addParameter('resid_datatype', 'single', @ischar);
ip.addParameter('residnorm_datatype', 'single', @ischar);
ip.addParameter('feature_scaling', 1, @isscalar);
ip.addParameter('resid_scaling', 1, @isscalar);
ip.addParameter('det_thresh', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('det_refrac', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('chunk_size', 4096, @isscalar);
ip.addParameter('deflate', 1, @isscalar);
ip.addParameter('shuffle', true, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Copy the datatype and scaling params into separate structs
datatype_prm = struct();
scaling_prm = struct();
datatype_suffix = '_datatype'; scaling_suffix = '_scaling';
for fn_cell = fieldnames(prm)'
    fn = fn_cell{1};
    if endsWith(fn, datatype_suffix)
        datatype_prm.(fn(1:end-length(datatype_suffix))) = prm.(fn);
    elseif endsWith(fn,scaling_suffix)
        scaling_prm.(fn(1:end-length(scaling_suffix))) = prm.(fn);
    end
end

% Parameters for creating the base-level datasinks
filename = prm.filename;
assert(isempty(filename) || exist(filename,'file')==0, ...
    [errid_pfx ':FileExists'], 'File "%s" already exists; aborting', filename);
h5_prm = struct('chunk_size',prm.chunk_size, ...
    'Shuffle',prm.shuffle, 'Deflate',prm.deflate);

% Initialize the <sinks> struct with the file version
sinks = struct('version','1.0');
attnames_and_ndims = {'version',NaN};

%% Construct the <outputs> struct

% Get some dimensions
D = basis.D;
C = basis.C;
L = basis.L;
% Create the underlying data sinks
datasets = struct();
dataset_names_and_shapes = {'feature',[D Inf]; 'index',Inf; 'subidx',Inf; ...
    'spknorm',Inf; 'resid',[L C Inf]; 'residnorm',Inf};
for name_dim = dataset_names_and_shapes'
    [name,shape] = deal(name_dim{:});
    datatype = datatype_prm.(name);
    % Create either a DataSinkMatrix or a DataSinkH5
    if isempty(filename)
        dims = shape; dims(isinf(shape)) = 0;
        data = zeros([dims,1], datatype);
        sink = spkdec.io.DataSinkMatrix(shape, data);
    else
        sink = spkdec.io.DataSinkH5(shape, filename, 'dsname',['/' name], ...
            'old_file_ok',true, 'old_ds_ok',false, 'Datatype',datatype, h5_prm);
    end
    datasets.(name) = sink;
end

% Copy these into the <sinks> struct
dataset_names = dataset_names_and_shapes(:,1);
for name_cell = dataset_names'
    name = name_cell{1};
    sinks.(name) = struct('dataset',datasets.(name));
end

% Throw in a gather() to take care of any gpuArrays
for name_cell = dataset_names'
    name = name_cell{1};
    datasets.(name) = spkdec.io.DataSinkScaled(datasets.(name), ...
        'scaling',@gather);
end

% DataSinkInt provides some useful bounds checking for integer types, so wrap
% any integer types inside a DataSinkInt
for name_cell = dataset_names'
    name = name_cell{1};
    datatype = datatype_prm.(name);
    if contains(datatype,'int')
        datasets.(name) = spkdec.io.DataSinkInt(datasets.(name), datatype, ...
            'err_mode','warn');
    end
end

% Apply scaling
for name_cell = fieldnames(scaling_prm)'
    name = name_cell{1};
    read_scale = scaling_prm.(name);
    if read_scale ~= 1
        write_scale = 1/read_scale;
        datasets.(name) = spkdec.io.DataSinkScaled(datasets.(name), ...
            'scaling',write_scale);
    end
    sinks.(name).scaling = read_scale;
    attnames_and_ndims(end+1,:) = {[name '.scaling'], 0}; %#ok<AGROW>
end

% Create the outputs struct
outputs = struct();
for name_cell = {'feature','index','subidx','spknorm'}
    name = name_cell{1};
    outputs.(name) = datasets.(name);
end
% Special case for the residuals
outputs.resid = spkdec.io.ResidSink( basis.toWhBasis(), ...
    'unwhitened',datasets.resid, 'norm',datasets.residnorm );

%% Add the other attributes

% Add them to the <sinks> struct
sinks.basis = struct('waveforms',basis.basis, 't0',basis.t0, ...
    'subshift',basis.interp.shifts);
sinks.subidx.R = basis.R;
sinks.whitener = basis.whitener.saveobj();
attnames_and_ndims = [attnames_and_ndims; {
    'subidx.R',0; 'basis.waveforms',3; 'basis.t0',0; 'basis.subshift',3; 
    'whitener.wh_filt',2; 'whitener.wh_ch',2; 'whitener.delay',0;
    }];
groups_reqd = {'basis','whitener'};
% detparams is optional
detparams = struct();
if ~isempty(prm.det_thresh)
    detparams.thresh = prm.det_thresh;
    detparams.norm = sqrt(D*detparams.thresh);
    attnames_and_ndims(end+1,:) = {'detparams.thresh',0};
    attnames_and_ndims(end+1,:) = {'detparams.norm',0};
end
if ~isempty(prm.det_refrac)
    detparams.refrac = prm.det_refrac;
    attnames_and_ndims(end+1,:) = {'detparams.refrac',0};
end
if ~isempty(fieldnames(detparams))
    sinks.detparams = detparams;
    groups_reqd{end+1} = 'detparams';
end

% Copy these to the HDF5 file
if ~isempty(filename)
    % Create the necessary groups
    cellfun(@(group) create_h5_group(filename, ['/' group]), groups_reqd);
    % Copy the attributes (some of which will be stored as small datasets)
    for name_ndims = attnames_and_ndims'
        [name, ndims] = deal(name_ndims{:});
        copy_struct_to_h5(sinks, filename, name, ndims);
    end
end


end


% -----------------------     Helper functions     -----------------------------


function copy_struct_to_h5(S, file, name, ndims)
% Copy the struct contents as HDF5 datasets or attributes
%   copy_struct_to_h5(S, file, name, ndims)
%
% Required arguments:
%   S       Struct to copy
%   file    HDF5 filename
%   name    Struct fieldname (with '.' for sub-levels)
%   ndims   Number of dimensions in the data
%
% If ndims==0 or the data is a string (regardless of the value of `ndims`), then
% the data will be stored as an attribute. Otherwise, the size of the data and
% `ndims` are used to construct an appropriately-sized HDF5 dataset.

% Traverse the struct tree to build the HDF5 location and get the value
loc = '';
val = S;
for name_portion_cell = strsplit(name,'.')
    name_portion = name_portion_cell{1};
    val = val.(name_portion);
    loc = [loc '/' name_portion]; %#ok<AGROW>
end
% Decide whether this should be a dataset or an attribute
use_att = ischar(val) || (ndims==0);
if use_att
    % Use the last part of the location as the attribute name
    idx = find(loc=='/', 1, 'last');
    attname = loc(idx+1:end);
    loc = loc(1:idx);
    % Write the attribute
    h5writeatt(file, loc, attname, val);
else
    % Determine the dataset dimensions
    dims = size(val);
    inferred_ndims = length(dims);
    if inferred_ndims > ndims
        assert(dims(ndims+1:end)==1, 'more dimensions than expected');
        dims = dims(1:ndims);
    else
        dims = [dims, ones(1,ndims-inferred_ndims)];
    end
    % Create and write the dataset
    h5create(file, loc, dims);
    h5write(file, loc, val);
end
end


function create_h5_group(file, group)
% Create the specified HDF5 group
%   create_h5_group(file, group)
%
% Required arguments:
%   file    HDF5 filename
%   group   HDF5 group name (full path)
fid = H5F.open(file, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
gid = H5G.create(fid, group, 'H5P_DEFAULT','H5P_DEFAULT','H5P_DEFAULT');
H5G.close(gid);
H5F.close(fid);
end
