function handles = plot_features(spk_X, varargin)
% Plot the given spike features as a scatterplots of orthogonal projections
%   handles = plot_features(spk_X)
%
% Returns:
%   handles     Struct of graphics object handles
% Required arguments:
%   spk_X       [D x N] spike features to plot
% Optional parameters (key/value pairs) [default]:
%   nBins       Number of bins per scatterplot          [ 1600/D ]
%   upsample    Upsample ratio used during rendering    [ 2 ]
%   range       [D x 2] data range                      [ auto ]
%
% These scatterplots compare all pairs of feature axes. Areas with a higher
% density of points are denoted using warmer colors.

% Dimensions
[D,N] = size(spk_X);
assert(D < 1000, 'D is way too big; perhaps you transposed the spk_X input?');

% Parse optional arguments
ip = inputParser();
ip.addParameter('nBins', round(1600/D), @isscalar);
ip.addParameter('upsample', 2, @isscalar);
ip.addParameter('range', [], @(x) isempty(x) || isequal(size(x),[D 2]));
ip.parse( varargin{:} );
prm = ip.Results;

% Set some parameters
P = prm.nBins;
img_R = prm.upsample;
feat_range = prm.range;
if isempty(feat_range)
    feat_range = quantile(spk_X, [.001, .999], 2);
    feat_range = feat_range * [1.5,-0.5; -0.5,1.5];
    feat_range(:,1) = max(feat_range(:,1), min(spk_X,[],2));
    feat_range(:,2) = min(feat_range(:,2), max(spk_X,[],2));
end

% Count the number of spikes in each upsampled bin
feat_min = feat_range(:,1);
feat_max = feat_range(:,2);
p = ceil((img_R*P-1)*(spk_X-feat_min)./(feat_max-feat_min) + 0.5);
p = max(p,1); p = min(p,img_R*P);
p = gather(p);
% Do this for each pair of features
img = arrayfun(@(i,j) {accumarray(p([i j],:)', 1, [img_R*P,img_R*P])}, ...
    repmat((1:D)',[1 D]), repmat(1:D,[D 1]));
% Then concatenate them into a [img_R*P*D x img_R*P*D] image
img = cell2mat(img);

% Render as translucent dots with warmer colors on overlap
% Generate the dot mask
dot_radius = img_R*2; dot_opacity = 0.5; bkg_color = [1 1 1];
dr = ceil(dot_radius);
dot_mask = sqrt((-dr:dr).^2 + (-dr:dr)'.^2);
dot_mask = max(0,min(1, dot_radius-dot_mask));
% Apply the dot mask, then resize back down to the desired size
img = imfilter(img, dot_mask);
img = imresize(img, [P*D, P*D], 'box');
% Apply the colormap
cmap = [
    233 233 233
    469 185 192
    690 179 136
    974 479  65
    980 657 126
    ] / 1000;
alpha = 1 - (1-dot_opacity).^img;
color = max(0,log(img)+2*log(dot_opacity));
color = min(1,color/quantile(color(color>0),0.995));
color = interp1(linspace(0,1,size(cmap,1))', cmap, color(:));
color = reshape(color, [P*D, P*D, 3]);
img = (1-alpha).*shiftdim(bkg_color,-1) + alpha.*color;

% Plot
fh = figure();
ih = imshow(img, 'XData',[0.5,D+0.5],'YData',[0.5,D+0.5], ...
    'InitialMagnification',50);
drawnow();
ah = gca(); set(ah,'Visible','on', 'YDir','normal');
ah_W = P*D/ah.Position(3); ah_H = P*D/ah.Position(4);
fh.PaperPosition = [0, 0, ah_W/300, ah_H/300];
% Make it pretty
set(ah,'XTick',1:D, 'YTick',1:D, 'TickLength',[0 0]);
hold on; plot([1;1]*(1.5:D-0.5),[0.5;D+0.5],'-', ...
    [0.5;D+0.5],[1;1]*(1.5:D-0.5),'-', 'Color',[0.3 0.3 0.3], ...
    'LineWidth',0.25);
xlabel('Feature axis #');
ylabel('Feature axis #');
title('Orthogonal projections of feature space');

% Collect the graphics handles
handles = struct();
handles.ah = ah;
handles.image = ih;

end