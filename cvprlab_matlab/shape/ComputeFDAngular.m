%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% ComputeFDAngular.m
%% Compute the angular Fourier descriptors of a closed polygon
%% Specifically the freq magnitudes of the signal created by taking angles
%% between adjacent vectors around polygon.  Angles actually cosine of ang
%% to prevent discontinuities in signal.
%%
%% Usage:  [FD] = ComputeFDAngular (P,[n])
%%
%% IN:  P            - regularly sample polygon, vertices [x1,x2..;y1,y2..]
%%                     To create a reguarly sampled polygon use the
%%                     SamplePolygonPerimeter function
%%      n            - OPTIONAL e.g. [1:5] compute first 5 FDs. 
%%      debug        - OPTIONAL 0 or 1, the latter will plot signal pre-FFT
%%
%% OUT: FD           - The fourier descriptor/s requested [fd1 fd2..etc]
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function FD = ComputeFDAngular (p, n, debug)

% handle second argument if absent
if nargin==1
    n=1:floor(size(p,2)/2);
    debug=0;
end
if nargin==2
    debug=0;
end


% compute vectors around polygon
v=[p(:,2:end) p(:,1)]-p;

% normalise them
v=v./repmat(sqrt(sum(v.^2)),2,1);

% compute cross product between normalised vectors, and from this the sine
% of the angle between them.
v(3,:)=0;
cross_v=cross([v(:,2:end) v(:,1)],v);
sintheta=cross_v(3,:);

% fourier decomposition of signal "sintheta"
F=fft(sintheta);
% only interested in the magnitude of each frequency component
mag=sqrt(real(F).^2 + imag(F).^2);

% return the descriptors we are interested in
FD=mag(n);





