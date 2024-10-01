%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Plot2D.m
%% Plots an eigenmodel with D=2 and 2 eigenvectors for debug purposes
%%
%% Usage:  Eigen_Plot2D(E)
%%
%% IN:  E              - The eigenmodel
%%      S              - Plot bounding ellipse at S standard devations
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function Eigen_Plot2D(E,STDDEV)

if nargin==1
    STDDEV=1;
end

th=0:0.01:2*pi;
P=[cos(th);sin(th)];
PP=E.vct*diag(STDDEV*sqrt(E.val))*P;
PP=PP+repmat(E.org,1,size(PP,2));

plot(PP(1,:),PP(2,:),'r');
