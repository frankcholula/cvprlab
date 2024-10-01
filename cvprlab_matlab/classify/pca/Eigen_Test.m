%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Test.m
%% Tests eigen model functions and serves as an example of library use
%%
%% Usage:  Eigen_Test()  - standalone function
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

close all;
clear all;

N=1000;  % samples
D=2;     % dimension (has to be 2 for plotting purposes)
S=1;     % std-dev to partition points in visualization

% generate some Gaussian test data
org=rand(D,1)*100;
theta=rand(1)*2*pi;
sf=rand(1);
R=[sf*cos(theta) -sin(theta); sf*sin(theta) cos(theta)];
obs=R*randn(D,N)+repmat(org,1,N);

% plot the data
plot(obs(1,:),obs(2,:),'bx');
hold on;
axis equal;

% build the Eigenmodel
E=Eigen_Build(obs);

% plot in green all points within 1 standard deviation of the model
d=Eigen_Mahalanobis(obs,E);
idx=find(d<S);
plot(obs(1,idx),obs(2,idx),'gx');

% visualize the Eigenmodel
Eigen_Plot2D(E,S);
