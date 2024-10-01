%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_PCA.m
%% Project data to eigenmodel basis
%%
%% Usage:  [pobs E] = Eigen_PCA(obs)
%%
%% IN:  obs            - Data in n x d format
%%      method         - Deflation method (see Eigen_Deflate)
%%      param          - Deflation method parameter (see Eigen_Deflate)
%%
%% OUT: pobs           - Data projected into subspace
%%      E              - Eigenmodel fitted to dad
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function [pobs E]=Eigen_PCA(obs,method,param)

    E=Eigen_Build(obs);
    E=Eigen_Deflate(E,method,param);
    pobs=Eigen_Project(obs,E);
    