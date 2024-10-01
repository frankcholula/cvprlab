%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Deflate.m
%% Project data to eigenmodel basis
%%
%% Usage:  pobs = Eigen_Project(obs,E)
%%
%% IN:  E              - Eigenmodel to project points
%%      obs            - Data to project
%%
%% OUT: pobs           - Projected data
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function pobs=Eigen_Project(obs,E)

    obs_translated=obs-repmat(E.org,1,size(obs,2));
    pobs=E.vct'*obs_translated;