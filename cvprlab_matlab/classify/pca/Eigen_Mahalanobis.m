%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Mahalanobis.m
%% Compute distance in std. devs of observations/data from eigen model 
%%
%% Usage:  d = Eigen_Mahalanobis(observations,E)
%%
%% IN:  E              - Eigenmodel to measure distance from   
%%      observations   - d row, n column matrix of observations
%%
%% OUT: d              - distance of each observation (1 column of n rows)
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function d=Eigen_Mahalanobis(obs,E)

    obs_translated=obs-repmat(E.org,1,size(obs,2));
    
    proj=E.vct'*obs_translated;
    
    dstsq=proj.*proj;
    
    E.val(E.val==0)=1; % check for eigenvalues of 0
    
    dst=dstsq./repmat((E.val),1,size(obs,2));
    
    d=sum(dst);
    
    d=sqrt(d);
    