%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Build.m
%% Builds an eigen model from n observation of d-dimensional data  
%%
%% Usage:  E = Eigen_Build(observations)
%%
%% IN:  observations   - d row, n column matrix of observations
%%
%% OUT: E              - Eigenmodel structure
%%                       E.org - mean
%%                       E.vct - matrix of eigenvectors one per column 
%%                       E.val - column of eigenvalues(matching E.vct cols)
%%                       E.N   - number of observations used i.e. = n
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function E=Eigen_Build(obs)

    E.N  =size(obs,2);
    E.D  =size(obs,1);
    E.org=mean(obs')';
    
    obs_translated=obs-repmat(E.org,1,E.N);
    
    
    C=(1/E.N) * (obs_translated * obs_translated');
    
    [U V]=eig(C);
    
    % sort eigenvectors and eigenvalues by eigenvalue size (desc)
    linV=V*ones(size(V,2),1);
    S=[linV U'];
    S=flipud(sortrows(S,1));
    U=S(:,2:end)';
    V=S(:,1);
    
    E.vct=U;
    E.val=V;
    