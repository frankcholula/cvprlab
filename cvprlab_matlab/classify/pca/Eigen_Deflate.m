%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Eigen_Deflate.m
%% Compute distance in std. devs of observations/data from eigen model 
%%
%% Usage:  E = Eigen_Deflate(E,method,param)
%%
%% IN:  E              - Eigenmodel to deflate
%%      method         - Deflation method to use               
%%      param          - Optional parameter, according to deflate method
%%
%% OUT: E              - Deflated eigenmodel
%%
%% Deflation methods:
%%             'keepn' - Keep 'param' most significant eigenvectors
%%             'keepf' - Keep 'param' % of energy
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function E=Eigen_Deflate(E,method,param)

    switch method
        
        case 'keepn'
            E.val=E.val(1:param);
            E.vct=E.vct(:,1:param);
        case 'keepf'
            totalenergy=sum(abs(E.val));
            currentenergy=0;
            rank=0;
            for i=1:size(E.vct,2)
                if currentenergy>(totalenergy*param)
                    break;
                end
                rank=rank+1;
                currentenergy=currentenergy+E.val(i);
            end
            E=Eigen_Deflate(E,'keepn',rank);
    
    end
    
    