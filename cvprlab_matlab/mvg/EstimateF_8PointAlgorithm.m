close all;

%% Build matrix for solving Fundamental matrix
A=[];
for pt=1:8
   
    x=Pleft(1,pt);
    y=Pleft(2,pt);
    u=Pright(1,pt);
    v=Pright(2,pt);
    
    A=[A ;  [x*u x*v x y*u y*v y u v 1] ];
    
end

%% We have a homogeneous linear system Ax=0 need to solve with SVD

[U S V]=svd(A);

% The best solution is the eigenvector corresponding to smallest
% eigenvector.  This will always be the last column of V in matlab SVD
x=V(:,end);

% reshape solution x into a 3x3 matrix
F=reshape(x,3,3)';
