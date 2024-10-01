%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% CentralMoment.m
%% Computes the (ij)th 2D central moment of a region from its binary mask
%%
%% Usage:  m = CentralMoment (mask,moment)
%%
%% IN:  mask         - The binary mask
%%      moment       - [i j] the order of moment required (i,j)
%% 
%% OUT: m            - Computed moment (scalar)
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function m=CentralMoment(mask,order)

i=order(1);
j=order(2);

[r c]=find(mask);
area=sum(sum(mask));
mn=[mean(c);mean(r)];

cx=(c-mn(1)).^i;
cy=(r-mn(2)).^j;

m=sum(cx.*cy)./area;
