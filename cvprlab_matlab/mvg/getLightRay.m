function [x0 x1]=getLightRay(R,t,K,pt2D)

x0=-R'*t;

pt2D=pt2D-K(1:2,3);



x1=R' * [pt2D(1) /K(1,1);...
         pt2D(2) /K(2,2);...
         1];
     
x1=x1/norm(x1);
