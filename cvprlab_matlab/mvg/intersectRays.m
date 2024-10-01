function out=intersectRays(Ax0,Ax1,Bx0,Bx1)

% A(s) = Ax0 + s Ax1
% B(t) = Bx0 + t Bx1

A=[ Ax1'*Ax1  -Ax1'*Bx1 ;...
    Ax1'*Bx1  -Bx1'*Bx1];

b=[Ax1'*(Bx0-Ax0) ; ...
   Bx1'*(Bx0-Ax0)];

x=inv(A)*b;

s=x(1);
t=x(2);

out = (Ax0 + s*Ax1) + ( (Bx0 + t*Bx1) - (Ax0 + s*Ax1) )/2;
