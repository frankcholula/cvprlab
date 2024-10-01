close all;
clear all;

image(ones(100,100,3));
hold on;
axis square; axis on; axis xy;

[u v]=ginput(4);
G=[u' ; v'];

M=[-1 3 -3 1 ; 3 -6 3 0 ; -3 3 0 0 ; 1 0 0 0];

s=0:0.01:1;
Q=[s.^3 ; s.^2 ; s ; ones(size(s))];


P=G*M*Q;

plot(u,v,'r*');
plot(P(1,:),P(2,:),'b');

