close all;
clear all;

M=[2 -3 0 1 ; -2 3 0 0 ; 1 -2 1 0 ; 1 -1 0 0];

G1=[ 0 9 0 0 ; 0 0 1 -1];
G2=[ 9 18 0 0 ; 0 0 -1 1];

s=0:0.01:1;

Q=[s.^3 ; s.^2 ; s ; ones(size(s))];

P1=G1*M*Q;
P2=G2*M*Q;

hold on;
plot(P1(1,:),P1(2,:),'b');
plot(P2(1,:),P2(2,:),'r');


