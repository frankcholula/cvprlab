close all;
clear all;

NPOINTS=8;

s=0:0.01:1;
Q=[s.^3 ; s.^2; s ; ones(size(s))];

M=0.5*[-1 2 -1 0 ; 3 -5 0 2; -3 4 1 0; 1 -1 0 0];  % Catmull-Rom Blending Matrix
%M=(1/6)*[-1 3 -3 1 ; 3 -6 0  4; -3 3 3 1; 1 0 0 0];  %B-spline blending

    
imgshow(ones(100));hold on;
[x y]=ginput(NPOINTS);
close all;

masterG=[x';y'];
masterG=[masterG(:,1) masterG masterG(:,end)];
plot(masterG(1,:),masterG(2,:),'rx');

hold on; 
axis equal;
axis ij;

for lp=1:size(masterG,2)-3
    
    G=masterG(:,lp:lp+3);
    P=G*M*Q;

    plot(P(1,:),P(2,:),'g');
    
end

