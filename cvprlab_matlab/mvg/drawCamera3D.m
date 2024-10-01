function drawCamera3D(R,t,K,width,height)

xlabel('x');
ylabel('y');
zlabel('z');
hold on;
axis equal;

d=mean([K(1,1) K(2,2)])/1000; % focal length
imageplane=[];
[COP pt]=getLightRay(R,t,K,[0;0]);
imageplane=[imageplane COP+pt.*d];
[COP pt]=getLightRay(R,t,K,[width;0]);
imageplane=[imageplane COP+pt.*d];
[COP pt]=getLightRay(R,t,K,[width;height]);
imageplane=[imageplane COP+pt.*d];
[COP pt]=getLightRay(R,t,K,[0;height]);
imageplane=[imageplane COP+pt.*d];

% Draw camera centre of projection (COP)
plot3(COP(1,:),COP(2,:),COP(3,:),'m*');

% Draw the rectangular image plane
imageplane=[imageplane imageplane(:,1)];
plot3(imageplane(1,:),imageplane(2,:),imageplane(3,:),'b');

% Draw lines from imageplane corners to COP
for pt=1:4
   lines=[imageplane(:,pt) COP];
   plot3(lines(1,:),lines(2,:),lines(3,:),'b');   
end
