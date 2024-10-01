close all;

CAMS=loadCamera();

% Show the cameras
figure; hold on;
plot3(0,0,0,'g*'); %origin (on studio floor, which is x-z plane)

for c=1:7 % for all 7 cameras
    drawCamera3D(CAMS(c).R,CAMS(c).T,CAMS(c).K,CAMS(c).imgsize(2),CAMS(c).imgsize(1));    
end

