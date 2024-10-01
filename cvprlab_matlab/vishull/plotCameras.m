% Plot camera positions
figure;
for cam=1:NUMCAM
    origin=-C(cam).R'*C(cam).T;
   drawCamera3D(C(cam).R,C(cam).T,C(cam).K,C(cam).imgsize(2),C(cam).imgsize(1));    
   hold on;
end
xlabel('x');
ylabel('y');
xlabel('z');
title('Visual Hull');
drawnow;
