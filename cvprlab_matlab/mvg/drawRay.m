function drawRay(x0,x1,col)

s=500;
plot3([x0(1) x0(1)+s*x1(1)], [x0(2) x0(2)+s*x1(2)], [x0(3) x0(3)+s*x1(3)],col);