close all;
clear all;

studio;

CAML=1;
CAMR=5;

pt1=loadViewpoint(CAML);
pt2=loadViewpoint(CAMR);

counter=0;
for n=1:1000
    
    if (pt1(1,n)<0 | pt2(1,n)<0)
        continue;
    end
    
    [A1 B1]=getLightRay(CAMS(CAML).R,CAMS(CAML).T,CAMS(CAML).K,pt1(:,n));
    [A2 B2]=getLightRay(CAMS(CAMR).R,CAMS(CAMR).T,CAMS(CAMR).K,pt2(:,n));  

% Show the light rays
%    sf=1.5;
%    plot3([A1(1)  A1(1)*sf*B1(1)],[A1(2)  A1(2)*sf*B1(2)],[A1(3)  A1(3)*sf*B1(3)],'r')
%    plot3([A2(1)  A2(1)*sf*B2(1)],[A2(2)  A2(2)*sf*B2(2)],[A2(3)  A2(3)*sf*B2(3)],'g')
    
    p=intersectRays(A1,B1,A2,B2);
    plot3(p(1),p(2),p(3),'b*');
    counter=counter+1;
    
    pause(0.1);
    
end