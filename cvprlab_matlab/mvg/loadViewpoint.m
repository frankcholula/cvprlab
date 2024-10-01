function points=loadViewpoint(camindex);

FILENAMEBASE='points2D_cam';

filename=[FILENAMEBASE,num2str(camindex),'.txt'];
fp=fopen(filename,'rt');

n=fscanf(fp,'%d',1);

pointsGREEN=[];
pointsRED=[];

for i=1:n
   pointsGREEN=[pointsGREEN fscanf(fp,'%f',2)];
   pointsRED=[pointsRED fscanf(fp,'%f',2)];
end

fclose(fp);

points=pointsGREEN;