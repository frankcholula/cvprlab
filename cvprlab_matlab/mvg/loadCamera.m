function CAM=loadCamera()

CALIBFILE='calibration.txt';

fp=fopen(CALIBFILE,'rt');

numcams=fscanf(fp,'%d',1);
reserved=fscanf(fp,'%d',1);

for cams=1:numcams
    CAM(cams).imgsize=fscanf(fp,'%d',4);
    CAM(cams).imgsize=CAM(cams).imgsize([2 4]);

    CAM(cams).fx=fscanf(fp,'%f',1);
    CAM(cams).fy=fscanf(fp,'%f',1);
    CAM(cams).cx=fscanf(fp,'%f',1);
    CAM(cams).cy=fscanf(fp,'%f',1);
    CAM(cams).k=fscanf(fp,'%f',1);
    CAM(cams).K = [CAM(cams).fx  0          CAM(cams).cx ; ...
                 0           CAM(cams).fy CAM(cams).cy ; ...
                 0           0          1];
 
    CAM(cams).R=reshape(fscanf(fp,'%f',9),3,3)';
    CAM(cams).T=fscanf(fp,'%f',3);

end

fclose(fp);
