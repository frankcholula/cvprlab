close all;
clear all;

XMIN=0;
XMAX=4;
YMIN=0;
YMAX=2;
ZMIN=0;
ZMAX=4;
XSTEP=0.05;
YSTEP=0.05;
ZSTEP=0.05;

VISHULL_THRESHOLD=5;
NUMCAM=8;

cd 'mvdata';
C=loadCamera();
cd '..';


% Plot camera positions
plotCameras();

% Get masks of foreground object from all 8 views
for cam=1:NUMCAM
   imgfile=sprintf('mvdata/cam%d.png',cam-1);
   im=double(imread(imgfile))./255;
   bluedominance=im(:,:,3)./(im(:,:,1)+im(:,:,2)+im(:,:,3));
   mask{cam}=bluedominance<0.38;
%   imshow(mask{cam});
%   pause;
end

voxelvolume=zeros(length(XMIN:XSTEP:XMAX),length(YMIN:YSTEP:YMAX),length(ZMIN:ZSTEP:ZMAX));
size(voxelvolume)
xctr=0;
for x=XMIN:XSTEP:XMAX
    drawnow;
    fprintf('Reconstructing... %d%% done\n',round(100*((x-XMIN)/(XMAX-XMIN))));
    xctr=xctr+1;
    yctr=0;
    for y=YMIN:YSTEP:YMAX
        yctr=yctr+1;
        zctr=0;
        for z=ZMIN:ZSTEP:ZMAX
            zctr=zctr+1;
            hit=0;
            for cam=1:NUMCAM
                p = C(cam).T + C(cam).R * [x; y; z];
                
                u = p(1) * C(cam).fx / p(3) + C(cam).cx;
                v = p(2) * C(cam).fy / p(3) + C(cam).cy;
                
                u=round(u);
                v=round(v);
                if (u<0 | v <0 | u>=size(mask{cam},2) | v>=size(mask{cam},1))
                    % out of bounds - voxel is not set
                else
                    % in bounds, check mask
                    if (mask{cam}(v+1,u+1)>0)
                        hit=hit+1;
                    end
                    if (hit>=VISHULL_THRESHOLD) 
                        plot3(x,y,z,'bx');
                        break;
                    end
                end
            end            
            if (hit>=VISHULL_THRESHOLD)
                voxelvolume(xctr,yctr,zctr)=1;
            end
        end
    end
end

fprintf('Voxel occupancy %f%%\n',sum(sum(sum(voxelvolume)))/sum(size(voxelvolume)));
