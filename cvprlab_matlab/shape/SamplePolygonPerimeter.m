%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% SamplePolygonPerimeter.m
%% Regularly sample the perimeter of a closed polygon and return the
%% co-ordinates of those sampled points.
%%
%% Usage:  points = SamplePolygonPerimeter (P, s)
%%
%% IN:  P            - closed polygon, vertices in format [x1 x2..;y1 y2..]
%%      s            - how many samples
%%
%% OUT: points       - sampled vertices format [x1 x2...;y1 y2...]
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom
 
function [points]=SamplePolygonPerimeter(p,s)

% polygon data check
if size(p,2)<2
    warning('Polygon must have at least two vertices');
    pts=[];
    return;
end

% compute total polygon length
v=[p(:,2:end) p(:,1)]-p;
totallen=sum(sqrt(sum(v.^2)));

% compute step size
rate=totallen/s;

% iterate around polygon
points=[];
currentlen=0;
currentpos=p(:,1);
nextvertexindex=2;
while (currentlen<totallen & size(points,2)<s)
    v=p(:,nextvertexindex)-currentpos;
    dstleft=norm(v);
    if dstleft>rate
        % move along path
        v=v./norm(v);
        currentpos=currentpos+v.*rate;
    else
        % advance to next vertex
        dstleft=rate-dstleft;
        while (dstleft > 0)
            currentpos=p(:,nextvertexindex);
            nextvertexindex=nextvertexindex+1;
            if nextvertexindex>size(p,2)
                nextvertexindex=1;
            end
            v=p(:,nextvertexindex)-currentpos;
            thisadv=min(dstleft,norm(v));
            v=v./norm(v);
            currentpos=currentpos+v.*thisadv;
            dstleft=dstleft-thisadv;
        end
    end
    currentlen=currentlen+rate;   
    points=[points currentpos];
end
