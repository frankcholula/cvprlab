%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% Chaincode.m
%% Obtain exterior contour of a region filled region in chaincode form
%% This is a simplistic 'walk round the edge pixels' method for teaching
%% purposes and more sophisticated methods exist.  In particular any spurs
%% on the region will break this approach, and so the mask should be
%% appropriately filtered before invoking this function.
%%
%% It is assumed only one connected component exists in the mask - if
%% there are more you should split the connected components into separate
%% masks and call this function for each of them
%%
%%
%% Usage:  [cc, startpoint] = Chaincode (mask)
%%
%% IN:  mask         - The binary mask, containing a single filled region
%%
%% OUT: cc           - The chain code string [0=N,1=NE...7=NW]
%%      startpoint   - The starting pixel from which to decode the chain
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function [cc, startpoint]=Chaincode(mask)

    % a constant
    VISITED=2;

    % mask is 1 for region, 0 for background (2=VISITED for visited)
    mask=double(mask>0);

    % find a starting point on the boundary
    [r c]=find(mask);
    startpoint=[-1;-1];
    for lp=1:length(r)
        if count_pixel_neighbourhood(mask,[c(lp) r(lp)])<8
            startpoint=[c(lp) r(lp)];
            break;
        end
    end
    if (startpoint(1)==-1) 
        warning('CHAINCODE:  Tried to create chaincode from empty mask');
        cc=[];
    end

    % visit the startpoint
    mask(startpoint(1),startpoint(2))=VISITED;

    cc=[];
    MOVES=[0 4 2 6 1 5 3 7];  %NE SW SE NW N S E W
    moved=1;
    currentpoint=startpoint;
    while(moved) 
    
        moved=0;

        for m=MOVES
            if validmove(mask,currentpoint,m)
                cc=[cc m];
                currentpoint=makemove(currentpoint,m);
                mask(currentpoint(2),currentpoint(1))=VISITED;
                moved=1;
                break;
            end
        end
    
    end
    
    startpoint=startpoint';
    
return;

function isok=validmove(mask,oldpt,movecode)

    pt=makemove(oldpt,movecode);
    if (pt(1)<1 || pt(2)<1 || pt(1)>size(mask,2) || pt(2)>size(mask,1))
        isok=0;
        return;
    end
    
    n=count_pixel_neighbourhood(mask,pt);
    %mask(pt(2),pt(1))
    if (mask(pt(2),pt(1))==1 & n<8)
        isok=1;
    else
        isok=0;
    end
    
return;

function pt=makemove(pt, movecode)

    switch movecode
        case 0 %N
            pt(2)=pt(2)-1;
        case 1 %NE
            pt(1)=pt(1)+1;
            pt(2)=pt(2)-1;
        case 2 %E
            pt(1)=pt(1)+1;
        case 3 %SE
            pt(1)=pt(1)+1;
            pt(2)=pt(2)+1;
        case 4 %S
            pt(2)=pt(2)+1;
        case 5 %SW
            pt(1)=pt(1)-1;
            pt(2)=pt(2)+1;
        case 6 %W
            pt(1)=pt(1)-1;
        case 7 %NW
            pt(1)=pt(1)-1;
            pt(2)=pt(2)-1;
    end
    
return;

function count=count_pixel_neighbourhood(mask,pt)

    wnd=mask(  max(1,pt(2)-1):min(size(mask,1),pt(2)+1) , max(1,pt(1)-1):min(size(mask,2),pt(1)+1));
    count=length(find(wnd))-1;

return;