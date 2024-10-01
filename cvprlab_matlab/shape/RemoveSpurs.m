%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% RemoveSpurs.m
%% Detects spurs on shapes and removes them.  Slow but simple iterative 
%% method.
%%
%% Usage:  mask = RemoveSpurs (mask)
%%
%% IN:  mask         - The binary mask containing filled region/s
%%
%% OUT: mask         - Filtered binary mask
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function mask=RemoveSpurs(mask)

    changed=1;
    while (changed)
        
        changed=0;
        [r c]=find(mask);    
        for lp=1:length(r)
            pt=[c(lp);r(lp)];
            if count_pixel_neighbourhood(mask,pt)==2                 
                mask(  max(1,pt(2)-1):min(size(mask,1),pt(2)+1) , max(1,pt(1)-1):min(size(mask,2),pt(1)+1))=0;                
                changed=1;
            end
        end

    end
       
    
return;

function count=count_pixel_neighbourhood(mask,pt)

    wnd=mask(  max(1,pt(2)-1):min(size(mask,1),pt(2)+1) , max(1,pt(1)-1):min(size(mask,2),pt(1)+1));
    count=length(find(wnd))-1;

return;