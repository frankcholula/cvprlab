%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% imgshow.m
%% Display an image in uint8 (unnormalised) or double (normalised) format
%% autodetecting greyscale or colour.  Serves as a simplified replacement
%% for the image processing toolbox imshow command for which we have
%% insufficient licences for use in an undergraduate class environment
%%
%% Usage:  imgshow (image)
%%
%% IN:  image        - The image to display, as speified above
%% 
%% OUT: N/A
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function imgshow(img_in)

% If the image is greyscale, turn it into RGB by copying data to all 3
% channels RGB.  From this point we are dealing with colour images only
if length(size(img_in))==2
    img_in(:,:,2)=img_in(:,:,1);
    img_in(:,:,3)=img_in(:,:,1);
end

% If the image is a uint8 image, then convert it into a normalised image
if strcmp(class(img_in),'uint8')
    img_in=double(img_in)./255;
end
img_in=double(img_in);

% At this stage we have a normalised colour image.

% We are going to display using Matlab's image function, which reqiures a
% uint8 0-255 image input - so, convert.
img_out=uint8(img_in.*255);

% Display and sort out axis properties to prevent distortion
image(img_out);
axis ij;
axis equal;
axis tight; % get rid of little white border Matlab introduces for us


