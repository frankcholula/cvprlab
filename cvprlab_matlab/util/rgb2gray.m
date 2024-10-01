function gimg=rgb2gray(img)

gimg=0.3 * img(:,:,1) + 0.59 * img(:,:,2) + 0.11 * img(:,:,3);