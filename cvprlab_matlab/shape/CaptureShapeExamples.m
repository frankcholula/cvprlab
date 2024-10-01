%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% CaptureShapeExamples.m
%% A short script to capture examples of shapes from the UI and save them
%% as bitmap images for later use when training/testing shape classificatn
%%
%% Usage:  CaptureShapeExamples (no parameters, stand-alone demo program)
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom
 
close all;
clear all;

SAMPLES=30;

category{1}='arch';
category{2}='fish';
category{3}='triangle';
category{4}='cross';
category{5}='square';
category{6}='circle';

for cat=1:length(category)
    for n=1:SAMPLES
        num=sprintf('%04d',n);
        fprintf('Draw a %s (#%s)\n',category{cat},num);
        [mask p]=DrawShape(200);
        fname=['c:\visiondemo\shape\shapeimages\',category{cat},num,'.bmp'];
        mask(:,:,2)=mask(:,:,1);
        mask(:,:,3)=mask(:,:,1);
        mask=double(mask).*255;
        imwrite(mask,fname);
        close all;
    end
end
