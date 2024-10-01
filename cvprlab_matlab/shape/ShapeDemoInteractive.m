%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% ShapeDemoInteractive.m
%% A lab demo of shape classification using free-hand drawn sketches of
%% shapes.  Using an eigenmodel based classifier, and shape descriptors
%% based on Fourier Descriptors.
%%
%% Usage:  ShapeDemoInteractive   (no parameters, stand-alone demo program)
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom
 
close all;
clear all;

FDMAX=8;  % Use 8 Fourier Descriptors for classification

% Define dataset location/size/category names
PATH='c:\visiondemo\labcode_wk1-4\shape\shapeimages\';
SAMPLES=30;
category{1}='arch';
category{2}='fish';
category{3}='triangle';
category{4}='cross';
category{5}='square';
category{6}='circle';

% Load all shape examples and convert to Chain Codes
samples=cell(0);
for cat=1:length(category)
   fprintf('Loading %ss\n',category{cat});
   for n=1:SAMPLES
       fname=[PATH,category{cat},sprintf('%04d',n),'.bmp'];
       img=rgb2gray(imread(fname));
       mask=img>0;
       [cc startpoint]=Chaincode(mask);
       samples{cat}{n}=cc;
   end
end

% Compute shape descriptors for all examples
observations=cell(0);
for cat=1:length(category)
   fprintf('Computing descriptors for %ss\n',category{cat});
   observations{cat}=[];
   for n=1:SAMPLES
       % convert chaincode to polygon
       [mask polyg]=ChaincodeRasterize(samples{cat}{n});
       polyg=SamplePolygonPerimeter(polyg,100);
       
       % get fourier descriptors
       D=ComputeFDAngular(polyg,2:(FDMAX+1));

       % add to observation data
       observations{cat}=[observations{cat} D'];
   end
end

% Train eigenmodel for each category
for cat=1:length(category)
     e{cat}=Eigen_Build(observations{cat});
     e{cat}=Eigen_Deflate(e{cat},'keepf',0.97);
end

% Prompt user for a sketch
[mask p]=DrawShape(200);

% We will use the mask i.e. bitmap, and forget about the polygon.  The
% intention here is to use the same type of input as the source files (i.e.
% an image) and thus make the sketch query subject to the same noise/
% distortion inherent in a pixel-dervied polygon.

% mask to chain code
[cc startpoint]=Chaincode(mask);
% chain code back to mask and polygon
[m polyg]=ChaincodeRasterize(cc,[200 200],startpoint);
% resample polygon to 100 regularly spaced points on boundary
polyg=SamplePolygonPerimeter(polyg,100);
% compute fourier descriptors from resampled polygons
queryFD=ComputeFDAngular(polyg,2:(FDMAX+1))';

% Classify
% measure distances of query to each of the trained Eigenmodels
scores=[];
for cat=1:length(category)
    scores=[scores ; Eigen_Mahalanobis(queryFD,e{cat})];
end

% find the minimum distance i.e. the closest eigenmodel/category
querycategory=find(scores==min(scores));

% turn scores into a probability vector i.e. make them sum to 1
% this is just for display/output purposes
probabilities=scores./norm(scores);

% output probabilities and the most likely shape category 
probabilities    
fprintf('You drew a %s\n',category{querycategory});
