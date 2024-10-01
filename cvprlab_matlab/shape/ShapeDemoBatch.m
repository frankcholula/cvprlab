%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% ShapeDemoBatch.m
%% A lab demo of shape classification using free-hand drawn sketches of
%% shapes.  Using an eigenmodel based classifier, and shape descriptors
%% based on Fourier Descriptors.
%%
%% Usage:  ShapeDemo   (no parameters, stand-alone demo program)
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom
 
close all;
clear all;

% Define dataset
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
       
       % get fourier descriptors and normalise
       D=ComputeFDAngular(polyg,2:9);
       % or maybe try central moments?
%       D=HuMoments(mask);
       % add to observation data
       observations{cat}=[observations{cat} D'];
   end
end

%observations=ShapeDemoDoPCA(observations);

TESTSET=0.25; %  10% of data to be test, the rest training
accuracy=zeros(1,10);

for crossvalidate_trials=1:10
    fprintf('Crossvalidation trial %d\n',crossvalidate_trials);
    
    % partition training and test data at random for each category
    for cat=1:length(category)
        index=randperm(SAMPLES);
        testindex=index(1:round(SAMPLES*TESTSET));
        trainindex=setdiff(index,testindex);        
        test{cat}=observations{cat}(:,testindex);
        train{cat}=observations{cat}(:,trainindex);
    end    

    % Train eigenmodels for each category
    for cat=1:length(category)
        e{cat}=Eigen_Build(train{cat});
        e{cat}=Eigen_Deflate(e{cat},'keepf',0.97);
    end

    % Build test query set 
    query=[]; % the test observations
    groundtruth=[]; % the correct classification (ground truth) foreach query
    for cat=1:length(category)
        query=[query test{cat}];
        groundtruth=[groundtruth repmat(cat,1,size(test{cat},2))];
    end

    % Classify
    scores=[];
    for cat=1:length(category)
        scores=[scores ; Eigen_Mahalanobis(query,e{cat})];
    end
    bestscores=min(scores);
    
    classification=[];
    for q=1:size(query,2)
        idx=find(scores(:,q)==bestscores(q));
        idx=idx(randperm(length(idx))); idx=idx(1);
        classification=[classification idx];
    end
    
    % generate statistics
    correct=double(sum(classification==groundtruth));
    total=size(query,2);
    
    accuracy(crossvalidate_trials)=correct/total;
    fprintf('Accuracy %d%%\n',round(100*accuracy(crossvalidate_trials)));
    
    % confusion matrix
    C{crossvalidate_trials}=BuildConfusionMatrix(classification,groundtruth,length(category));
    
end

% Overall statistics
fprintf('Mean average precision is  %f%% (std.dev. %f%%)\n',100*mean(accuracy),100*std(accuracy));
fprintf('Pure chance precision would be %f%%\n',100/length(category));

avgconf=zeros(length(category));
for i=1:length(C)
   avgconf=avgconf+C{i}./length(C); 
end
category
avgconf
