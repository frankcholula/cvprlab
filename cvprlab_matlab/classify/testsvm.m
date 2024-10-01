close all;
clear all;


SPLITFORTRAINING=0.2;
RANGE=100;
NCLUSTERS=3;
SPREAD=5;
NPOINTS=100;
DIMENSION=2;    % if you change this from 2, the clustering will work but 
                % the plot will break


% 1) Generate NCLUSTERS of synthetic data
data=cell(1,NCLUSTERS); % each entry holds a cluster of generated data
for n=1:NCLUSTERS
   clustermean=(rand(DIMENSION,1)-rand(DIMENSION,1))*RANGE;
   data{n}=randn(DIMENSION,NPOINTS)*SPREAD + repmat(clustermean,1,NPOINTS);
end


% 2) Separate the data in training and test
test=cell(1,NCLUSTERS);
train=cell(1,NCLUSTERS);
cutoff=round(NPOINTS*SPLITFORTRAINING);  % take 20% for training
for n=1:NCLUSTERS
    train{n}=data{n}(:,1:cutoff);
    test{n}=data{n}(:,(cutoff+1):end);
end

% Put all train points and all train classifications in individual rows
alltrain_obs=[];
alltrain_class=[];
alltest_obs=[];
alltest_class=[];
for n=1:NCLUSTERS
    alltrain_obs=[alltrain_obs train{n}];
    alltest_obs=[alltest_obs test{n}];
    alltrain_class=[alltrain_class ones(1,cutoff)*n];
    alltest_class=[alltest_class ones(1,(NPOINTS-cutoff))*n];
end
alltrain_obs=alltrain_obs';
alltrain_class=alltrain_class';
alltest_obs=alltest_obs';
alltest_class=alltest_class';

% 3) Setup SVM parameters
kernel='gaussian';
kerneloption=5;
C=10000000;
verbose=1;
lambda=1e-7;
nbclass=NCLUSTERS;

% 4) Train the SVM with training data
[xsup,w,b,nbsv,pos,alpha]=svmmulticlass(alltrain_obs,alltrain_class,nbclass,C,lambda,kernel,kerneloption,verbose);
[ypred] = svmmultival(alltrain_obs,xsup,w,b,nbsv,kernel,kerneloption);
fprintf( '\nClassification correct on training data : %2.2f \n',100*sum(ypred==alltrain_class)/length(alltrain_class)); 

% 5) Test the classifier on test data
[ypred,maxi] = svmmultival(alltest_obs,xsup,w,b,nbsv,kernel,kerneloption);
fprintf( '\nClassification correct on test data : %2.2f \n',100*sum(ypred==alltest_class)/length(alltest_class)); 


