close all;
clear all;

RANGE=100;
NCLUSTERS=5;
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

% 2) Plot data - this is the "ground truth" we hope kmeans can recover
figure;
title('Synthetic data (ground truth)');
cla; hold on; 
axis ([-RANGE RANGE -RANGE RANGE]);
for n=1:NCLUSTERS
    h=plot(data{n}(1,:),data{n}(2,:),'x');
    set(h,'color',rand(1,3));
end

% 3) Concatenate all the data 
alldata=[];
for n=1:NCLUSTERS
    alldata=[alldata data{n}];
end

% 4) Run KMeans and plot the centres of the identified clusters
fprintf('Running KMeans over %d points (of dimension %d)\n',size(alldata,2),DIMENSION);
startingcentres=rand(NCLUSTERS,DIMENSION);
centres=Kmeans(startingcentres,alldata',zeros(1,14));
plot(centres(:,1),centres(:,2),'c*');

% 5) Make a nearest neighbour assignment
alldists=[];
for n=1:NCLUSTERS
   % compute distance of allpoints to this cluster 
   dst=alldata-repmat(centres(n,:)',1,size(alldata,2));
   dst=sqrt(sum(dst.^2));    
   alldists=[alldists;dst];
end
mindists=min(alldists);
alldists=double(alldists==repmat(mindists,NCLUSTERS,1));
for n=1:NCLUSTERS
   alldists(n,:)=alldists(n,:).*n; 
end
classification=max(alldists);

% 6) Plot the assignments
figure;
title('Clustered data after KMeans/KNN');
cla; hold on; 
axis ([-RANGE RANGE -RANGE RANGE]);
ctr=1;

colours=rand(NCLUSTERS,3);
for n=1:size(alldata,2)
    h=plot(alldata(1,n),alldata(2,n),'x');
    set(h,'color',colours(classification(n),:));
end

% 7) How many were correct?
groundtruth=[];
for n=1:NCLUSTERS
    groundtruth=[groundtruth ones(1,NPOINTS)*n];
end
