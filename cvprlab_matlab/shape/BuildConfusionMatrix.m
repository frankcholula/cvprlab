%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% BuildConfusionMatrix.m
%% Builds a confusion matrix from classification and ground truth data
%% Confusion matrix to be read row-by-row for each ground truth category
%% i.e. row 1 shows the classifications for all of the category 1 examples
%% in the training set, labelled as category 1 in the ground truth
%%
%% Usage:  C = BuildConfusionMatrix(classification, groundtruth)
%%
%% IN:  classification - Algorithm labels of each test [l1 l2 l3...ln]
%%      groundtruth    - Ground truth labels for each test [g1 g2 g3...gn]
%%
%% OUT: C              - Confusion matrix
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

function C=BuildConfusionMatrix(classification, ground_truth, number_categories)

C=zeros(number_categories,number_categories);

for gt=1:number_categories
    this_cat_querycount=sum(ground_truth==gt);
    clsidx=classification(find(ground_truth==gt));
    for cls=1:number_categories
        C(gt,cls)=sum(clsidx==cls);        
    end
    C(gt,:)=C(gt,:)./norm(C(gt,:));
end