%from 费南多
%email：571428374@qq.com
%email：fpf0103@163.com
%time:20230617
clc
clear

%%
%读取数据
%加载数字样本数据作为图像数据存储。imageDatastore 根据文件夹名称自动标注图像，
Trainname="imageTrain"; %训练集名称
Testname="imageTest";%测试集名称
imdsTrain = imageDatastore(Trainname,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
imdsValidation = imageDatastore(Testname,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
%%
%定义卷积神经网络架构
layers = [
    imageInputLayer([224 336 3],"Name","imageinput")

    convolution2dLayer(3,16,'Stride',1,"Name","conv_1_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_1")
    leakyReluLayer(0.1,"Name","leakyRelu_1_1")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_1_1")
    
    
    convolution2dLayer(3,32,'Stride',1,"Name","conv_1_2","Padding","same")
    leakyReluLayer(0.1,"Name","leakyRelu_1_2")
    batchNormalizationLayer("Name","batchnorm_1_2")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_1_2")

    convolution2dLayer(3,64,'Stride',1,"Name","conv_1_3","Padding","same")
    % additionLayer(2,"Name","add_2")
    batchNormalizationLayer("Name","batchnorm_1_3")
    leakyReluLayer(0.1,"Name","leakyRelu_1_3")
    % maxPooling2dLayer(3,'Stride',2,"Name","pool_1_3")

    % convolution2dLayer(1,64,'Stride',1,"Name","conv_1_4","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_4")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_4")

    convolution2dLayer(3,64,'Stride',1,"Name","conv_1_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_5")
    leakyReluLayer(0.1,"Name","leakyRelu_1_5")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_1_5")

    convolution2dLayer(3,128,'Stride',1,"Name","conv_1_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_6")
    leakyReluLayer(0.1,"Name","leakyRelu_1_6")

    % convolution2dLayer(1,128,'Stride',1,"Name","conv_1_7","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_7")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_7")

    convolution2dLayer(3,128,'Stride',1,"Name","conv_1_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_8")
    leakyReluLayer(0.1,"Name","leakyRelu_1_8")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_1_8")

    convolution2dLayer(3,256,'Stride',1,"Name","conv_1_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_9")
    leakyReluLayer(0.1,"Name","leakyRelu_1_9")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_10","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_10")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_10")

    convolution2dLayer(3,256,'Stride',1,"Name","conv_1_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_11")
    leakyReluLayer(0.1,"Name","leakyRelu_1_11")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_12","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_12")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_12")

    convolution2dLayer(3,256,'Stride',1,"Name","conv_1_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_13")
    leakyReluLayer(0.1,"Name","leakyRelu_1_13")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_1_13")

    convolution2dLayer(3,512,'Stride',1,"Name","conv_1_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_14")
    leakyReluLayer(0.1,"Name","leakyRelu_1_14")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_15","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_15")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_15")

    convolution2dLayer(3,512,'Stride',1,"Name","conv_1_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_16")
    leakyReluLayer(0.1,"Name","leakyRelu_1_16")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_17","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_17")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_17")

    convolution2dLayer(3,512,'Stride',1,"Name","conv_1_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_18")
    leakyReluLayer(0.1,"Name","leakyRelu_1_18")

    convolution2dLayer(1,2,'Stride',1,"Name","conv_1_19","Padding","same")

    averagePooling2dLayer(7,'Stride',1,"Name","avgpool_1")

    

    concatenationLayer(3,3,"Name",'concat1')
    fullyConnectedLayer(2,"Name","fc")

%     fullyConnectedLayer(100,"Name","fc")
%     dropoutLayer("Name","drop1")
    % fullyConnectedLayer(10,"Name","fc1")
    % leakyReluLayer(0.1,"Name","leakyRelu__1")
    % dropoutLayer("Name","drop1")
    % fullyConnectedLayer(6,"Name","fc2")
    % leakyReluLayer(0.1,"Name","leakyRelu__2")
    % dropoutLayer("Name","drop2")
    % fullyConnectedLayer(2,"Name","fc3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
    ];
% analyzeNetwork(layers)
lgraph = layerGraph(layers);
% lgraph = connectLayers(lgraph,'pool_1_1','add_1/in2');

layers2 = [
    convolution2dLayer(6,16,'Stride',1,"Name","conv_2_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_1")
    leakyReluLayer(0.1,"Name","leakyRelu_2_1")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_2_1")
    
    
    convolution2dLayer(6,32,'Stride',1,"Name","conv_2_2","Padding","same")
    leakyReluLayer(0.1,"Name","leakyRelu_2_2")
    batchNormalizationLayer("Name","batchnorm_2_2")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_2_2")

    convolution2dLayer(6,64,'Stride',1,"Name","conv_2_3","Padding","same")
    % additionLayer(2,"Name","add_2")
    batchNormalizationLayer("Name","batchnorm_2_3")
    leakyReluLayer(0.1,"Name","leakyRelu_2_3")
    % maxPooling2dLayer(3,'Stride',2,"Name","pool_1_3")

    % convolution2dLayer(1,64,'Stride',1,"Name","conv_1_4","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_4")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_4")

    convolution2dLayer(6,64,'Stride',1,"Name","conv_2_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_5")
    leakyReluLayer(0.1,"Name","leakyRelu_2_5")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_2_5")

    convolution2dLayer(6,128,'Stride',1,"Name","conv_2_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_6")
    leakyReluLayer(0.1,"Name","leakyRelu_2_6")

    % convolution2dLayer(1,128,'Stride',1,"Name","conv_1_7","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_7")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_7")

    convolution2dLayer(6,128,'Stride',1,"Name","conv_2_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_8")
    leakyReluLayer(0.1,"Name","leakyRelu_2_8")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_2_8")

    convolution2dLayer(6,256,'Stride',1,"Name","conv_2_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_9")
    leakyReluLayer(0.1,"Name","leakyRelu_2_9")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_10","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_10")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_10")

    convolution2dLayer(3,256,'Stride',1,"Name","conv_2_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_11")
    leakyReluLayer(0.1,"Name","leakyRelu_2_11")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_12","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_12")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_12")

    convolution2dLayer(6,256,'Stride',1,"Name","conv_2_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_13")
    leakyReluLayer(0.1,"Name","leakyRelu_2_13")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_2_13")

    convolution2dLayer(6,512,'Stride',1,"Name","conv_2_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_14")
    leakyReluLayer(0.1,"Name","leakyRelu_2_14")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_15","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_15")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_15")

    convolution2dLayer(6,512,'Stride',1,"Name","conv_2_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_16")
    leakyReluLayer(0.1,"Name","leakyRelu_2_16")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_17","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_17")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_17")

    convolution2dLayer(6,512,'Stride',1,"Name","conv_2_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_18")
    leakyReluLayer(0.1,"Name","leakyRelu_2_18")

    convolution2dLayer(1,2,'Stride',1,"Name","conv_2_19","Padding","same")

    averagePooling2dLayer(7,'Stride',1,"Name","avgpool_2")
    ];
lgraph = addLayers(lgraph,layers2);
lgraph = connectLayers(lgraph,'imageinput','conv_2_1');
lgraph = connectLayers(lgraph,'avgpool_2','concat1/in2');
% % lgraph = connectLayers(lgraph,'pool_2_1','add_2/in2');

layers3 = [
    convolution2dLayer(8,16,'Stride',1,"Name","conv_3_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_1")
    leakyReluLayer(0.1,"Name","leakyRelu_3_1")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_3_1")
    
    
    convolution2dLayer(8,32,'Stride',1,"Name","conv_3_2","Padding","same")
    leakyReluLayer(0.1,"Name","leakyRelu_3_2")
    batchNormalizationLayer("Name","batchnorm_3_2")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_3_2")

    convolution2dLayer(8,64,'Stride',1,"Name","conv_3_3","Padding","same")
    % additionLayer(2,"Name","add_2")
    batchNormalizationLayer("Name","batchnorm_3_3")
    leakyReluLayer(0.1,"Name","leakyRelu_3_3")
    % maxPooling2dLayer(3,'Stride',2,"Name","pool_1_3")

    % convolution2dLayer(1,64,'Stride',1,"Name","conv_1_4","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_4")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_4")

    convolution2dLayer(8,64,'Stride',1,"Name","conv_3_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_5")
    leakyReluLayer(0.1,"Name","leakyRelu_3_5")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_3_5")

    convolution2dLayer(8,128,'Stride',1,"Name","conv_3_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_6")
    leakyReluLayer(0.1,"Name","leakyRelu_3_6")

    % convolution2dLayer(1,128,'Stride',1,"Name","conv_1_7","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_7")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_7")

    convolution2dLayer(8,128,'Stride',1,"Name","conv_3_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_8")
    leakyReluLayer(0.1,"Name","leakyRelu_3_8")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_3_8")

    convolution2dLayer(8,256,'Stride',1,"Name","conv_3_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_9")
    leakyReluLayer(0.1,"Name","leakyRelu_3_9")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_10","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_10")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_10")

    convolution2dLayer(8,256,'Stride',1,"Name","conv_3_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_11")
    leakyReluLayer(0.1,"Name","leakyRelu_3_11")

    % convolution2dLayer(1,256,'Stride',1,"Name","conv_1_12","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_12")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_12")

    convolution2dLayer(8,256,'Stride',1,"Name","conv_3_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_13")
    leakyReluLayer(0.1,"Name","leakyRelu_3_13")
    maxPooling2dLayer(2,'Stride',2,"Name","pool_3_13")

    convolution2dLayer(8,512,'Stride',1,"Name","conv_3_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_14")
    leakyReluLayer(0.1,"Name","leakyRelu_3_14")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_15","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_15")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_15")

    convolution2dLayer(8,512,'Stride',1,"Name","conv_3_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_16")
    leakyReluLayer(0.1,"Name","leakyRelu_3_16")

    % convolution2dLayer(1,512,'Stride',1,"Name","conv_1_17","Padding","same")
    % batchNormalizationLayer("Name","batchnorm_1_17")
    % leakyReluLayer(0.1,"Name","leakyRelu_1_17")

    convolution2dLayer(8,512,'Stride',1,"Name","conv_3_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3_18")
    leakyReluLayer(0.1,"Name","leakyRelu_3_18")

    convolution2dLayer(1,2,'Stride',1,"Name","conv_3_19","Padding","same")

    averagePooling2dLayer(7,'Stride',1,"Name","avgpool_3")
    ];
lgraph = addLayers(lgraph,layers3);
lgraph = connectLayers(lgraph,'imageinput','conv_3_1');
lgraph = connectLayers(lgraph,'avgpool_3','concat1/in3');
% % lgraph = connectLayers(lgraph,'pool_3_1','add_3/in2');
% clear layers2;
% clear layers3;
analyzeNetwork(lgraph)
%%
%指定训练选项adam,sgdm
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'MiniBatchSize',10,...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',62, ...
    'Verbose',false, ...
    'Plots','training-progress');
%使用训练数据训练网络
[net,info] = trainNetwork(imdsTrain,lgraph,options);%trainNetwork(imdsTrain,layers,options)
%%
%Classify Validation Images
[YPred,scores] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)