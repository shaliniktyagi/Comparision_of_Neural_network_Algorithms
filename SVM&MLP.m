% -------------------------------------------------------------------------
% LOAD THE DATA AND PREPARE TRAINING AND TEST DATASETS
% -------------------------------------------------------------------------

% read the dataset
data = readtable('normdata2.xls');
summary(data);

%dividing the data set into 70% training, and 30% test data
rng(3);
[M,N] = size(data);
P = 0.70;
Rand = randperm(M);
Training = data(Rand(1:round(P*M)),:); 
Test = data(Rand(round(P*M)+1:end),:);
Parameters = [];
Parameters_all=[];
Errors = [] ;
Error_all=[];
Accuracy = [] ;
Accuracy_all=[];
Results = [] ;
Results_all=[];


n=size(Training,1);
k = 10
cv = cvpartition(n,'kfold',k);
tic
for K = 1:k
    train_idx = training(cv,K);
    train_data = Training(train_idx,:);
    val_idx = test(cv,K);
    val_data = Training(val_idx,:);
  % split the  training data in to input and target
    x = train_data{:,1:9};
    y = train_data{:,10:11};
    
                                     
    %Split the validation data in to inputs and target
    val_inputs = (val_data{:,1:9})';
    val_target = (val_data{:,10:11})';
    class_label= (val_data{:,12})';
    
      
      % initialise parameters for the grid search 
     alpha = 0.1:0.2:1; % Learning rate from 0.1 to 1
     momentum = 0.1:0.2:1; % Momentum rate from 0.1 to 1
     hidden_neurons = 2:4:20; % Hidden neurons from 2 to 20
     for i = 1: length(alpha)
         for j = 1: length(momentum)
             for l = 1: length(hidden_neurons)
                % parameters of each combination are stored in an array

                  Parameters = [Parameters; alpha(i), momentum(j), hidden_neurons(l)];
                  net = patternnet(hidden_neurons(l),'traingdx');
                  net.trainParam.lr = alpha(i);
                  net.trainParam.mc = momentum(j);
                   % TRAINING THE MODEL
                  net = train(net, x', y');
                  y_pred = net(val_inputs);
                  error = perform(net,val_target,y_pred);
                  Errors = [Errors; error];
                  % plot a confusion matrix
                  plotconfusion(val_target,y_pred)
                  [~, y_pred] = max(y_pred);
                  accuracy=(sum(class_label == y_pred) / length(class_label))*100;
                  Accuracy = [Accuracy; accuracy];
                  Results = [Parameters Accuracy Errors] ;
              end
         end
     end
     Error_all= Errors;
     Accuracy_all= Accuracy ;
     Results_all =  Results;
     Parameters_all = Parameters;
end
toc

Error_all = array2table(Error_all);
Accuracy_all = array2table(Accuracy_all);
Parameters_all = array2table(Parameters_all)
Errors_mean = array2table(mean(Error_all{:,:},2))
Accuracy_mean = array2table(mean(Accuracy_all{:,:},1));
FinalValue = [Parameters_all(:,1:3) Errors_mean]


FinalValue.Properties.VariableNames = {'Momentum', 'Alpha' , 'Hiddenlayers', 'ErrorValue'} ;
min_Error = min(FinalValue{:,4})

%% BEST MODEL CREATION

best_model = FinalValue(FinalValue.ErrorValue == min_Error, :)
