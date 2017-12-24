%% Written by Peter Sarvari, 2017
% Imperial College, London, ID: 00987075


load Activities.mat

%% Multiclass classification

params = TrainClassifierX(train_data, train_labels);
[pred, res] = ClassifyX(test_data, params);

disp('Accuracy:')
sum(pred==test_labels)/length(test_labels)

% Confusion matrix

figure(1)

res_matrix = [test_labels, pred];
confusion = zeros(4,4);
for i = 1:size(res_matrix,1)
    y = res_matrix(i,1);
    x = res_matrix(i,2);
    confusion(y,x) = confusion(y,x) + 1;
end
accurate = trace(confusion);
all = sum(sum(confusion));
MCR = (all-accurate)/(all); %Misclassification rate
confusion = confusion/length(test_labels); 
imagesc(confusion); colorbar
title(['Normalized confusion matrix for the generative model, MCR = ', num2str(MCR)], 'FontSize', 20)
xlabel('Predicted classes', 'FontSize', 16)
ylabel('Actual classes', 'FontSize', 16)

figure(2)
%Pairwise confusion matrix from the results of the multiclass
%classification to estimate which classes can be separated the easiest
for i = 1:3
    for j = i+1:4 %iterating through all possible pairwise combinations
        if i == 1
            subplot(2,3,i+j-2)
        else 
            subplot(2,3,i+j-1)
        end
        choice = confusion([i j], [i j]);
        accurate = trace(choice);
        fail = choice(1,2) + choice(2,1);
        MCR = fail/(fail+accurate); %Misclassification rate
        precision = choice(1,1)/(choice(1,1)+choice(2,1));
        recall = choice(1,1)/(choice(1,1)+choice(1,2));
        F1 = 2*precision*recall / (precision+recall); %F1-score
        imagesc(choice)
        caxis([0 0.2]) %unified color axis
        colorbar;
        title([num2str(i), '&', num2str(j), ': MCR = ', num2str(MCR) ' F1 = ' num2str(F1)], 'FontSize', 20)
        xlabel('Predicted classes', 'FontSize', 16)
        ylabel('Actual classes', 'FontSize', 16)
    end
end

pause;
%% Binary classification
%Choosing class 2 and class 3 for binary classification
train_run = train_data(train_labels==2, :);
train_walkup = train_data(train_labels==3, :);
test_run = test_data(test_labels==2, :);
test_walkup = test_data(test_labels==3, :);
train_bin_labels = [ones(size(train_run,1),1);2*ones(size(train_walkup,1),1)];
test_bin_labels = [ones(size(test_run,1),1);2*ones(size(test_walkup,1),1)];
train_bin_data = [train_run; train_walkup];
test_bin_data = [test_run; test_walkup];

param_bin = TrainClassifierX(train_bin_data, train_bin_labels);
[pred_bin, res_bin] = ClassifyX(test_bin_data, param_bin);

disp('Accuracy:')
sum(pred_bin==test_bin_labels)/length(test_bin_labels)

% Confusion matrix

figure(3)

res_matrix = [test_bin_labels, pred_bin];
confusion = zeros(2,2);
for i = 1:size(res_matrix,1)
    y = res_matrix(i,1);
    x = res_matrix(i,2);
    confusion(y,x) = confusion(y,x) + 1;
end
confusion = confusion/size(test_bin_labels,1); 
imagesc(confusion); colorbar

accurate = trace(confusion);
fail = confusion(1,2) + confusion(2,1);
MCR = fail/(fail+accurate); %Misclassification rate
precision = confusion(1,1)/(confusion(1,1)+confusion(2,1));
recall = confusion(1,1)/(confusion(1,1)+confusion(1,2));
F1 = 2*precision*recall / (precision+recall); %F1-score

title(['Normalized confusion matrix for binary class (2 and 3) prediction', ': MCR = ', num2str(MCR) ' F1 = ' num2str(F1)], 'FontSize', 20)
xlabel('Predicted classes', 'FontSize', 16)
ylabel('Actual classes', 'FontSize', 16)