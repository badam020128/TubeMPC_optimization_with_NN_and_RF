% =========================================================================
% DUAL-STAGE DEEP ENSEMBLE TANÍTÁS (15-es bemenet, BAGGING-EL)
% =========================================================================
disp('=== Kétlépcsős Deep Ensemble Tanítása Indul (Bagging) ===');

num_nets = 10; 
ensemble_mean = cell(num_nets, 1);
ensemble_sigma = cell(num_nets, 1);

% Adatpontok számának meghatározása
total_samples = size(Training_Inputs, 1);

% --- Architektúra A hálózatoknak (15 BEMENET, 128 -> 64 -> 32) ---
layers = [
    featureInputLayer(15, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    tanhLayer
    fullyConnectedLayer(64)  
    tanhLayer
    fullyConnectedLayer(32)  
    tanhLayer
    fullyConnectedLayer(4) 
    ];

% --- Tanítási paraméterek ---
options = trainingOptions('adam', 'MaxEpochs', 200, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.005, 'Verbose', false, 'Plots', 'none');

disp('--> 1. Lépcső: Ensemble A (Átlag-kompenzáció) tanítása...');
for i = 1:num_nets
    fprintf('    A-Hálózat %d / %d...\n', i, num_nets);
    
    % --- BAGGING LÉPÉS ---
    % Visszatevéses mintavétel generálása
    idx = randsample(total_samples, total_samples, true); 
    
    % Egyedi tanítóhalmaz létrehozása az i-edik hálózat számára
    XTrain_bag_dl = dlarray(Training_Inputs(idx, :)', 'CB');
    YTrain_bag_dl = dlarray(Training_Outputs(idx, :)', 'CB');
    
    ensemble_mean{i} = trainnet(XTrain_bag_dl, YTrain_bag_dl, layers, "mse", options);
end

disp('--> Köztes lépés: Az aleatorikus zaj (turbulencia) kiszámítása...');
% A hiba kiszámításához a teljes halmazt használjuk!
XTrain_Full_dl = dlarray(Training_Inputs', 'CB');
preds_mean = zeros(size(Training_Outputs));

for i = 1:num_nets
    preds_mean = preds_mean + double(extractdata(predict(ensemble_mean{i}, XTrain_Full_dl)))';
end
preds_mean = preds_mean / num_nets;

YTrain_sigma = abs((Training_Outputs - preds_mean)') * 100.0;

% --- Architektúra B hálózatoknak (15 BEMENET) ---
layers_sigma = [
    featureInputLayer(15, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    leakyReluLayer(0.01) 
    fullyConnectedLayer(64)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(32)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(4)
    reluLayer 
    ];

disp('--> 2. Lépcső: Ensemble B (Bizonytalanság) tanítása...');
for i = 1:num_nets
    fprintf('    B-Hálózat %d / %d...\n', i, num_nets);
    
    % --- BAGGING LÉPÉS ---
    idx = randsample(total_samples, total_samples, true); 
    
    XTrain_sigma_bag_dl = dlarray(Training_Inputs(idx, :)', 'CB');
    YTrain_sigma_bag_dl = dlarray(YTrain_sigma(:, idx), 'CB'); % Figyelem: YTrain_sigma itt 4xN méretű!
    
    ensemble_sigma{i} = trainnet(XTrain_sigma_bag_dl, YTrain_sigma_bag_dl, layers_sigma, "mse", options);
end
disp('=== TANÍTÁS KÉSZ! ===');