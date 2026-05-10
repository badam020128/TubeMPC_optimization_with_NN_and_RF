% =========================================================================
% DUAL-STAGE DEEP ENSEMBLE TANÍTÁS (14-es bemenet, NÖVELT KAPACITÁS)
% =========================================================================
disp('=== Kétlépcsős Deep Ensemble Tanítása Indul ===');

num_nets = 5; 
ensemble_mean = cell(num_nets, 1);
ensemble_sigma = cell(num_nets, 1);

XTrain_dl = dlarray(Training_Inputs', 'CB');
YTrain_dl = dlarray(Training_Outputs', 'CB');

% --- Architektúra A hálózatoknak (14 BEMENET, 128 -> 64 -> 32) ---
layers = [
    featureInputLayer(14, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    tanhLayer
    fullyConnectedLayer(64)  
    tanhLayer
    fullyConnectedLayer(32)  
    tanhLayer
    fullyConnectedLayer(4) 
    ];

% --- Tanítási paraméterek (MiniBatchSize 512-re emelve a gyorsabb tanításért) ---
options = trainingOptions('adam', 'MaxEpochs', 200, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.005, 'Verbose', false, 'Plots', 'none');

disp('--> 1. Lépcső: Ensemble A (Átlag-kompenzáció) tanítása...');
for i = 1:num_nets
    fprintf('    A-Hálózat %d / %d...\n', i, num_nets);
    ensemble_mean{i} = trainnet(XTrain_dl, YTrain_dl, layers, "mse", options);
end

disp('--> Köztes lépés: Az aleatorikus zaj (turbulencia) kiszámítása...');
preds_mean = zeros(size(Training_Outputs));
for i = 1:num_nets
    preds_mean = preds_mean + double(extractdata(predict(ensemble_mean{i}, XTrain_dl)))';
end
preds_mean = preds_mean / num_nets;

% Mivel a hiba mikroszkopikus, a hálózat nem tud belőle tanulni. 
% A 100-szoros szorzó "felébreszti" a gradienseket.
YTrain_sigma_dl = dlarray(abs(Training_Outputs - preds_mean)' * 100.0, 'CB');

% --- Architektúra B hálózatoknak (14 BEMENET, 128 -> 64 -> 32) ---
layers_sigma = [
    featureInputLayer(14, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    tanhLayer
    fullyConnectedLayer(64)  
    tanhLayer
    fullyConnectedLayer(32)  
    tanhLayer
    fullyConnectedLayer(4)
    ];

disp('--> 2. Lépcső: Ensemble B (Bizonytalanság) tanítása...');
for i = 1:num_nets
    fprintf('    B-Hálózat %d / %d...\n', i, num_nets);
    ensemble_sigma{i} = trainnet(XTrain_dl, YTrain_sigma_dl, layers_sigma, "mse", options);
end
disp('=== TANÍTÁS KÉSZ! ===');
