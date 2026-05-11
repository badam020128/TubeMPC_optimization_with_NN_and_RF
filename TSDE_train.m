% =========================================================================
% DUAL-STAGE DEEP ENSEMBLE TANÍTÁS (15-es bemenet, NÖVELT KAPACITÁS)
% =========================================================================
disp('=== Kétlépcsős Deep Ensemble Tanítása Indul ===');

num_nets = 10; 
ensemble_mean = cell(num_nets, 1);
ensemble_sigma = cell(num_nets, 1);

XTrain_dl = dlarray(Training_Inputs', 'CB');
YTrain_dl = dlarray(Training_Outputs', 'CB');

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

% Visszatérünk az abszolút hiba (szórás) közvetlen tanulásához!
% A négyzetre emelés túlságosan "eltüntette" a parányi hibákat.
YTrain_sigma_dl = dlarray(abs((Training_Outputs - preds_mean)') * 100.0, 'CB');

% --- Architektúra B hálózatoknak (15 BEMENET) ---
layers_sigma = [
    featureInputLayer(15, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    leakyReluLayer(0.01) % A Leaky ReLU átenged pici negatív gradienst, így nem "hal meg" a hálózat
    fullyConnectedLayer(64)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(32)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(4)
    reluLayer % A LÉNYEG: a legutolsó réteg sima ReLU, hogy a kimenet szigorúan >= 0 maradjon!
    ];

disp('--> 2. Lépcső: Ensemble B (Bizonytalanság) tanítása...');
for i = 1:num_nets
    fprintf('    B-Hálózat %d / %d...\n', i, num_nets);
    ensemble_sigma{i} = trainnet(XTrain_dl, YTrain_sigma_dl, layers_sigma, "mse", options);
end
disp('=== TANÍTÁS KÉSZ! ===');
