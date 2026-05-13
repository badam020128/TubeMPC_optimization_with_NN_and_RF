% =========================================================================
% DUAL-STAGE DEEP ENSEMBLE TANÍTÁS BAGGING-EL ÉS KVANTILIS REGRESSZIÓVAL
% =========================================================================
disp('=== Kétlépcsős Deep Ensemble Tanítása Indul (Bagging + Pinball Loss) ===');

num_nets = 10; 
ensemble_mean = cell(num_nets, 1);
ensemble_sigma = cell(num_nets, 1);

total_samples = size(Training_Inputs, 1);

% --- Architektúra A hálózatoknak (Átlag kompenzáció) ---
layers = [
    featureInputLayer(15, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    reluLayer
    fullyConnectedLayer(64)  
    reluLayer
    fullyConnectedLayer(32)  
    reluLayer
    fullyConnectedLayer(4) 
    ];

options = trainingOptions('adam', 'MaxEpochs', 200, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.005, 'Verbose', false, 'Plots', 'none');

disp('--> 1. Lépcső: Ensemble A (Átlag-kompenzáció) tanítása Bagging-el...');
for i = 1:num_nets
    fprintf('    A-Hálózat %d / %d...\n', i, num_nets);
    
    % BAGGING: Visszatevéses mintavétel
    idx = randsample(total_samples, total_samples, true); 
    XTrain_bag = dlarray(Training_Inputs(idx, :)', 'CB');
    YTrain_bag = dlarray(Training_Outputs(idx, :)', 'CB');
    
    ensemble_mean{i} = trainnet(XTrain_bag, YTrain_bag, layers, "mse", options);
end

disp('--> Köztes lépés: Az ensemble hiba kiszámítása a teljes halmazon...');
XTrain_Full_dl = dlarray(Training_Inputs', 'CB');
preds_mean = zeros(size(Training_Outputs));
for i = 1:num_nets
    preds_mean = preds_mean + double(extractdata(predict(ensemble_mean{i}, XTrain_Full_dl)))';
end
preds_mean = preds_mean / num_nets;

% A hiba kiszámítása
residual_errors = Training_Outputs - preds_mean; 
% Felszorzás 100-zal a numerikus stabilitás miatt (tesztelésnél visszaosztjuk)
YTrain_sigma = residual_errors * 100.0; 

% --- Architektúra B hálózatoknak (Worst-case határ becslése) ---
layers_sigma = [
    featureInputLayer(15, 'Normalization', 'zscore') 
    fullyConnectedLayer(128) 
    leakyReluLayer(0.01)
    fullyConnectedLayer(64)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(32)  
    leakyReluLayer(0.01)
    fullyConnectedLayer(4)
    ];

disp('--> 2. Lépcső: Ensemble B (95%-os Bizonytalansági korlát) tanítása Pinball Loss-szal...');
for i = 1:num_nets
    fprintf('    B-Hálózat %d / %d...\n', i, num_nets);
    
    % BAGGING itt is
    idx = randsample(total_samples, total_samples, true); 
    XTrain_sigma_bag = dlarray(Training_Inputs(idx, :)', 'CB');
    YTrain_sigma_bag = dlarray(YTrain_sigma(idx, :)', 'CB');
    
    % ÚJÍTÁS: Itt a "mse" helyett a @pinball_loss custom függvényt használjuk!
    ensemble_sigma{i} = trainnet(XTrain_sigma_bag, YTrain_sigma_bag, layers_sigma, @pinball_loss, options);
end
disp('=== TANÍTÁS KÉSZ! ===');


% =========================================================================
% SEGÉDFÜGGVÉNY: Kvantilis Regresszió (Pinball Loss)
% =========================================================================
function loss = pinball_loss(YPred, YTrue)
    % A 95%-os kvantilis keresése (q = 0.95). 
    % A hálózat a legrosszabb esetek (worst-case) felső határát fogja megtanulni.
    q = 0.95; 
    
    % Mivel felső határt keresünk, a hibák abszolút értékén dolgozunk
    % Így a negatív és pozitív kilengésekre is szimmetrikus felső burkológörbét kapunk
    error = abs(YTrue) - YPred;
    
    % Pinball Loss: max(q * error, (q - 1) * error)
    loss_matrix = max(q * error, (q - 1) * error);
    
    % Átlagolás a batchen
    loss = mean(sum(loss_matrix, 1));
end