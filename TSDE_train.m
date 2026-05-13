% =========================================================================
% DUAL-STAGE DEEP ENSEMBLE TANÍTÁS: LSTM + BAGGING + PINBALL LOSS
% =========================================================================
disp('=== Kétlépcsős LSTM Deep Ensemble Tanítása Indul ===');

num_nets = 10; 
ensemble_mean = cell(num_nets, 1);
ensemble_sigma = cell(num_nets, 1);
total_samples = size(Training_Inputs, 1);

% --- ADATOK ELŐKÉSZÍTÉSE LSTM-HEZ ---
X_Seq_3D = zeros(7, total_samples, 3);

for i = 1:total_samples
    x_now = Training_Inputs(i, 1:4)';
    x_h1  = Training_Inputs(i, 5:8)';
    x_h2  = Training_Inputs(i, 9:12)';
    u_p   = Training_Inputs(i, 13:14)';
    kap   = Training_Inputs(i, 15);
    
    % NINCS TÖBB NULLA! Bemásoljuk az 'u_p' és 'kap' értékeket a múltba is,
    % így az LSTM érti, hogy a múltbeli elmozdulás is emiatt a kormányszög miatt volt!
    X_Seq_3D(:, i, 1) = [x_h2; u_p; kap];
    X_Seq_3D(:, i, 2) = [x_h1; u_p; kap];
    X_Seq_3D(:, i, 3) = [x_now; u_p; kap];
end

% Konvertálás dlarray-be explicit 'CBT' (Channel, Batch, Time) címkékkel
X_Seq_dl = dlarray(X_Seq_3D, 'CBT');

% Célváltozók előkészítése 'CB' (Channel, Batch) címkékkel
Y_Mean_dl = dlarray(Training_Outputs', 'CB'); % 4 x N

% --- Architektúra A (Átlag-kompenzáció LSTM-el) ---
layers = [
    sequenceInputLayer(7) 
    lstmLayer(64, 'OutputMode', 'last') 
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(4) 
    ];

options = trainingOptions('adam', 'MaxEpochs', 150, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.005, 'Verbose', false, 'Plots', 'none');

disp('--> 1. Lépcső: Ensemble A (LSTM Átlag) tanítása Bagging-el...');
for i = 1:num_nets
    fprintf('    A-LSTM %d / %d...\n', i, num_nets);
    
    % Bagging
    idx = randsample(total_samples, total_samples, true); 
    
    % Indexelés a Batch (2.) dimenzió mentén!
    X_bag = X_Seq_dl(:, idx, :);
    Y_bag = Y_Mean_dl(:, idx);
    
    ensemble_mean{i} = trainnet(X_bag, Y_bag, layers, "mse", options);
end

% Hiba kiszámítása a B lépcsőhöz
preds_mean = zeros(4, total_samples);
for i = 1:num_nets
    % predict egy 4xN mátrixot (CB) ad vissza dlarray-ként
    preds_mean = preds_mean + double(extractdata(predict(ensemble_mean{i}, X_Seq_dl)));
end
preds_mean = preds_mean / num_nets;

% Felszorzás 100-zal, a kimenet szintén 'CB' (4 x N)
Y_Sigma_dl = dlarray((Training_Outputs' - preds_mean) * 100.0, 'CB'); 

% --- Architektúra B (Bizonytalanság/Csőméret LSTM-el) ---
layers_sigma = [
    sequenceInputLayer(7)
    lstmLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(4)
    ];

disp('--> 2. Lépcső: Ensemble B (LSTM 95% Bound) tanítása Pinball Loss-al...');
for i = 1:num_nets
    fprintf('    B-LSTM %d / %d...\n', i, num_nets);
    
    idx = randsample(total_samples, total_samples, true); 
    
    X_bag = X_Seq_dl(:, idx, :);
    Y_sigma_bag = Y_Sigma_dl(:, idx);
    
    ensemble_sigma{i} = trainnet(X_bag, Y_sigma_bag, layers_sigma, @pinball_loss, options);
end
disp('=== LSTM TANÍTÁS KÉSZ! ===');

function loss = pinball_loss(YPred, YTrue)
    % 0.95 helyett 0.90: Ezzel a cső (tube) mérete szűkebb lesz, 
    % az MPC pedig bátrabban tud majd a referenciához tapadni.
    q = 0.90; 
    error = abs(YTrue) - YPred;
    loss_matrix = max(q * error, (q - 1) * error);
    loss = mean(sum(loss_matrix, 1));
end