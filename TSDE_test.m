% =========================================================================
% ÉLES TESZTELÉS (Self-Tuning Dual-Stage Deep Ensemble AI)
% =========================================================================
import casadi.*
disp('=== TESZTELÉS INDÍTVA: Klasszikus vs. Önhangoló Hibrid AI ===');

if ~exist('ensemble_mean', 'var') || ~exist('ensemble_sigma', 'var')
    error('Hiba: Nem találom a hálózatokat!');
end
num_nets = length(ensemble_mean);

% Pálya (1 kör teszt)
egy_kor_pontjai = smooth_path(1:end-1, :); 
multi_lap_path = [egy_kor_pontjai; smooth_path(end, :)];
Ts = 0.1; V_target = 5; N = 10; 

% Saját dist és t_path számítása 1 körre! ---
ds = sqrt(diff(multi_lap_path(:,1)).^2 + diff(multi_lap_path(:,2)).^2);
dist_test = [0; cumsum(ds)]; 
t_path_test = dist_test / V_target; 
t_sim_test = 0:Ts:t_path_test(end); 

X_ref = interp1(t_path_test, multi_lap_path(:,1), t_sim_test, 'linear', 'extrap');
Y_ref = interp1(t_path_test, multi_lap_path(:,2), t_sim_test, 'linear', 'extrap');
Vx_ref_test = [diff(X_ref)/Ts, 0];
Vy_ref_test = [diff(Y_ref)/Ts, 0];

% --- ÚJ: Teszt pálya görbületének kiszámítása ---
Ax_ref_test = [diff(Vx_ref_test)/Ts, 0];
Ay_ref_test = [diff(Vy_ref_test)/Ts, 0];
kappa_ref_test = (Vx_ref_test .* Ay_ref_test - Vy_ref_test .* Ax_ref_test) ./ max((Vx_ref_test.^2 + Vy_ref_test.^2).^(3/2), 1e-6);

RefMatrix = [X_ref', Y_ref', Vx_ref_test', Vy_ref_test'];

% Dinamika és LQR (Azonos a tanítóval!)
A = [1 0 Ts 0; 0 1 0 Ts; 0 0 1 0; 0 0 0 1]; B = [0 0; 0 0; Ts 0; 0 Ts];
% Eredeti:
% K_lqr = dlqr(A, B, diag([50, 50, 5, 5]), diag([0.5, 0.5]));

% Próbáld meg ezt (jobban odarántja az autót a prediktált z_nom állapotokhoz):
K_lqr = dlqr(A, B, diag([150, 150, 10, 10]), diag([0.1, 0.1]));

% CasADi Felépítése Self-Tuning Paraméterrel
opti = casadi.Opti();
X = opti.variable(4, N+1); U = opti.variable(2, N);   
x0_param = opti.parameter(4, 1); ref_param = opti.parameter(4, N+1);      
w_nn_param = opti.parameter(4, 1); tube_uncertainty_param = opti.parameter(1, 1); 
q_mult_param = opti.parameter(1, 1); % SELF-TUNING SZORZÓ

% Az X és Y pozíció büntetését felemeljük 200-ra, és adunk a sebességeknek (Vx, Vy) is 5-5 súlyt.
Q_mpc_base = diag([200, 200, 5, 5]); 
R_mpc_base = diag([0.1, 0.1]); % Az R (beavatkozó jel büntetése) maradhat alacsony     
cost = 0; max_y_elteres = 2.0; 

% Definiáljunk egy R_delta_mpc mátrixot a cikluson kívül
R_delta_mpc = diag([1.0, 1.0]); % A változás büntetése

for k = 1:N
    err = X(:, k) - ref_param(:, k);
    aktualis_R = R_mpc_base / sqrt(q_mult_param);
    
    % Alap költség
    cost = cost + q_mult_param * (err' * Q_mpc_base * err) + U(:, k)' * aktualis_R * U(:, k);
    
    % ÚJ: Kormányzás változásának (Delta U) büntetése!
    if k == 1
        % Az első lépésnél az előző időlépés (u_prev) beavatkozásához képest nézzük a változást
        % Ehhez fel kell venned egy új paramétert a CasADiban: u_prev_param = opti.parameter(2,1);
        % cost = cost + (U(:, k) - u_prev_param)' * R_delta_mpc * (U(:, k) - u_prev_param);
    else
        % A többi lépésnél az előző horizon-pontbeli beavatkozáshoz képest
        delta_u = U(:, k) - U(:, k-1);
        cost = cost + delta_u' * R_delta_mpc * delta_u;
    end
    
    opti.subject_to(X(:, k+1) == A * X(:, k) + B * U(:, k) + w_nn_param);
    akt_cs_szelesseg = max_y_elteres - tube_uncertainty_param;
    opti.subject_to( -akt_cs_szelesseg <= err(1:2) <= akt_cs_szelesseg );
end
cost = cost + q_mult_param * ((X(:, N+1) - ref_param(:, N+1))' * Q_mpc_base * (X(:, N+1) - ref_param(:, N+1)));
opti.minimize(cost); opti.subject_to(X(:, 1) == x0_param); opti.subject_to(-20 <= U <= 20); 
opti.solver('ipopt', struct('expand', true), struct('max_iter', 100, 'print_level', 0, 'acceptable_tol', 1e-6));

n_steps_test = length(t_sim_test) - N - 1;
rng(100); zaj_alap_test = randn(2, n_steps_test); 

history_cl = zeros(n_steps_test, 4); errors_cl = zeros(n_steps_test, 1);
history_ai = zeros(n_steps_test, 4); errors_ai = zeros(n_steps_test, 1);
tube_history = zeros(n_steps_test, 1); q_mult_history = zeros(n_steps_test, 1);

for mode = 1:2
    x_real = RefMatrix(1,:)'; z_nom = x_real; u_prev = [0; 0];
    szel_memoria = [0;0]; w_smoothed = [0; 0; 0; 0]; 
    x_hist1 = x_real; x_hist2 = x_real; % Csúszóablak pufferek
    
    for k = 1 : n_steps_test
        current_ref = RefMatrix(k : k+N, :)';
        if mode == 1
            w_becsult = [0; 0; 0; 0]; margin_fizikai = 0.0; current_q_mult = 1.0; 
        else
            current_kappa = kappa_ref_test(k);
            nn_input = dlarray([x_real; x_hist1; x_hist2; u_prev; current_kappa], 'CB'); 
            
            tippek_mean = zeros(4, num_nets); tippek_sigma = zeros(4, num_nets);
            for i = 1:num_nets
                tippek_mean(:, i) = double(extractdata(predict(ensemble_mean{i}, nn_input)));
                tippek_sigma(:, i) = double(extractdata(predict(ensemble_sigma{i}, nn_input)));
            end
            
            % --- VISSZAOSZTJUK A 100-AS SZORZÓT ---
            w_zaj_mertek = mean(tippek_sigma, 2) / 100.0; 
            
            % --- ÚJ: VALÓS IDEJŰ HIBA VISSZACSATOLÁSA (DINAMIKUS CSŐ) ---
            % Kiszámoljuk az éppen aktuális eltérést a nominális állapottól (e_k)
            % Ez felel meg a cikk szerinti dinamikus csőfrissítésnek!
            aktualis_hiba = norm(x_real(1:2) - z_nom(1:2));
            
            alap_zaj_szoras = 0.02; % 2 cm-es alapzaj
            
            % A teljes bizonytalanság: az NN jóslata + alapzaj + az aktuális fizikai megcsúszás
            % Ha az autó letér az ívről, az aktualis_hiba megnő, és a cső "kifújja magát"!
            sigma_max = max(w_zaj_mertek(1:2)) + alap_zaj_szoras + (aktualis_hiba * 0.15); 
            
            w_raw = mean(tippek_mean, 2); 
            % Eredeti: alpha_ema = max(0.02, 0.9 - (sigma_max * 10.0));
            % ÚJ: Sokkal gyorsabb reakció az AI részéről!
            alpha_ema = max(0.1, 1.0 - (sigma_max * 2.0)); 

            w_smoothed = (1 - alpha_ema) * w_smoothed + alpha_ema * w_raw; 

            % TÚLKOMPENZÁLÁS: Mivel az MPC kicsit lassan reagál a belső tehetetlenség miatt, 
            % szorozzuk fel a prediktált szél/hiba hatást 1.1-gyel (10% proaktív túlkormányzás)!
            w_becsult = max(min(w_smoothed * 1.1, 0.5), -0.5); 
            
            % 3 helyett 2-szigma is elég lehet (95% konfidencia), és vegyük ki az 1.5-ös szorzót.
            d_max = 2.0 * sigma_max; 
            margin_fizikai = min(d_max, max_y_elteres * 0.90); 

            % Ha ezt megléped, az autó sokkal többször fogja a current_q_mult-ot a maximális
            % közelébe tolni, mert a cső indokolatlanul nem fog felfújódni.
            biztonsagi_tenyezo = (max_y_elteres - margin_fizikai) / max_y_elteres;
            current_q_mult = 1.0 + (biztonsagi_tenyezo^2) * 20.0; % Mehet akár 20-as szorzó is!
            
            tube_history(k) = margin_fizikai; 
            q_mult_history(k) = current_q_mult;
        end
        
        opti.set_value(x0_param, z_nom); opti.set_value(ref_param, current_ref);
        opti.set_value(w_nn_param, w_becsult); 
        opti.set_value(tube_uncertainty_param, margin_fizikai); % <-- Itt adjuk át a fizikai margót!
        opti.set_value(q_mult_param, current_q_mult); 
        
        try, sol = opti.solve(); v_k_opt = sol.value(U(:, 1)); catch, v_k_opt = opti.debug.value(U(:, 1)); end
        
        u_k = max(min(v_k_opt - K_lqr * (x_real - z_nom), 50), -50); 
        
        % Fizika
        drag_x = -0.02 * x_real(3) * abs(x_real(3)) * Ts; drag_y = -0.02 * x_real(4) * abs(x_real(4)) * Ts;
        spatial_x = 0.5 * sin(x_real(1)/15); spatial_y = 0.5 * cos(x_real(2)/15);
        zona_szorzo = 0.1 + 1.4 * (0.5 + 0.5 * tanh((x_real(1)-40)/5));
        szel_memoria(1) = 0.8 * szel_memoria(1) + zaj_alap_test(1, k)*0.4;
        szel_memoria(2) = 0.8 * szel_memoria(2) + zaj_alap_test(2, k)*0.4;
        w_k = [0; 0; (0.8*Ts) + drag_x + (spatial_x + szel_memoria(1))*zona_szorzo; 
                     (0.5*Ts) + drag_y + (spatial_y + szel_memoria(2))*zona_szorzo];
        
        x_real_new = A * x_real + B * u_k + w_k;
        z_nom = A * z_nom + B * v_k_opt + w_becsult; 
        
        % Csúszóablak frissítése
        x_hist2 = x_hist1; x_hist1 = x_real; u_prev = u_k; x_real = x_real_new;
        
        hiba = min(sqrt((smooth_path(:,1)-x_real(1)).^2 + (smooth_path(:,2)-x_real(2)).^2));
        if mode == 1, history_cl(k, :) = x_real'; errors_cl(k) = hiba; 
        else, history_ai(k, :) = x_real'; errors_ai(k) = hiba; end
    end
end

disp('=== SZIMULÁCIÓK KÉSZ. Ábrázolás... ===');
figure('Name', 'Self-Tuning AI MPC', 'Position', [50, 50, 1600, 600], 'Color', 'k');

ax1 = subplot(1, 4, 1); hold on; grid on;
patch([min(smooth_path(:,1))-5, 40, 40, min(smooth_path(:,1))-5], [min(smooth_path(:,2))-5, min(smooth_path(:,2))-5, max(smooth_path(:,2))+5, max(smooth_path(:,2))+5], [0.1 0.4 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([40, max(smooth_path(:,1))+5, max(smooth_path(:,1))+5, 40], [min(smooth_path(:,2))-5, min(smooth_path(:,2))-5, max(smooth_path(:,2))+5, max(smooth_path(:,2))+5], [0.6 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot(smooth_path(:,1), smooth_path(:,2), '-g', 'LineWidth', 3); plot(history_cl(:,1), history_cl(:,2), 'b--', 'LineWidth', 1.5); plot(history_ai(:,1), history_ai(:,2), 'r-', 'LineWidth', 1.5);
title('Útvonal', 'Color', 'w'); axis equal; set(ax1, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');

ax2 = subplot(1, 4, 2); hold on; grid on;
plot(t_sim_test(1:n_steps_test), errors_cl, 'b--', 'LineWidth', 1.5); plot(t_sim_test(1:n_steps_test), errors_ai, 'r-', 'LineWidth', 1.5);
title('Hiba', 'Color', 'w'); set(ax2, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');

ax3 = subplot(1, 4, 3); hold on; grid on;
plot(t_sim_test(1:n_steps_test), tube_history, 'y-', 'LineWidth', 1.5); 
title('Csőszűkítés [m]', 'Color', 'w'); 
axis tight; % <--- EZT ÁLLÍTSD BE! (A fix ylim helyett ez ránagyít a pontos értékekre)
set(ax3, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
ax4 = subplot(1, 4, 4); hold on; grid on;
plot(t_sim_test(1:n_steps_test), q_mult_history, 'm-', 'LineWidth', 1.5); title('MPC Agresszivitás (Q Szorzó)', 'Color', 'w'); set(ax4, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');

mean_cl = mean(errors_cl); mean_ai = mean(errors_ai);
fprintf('\n=== VÉGSŐ EREDMÉNYEK ===\nKlasszikus: %.4f m\nAI (TSDE): %.4f m\nJAVULÁS: +%.1f %%\n', mean_cl, mean_ai, (1 - (mean_ai / mean_cl)) * 100);
