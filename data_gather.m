% =========================================================================
% ADATGYŰJTÉS (Brute-Force: 20 körös masszív adathalmaz)
% =========================================================================
import casadi.*
disp('=== 1. FÁZIS: CasADi MPC Szimuláció és Adatgyűjtés Indul ===');

% --- BRUTE-FORCE ADATGYŰJTÉS ---
korok_szama = 10;
egy_kor_pontjai = smooth_path(1:end-1, :); 
multi_lap_path = repmat(egy_kor_pontjai, korok_szama, 1);
multi_lap_path = [multi_lap_path; smooth_path(end, :)];

% --- VÉLETLENSZERŰSÉG BIZTOSÍTÁSA ---
% Fontos: Minden futtatásnál más legyen a széllökés alapja
rng('shuffle');
Ts = 0.1; 
V_target = 3.5; 
N = 10; 

ds = sqrt(diff(multi_lap_path(:,1)).^2 + diff(multi_lap_path(:,2)).^2);
dist = [0; cumsum(ds)];
t_path = dist / V_target; 
t_sim = 0:Ts:t_path(end); 

X_ref = interp1(t_path, multi_lap_path(:,1), t_sim, 'linear', 'extrap');
Y_ref = interp1(t_path, multi_lap_path(:,2), t_sim, 'linear', 'extrap');
Vx_ref = [diff(X_ref)/Ts, 0]; 
Vy_ref = [diff(Y_ref)/Ts, 0];
% --- ÚJ: Görbület (kappa) pontos analitikus kiszámítása ---
Ax_ref = [diff(Vx_ref)/Ts, 0];
Ay_ref = [diff(Vy_ref)/Ts, 0];
kappa_ref = (Vx_ref .* Ay_ref - Vy_ref .* Ax_ref) ./ max((Vx_ref.^2 + Vy_ref.^2).^(3/2), 1e-6);

RefMatrix = [X_ref', Y_ref', Vx_ref', Vy_ref'];

% 2. Dinamika
A = [1 0 Ts 0; 0 1 0 Ts; 0 0 1 0; 0 0 0 1];
B = [0 0; 0 0; Ts 0; 0 Ts];

% --- JAVÍTOTT LQR BEÁLLÍTÁS ---
% --- KISEGÍTŐ SZABÁLYOZÓ (Ancillary LQR) JAVÍTÁSA ---
Q_lqr = diag([50, 50, 5, 5]); 
R_lqr = diag([0.5, 0.5]);     
K_lqr = dlqr(A, B, Q_lqr, R_lqr);

% ==========================================
% 4. CASADI OPTIMIZÁCIÓS PROBLÉMA FELÉPÍTÉSE
% ==========================================
opti = casadi.Opti();

X = opti.variable(4, N+1); 
U = opti.variable(2, N);   

x0_param = opti.parameter(4, 1);         
ref_param = opti.parameter(4, N+1);      
tube_scale_param = opti.parameter(1, 1);

Q_mpc = diag([100, 100, 0, 0]); 
R_mpc = diag([0.1, 0.1]);     

cost = 0;
for k = 1:N
    err = X(:, k) - ref_param(:, k);
    cost = cost + err' * Q_mpc * err + U(:, k)' * R_mpc * U(:, k);
    opti.subject_to(X(:, k+1) == A * X(:, k) + B * U(:, k));
end
err_N = X(:, N+1) - ref_param(:, N+1);
cost = cost + err_N' * Q_mpc * err_N;

opti.minimize(cost);
opti.subject_to(X(:, 1) == x0_param);

% CasADi limit (nominális terv)
u_tight_base = 20; 
opti.subject_to( -(u_tight_base) <= U <= (u_tight_base) );

p_opts = struct('expand', true); 
s_opts = struct('max_iter', 100, 'print_level', 0, 'acceptable_tol', 1e-6); 
opti.solver('ipopt', p_opts, s_opts);

% ==========================================
% 5. SZIMULÁCIÓS KÖRNYEZET
% ==========================================
x_real = RefMatrix(1,:)'; 
z_nom = x_real;           

figure('Name', 'CasADi Tube MPC Szimuláció (ADATGYŰJTÉS)', 'NumberTitle', 'off');
plot(smooth_path(:,1), smooth_path(:,2), '-g', 'LineWidth', 2); hold on; grid on;
h_nom = plot(z_nom(1), z_nom(2), 'ob', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
h_real = plot(x_real(1), x_real(2), 'or', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
lgd = legend('Ideális Ív', 'CasADi Nominális', 'Valós (Zavarva)', 'Location', 'best');
lgd.AutoUpdate = 'off'; 
title('CasADi Tube MPC - Tanító Adat Gyűjtése');

real_history = []; nom_history = [];
disp('CasADi szimuláció fut... (az IPOPT megoldó dolgozik)');

x_real_prev = x_real; 

% Létrehozzuk az üres mátrixokat (MÁR 15 OSZLOP!)
Training_Inputs = zeros(length(t_sim) - N - 1, 15);
Training_Outputs = zeros(length(t_sim) - N - 1, 4);

szel_memoria = [0; 0]; 
x_real_prev = x_real; 

% --- ÚJ: TÖRTÉNETI PUFFEREK INICIALIZÁLÁSA ---
x_hist1 = x_real; 
x_hist2 = x_real; 
u_prev_log = [0; 0];

% A ciklus ELŐTT
zaj_kormany_memoria = 0;
zaj_gaz_memoria = 0;

% 6. VEZÉRLÉSI HUROK
for k = 1 : length(t_sim) - N - 1
    
    current_ref = RefMatrix(k : k+N, :)';
    opti.set_value(x0_param, z_nom);       
    opti.set_value(ref_param, current_ref);
    
    try, sol = opti.solve(); v_k_opt = sol.value(U(:, 1)); catch, v_k_opt = opti.debug.value(U(:, 1)); end
    
    u_k = v_k_opt - K_lqr * (x_real - z_nom);
    u_k = max(min(u_k, 50), -50);
    z_nom = A * z_nom + B * v_k_opt; 

    % --- FIZIKA ÉS SZÉL (Valódi Brute-Force Változat) ---
    drag_x = -0.02 * x_real(3) * abs(x_real(3)) * Ts;
    drag_y = -0.02 * x_real(4) * abs(x_real(4)) * Ts;
    const_wind_x = 0.8 * Ts; 
    const_wind_y = 0.5 * Ts;
    
    % ÚJ: Lassan "vándorló" szél!
    % A 'k' (idő) segítségével lassan eltoljuk a szinusz fázisát és változtatjuk az erejét.
    % Így a 25 kör alatt minden egyes körben MÁSHONNAN és MÁS ERŐVEL fog fújni a térbeli szél!
    lassu_drift = k / 400; 
    
    spatial_wind_x = (0.3 + 0.3 * abs(sin(lassu_drift))) * sin((x_real(1) / 15) + lassu_drift);
    spatial_wind_y = (0.3 + 0.3 * abs(cos(lassu_drift))) * cos((x_real(2) / 15) + lassu_drift);
    
    zona_szorzo = 0.1 + 1.4 * (0.5 + 0.5 * tanh((x_real(1) - 40) / 5));
    
    % A turbulencia (amit ténylegesen kever az rng('shuffle'))
    uj_zaj_x = randn() * 0.4;
    uj_zaj_y = randn() * 0.4;
    szel_memoria(1) = 0.8 * szel_memoria(1) + uj_zaj_x;
    szel_memoria(2) = 0.8 * szel_memoria(2) + uj_zaj_y;
    
    zaj_x = (spatial_wind_x + szel_memoria(1)) * zona_szorzo;
    zaj_y = (spatial_wind_y + szel_memoria(2)) * zona_szorzo;

    w_k = [0; 0; const_wind_x + drag_x + zaj_x; const_wind_y + drag_y + zaj_y];
    
    % --- RENDSZER-IDENTIFIKÁCIÓS (FELFEDEZŐ) ZAJ ---
    % Mesterségesen picit "megrángatjuk" a kormányt és a gázt, hogy a gumiabroncs
    % különböző csúszási szögeket (Slip Angle) tapasztaljon meg. Ezt az MPC nem látja!
    % Ebből fogja az LSTM megtanulni az extrém fizikai határokat.
    
    % --- RENDSZER-IDENTIFIKÁCIÓS (AUTOREGRESSZÍV) ZAJ ---
    % Ez már nem tiszta fehér zaj! Ez egy "vándorló" hiba, ami 
    % tartósan kitolja az autót az ívről (pl. rossz futómű-beállítás,
    % vagy egy több másodperces vízátfolyás szimulálása).
    
    % 90%-ban megtartja az előző értéket, 10%-ban kap új véletlent
    zaj_kormany_memoria = 0.90 * zaj_kormany_memoria + 0.10 * randn() * 0.4;
    zaj_gaz_memoria     = 0.90 * zaj_gaz_memoria     + 0.10 * randn() * 1.0;
    
    u_k_zajos = u_k + [zaj_gaz_memoria; zaj_kormany_memoria];
    
    % A VALÓSÁG
    x_real_new = dynamic_car_step(x_real, u_k_zajos, Ts);
    
    % A VALÓSÁG: Az új dinamikus fizikai motor hajtja az autót a zajos bemenettel
    x_real_new = dynamic_car_step(x_real, u_k_zajos, Ts);
    
    % Az MPC (buta modell) szerinti predikció
    x_predicted = A * x_real_prev + B * u_k; 
    residual_error = x_real_new - x_predicted; % A gumi csúszásából adódó hiba!

    % --- ÚJ: 15 ELEMŰ BEMENET MENTÉSE (kappa-val kiegészítve) ---
    current_kappa = kappa_ref(k);
    Training_Inputs(k, :) = [x_real_prev', x_hist1', x_hist2', u_prev_log', current_kappa]; 
    Training_Outputs(k, :) = residual_error';

    % --- ÚJ: PUFFEREK LÉPTETÉSE A KÖVETKEZŐ KÖRRE ---
    x_hist2 = x_hist1;
    x_hist1 = x_real_prev;
    x_real_prev = x_real_new;
    x_real = x_real_new;
    u_prev_log = u_k;
    
    % mentés
    real_history = [real_history; x_real'];
    nom_history = [nom_history; z_nom'];
    
    % animáció
    if mod(k, 3) == 0 
        set(h_nom, 'XData', z_nom(1), 'YData', z_nom(2));
        set(h_real, 'XData', x_real(1), 'YData', x_real(2));
        plot([z_nom(1), x_real(1)], [z_nom(2), x_real(2)], '-k', 'LineWidth', 0.5); 
        drawnow;
    end

end

disp('Adatgyűjtés (Sliding Window) kész!');

% Végső trajektóriák kirajzolása
plot(real_history(:,1), real_history(:,2), 'r--', 'LineWidth', 1.5);
plot(nom_history(:,1), nom_history(:,2), 'b:', 'LineWidth', 1.5);

% =========================================================================
% ÚJ 4. PONT: DOMAIN RANDOMIZATION (Állapottér felfedezése)
% =========================================================================
disp('--- Szintetikus Extrém Adatok Generálása (Domain Randomization) ---');
N_rand = 5000; % 5000 véletlenszerű fizikai szituáció betanítása
Rand_Inputs = zeros(N_rand, 15);
Rand_Outputs = zeros(N_rand, 4);

for i = 1:N_rand
    % 1. Véletlenszerű fizikai állapotok (A pálya mérete 100x100m, V max ~20m/s)
    x_r = rand() * 100; 
    y_r = rand() * 100; 
    vx_r = (rand() - 0.5) * 40; % -20 és +20 m/s közötti extrém sebességek
    vy_r = (rand() - 0.5) * 40; 
    x_rand = [x_r; y_r; vx_r; vy_r];

    % 2. Véletlenszerű múltbeli állapotok (hogy mozgásban lévő autót szimuláljunk)
    x_hist1_rand = x_rand - [vx_r*Ts; vy_r*Ts; 0; 0] + randn(4,1)*0.2;
    x_hist2_rand = x_hist1_rand - [vx_r*Ts; vy_r*Ts; 0; 0] + randn(4,1)*0.2;

    % 3. Véletlenszerű irányítás és görbület
    u_prev_r = (rand(2,1) - 0.5) * 100; % -50 és 50 közötti erők
    u_k_r = (rand(2,1) - 0.5) * 100;
    kappa_r = (rand() - 0.5) * 1.0; % Extrém görbületek

    % 4. A Valós Fizika kiszámítása ezekre a pontokra (mint a fő ciklusban)
    drag_x_r = -0.02 * x_rand(3) * abs(x_rand(3)) * Ts;
    drag_y_r = -0.02 * x_rand(4) * abs(x_rand(4)) * Ts;

    % Randomizált térbeli szél és turbulencia
    spatial_wind_x_r = (0.3 + 0.3 * rand()) * sin((x_rand(1) / 15) + rand()*2*pi);
    spatial_wind_y_r = (0.3 + 0.3 * rand()) * cos((x_rand(2) / 15) + rand()*2*pi);
    zona_szorzo_r = 0.1 + 1.4 * (0.5 + 0.5 * tanh((x_rand(1) - 40) / 5));

    zaj_x_r = spatial_wind_x_r * zona_szorzo_r; 
    zaj_y_r = spatial_wind_y_r * zona_szorzo_r;

    % Zavarójel
    w_k_r = [0; 0; (0.8*Ts) + drag_x_r + zaj_x_r; (0.5*Ts) + drag_y_r + zaj_y_r];

    % 5. Valós jövőkép vs. Lineáris modell szerinti predikció
    x_real_new_r = A * x_rand + B * u_k_r + w_k_r;
    x_predicted_r = A * x_rand + B * u_k_r;
    residual_r = x_real_new_r - x_predicted_r;

    % 6. Elmentjük az adatbázisba
    Rand_Inputs(i, :) = [x_rand', x_hist1_rand', x_hist2_rand', u_prev_r', kappa_r];
    Rand_Outputs(i, :) = residual_r';
end

% A szintetikus adatokat hozzácsapjuk az eddigi "normál" körözős adatokhoz
Training_Inputs = [Training_Inputs; Rand_Inputs];
Training_Outputs = [Training_Outputs; Rand_Outputs];

disp(['Sikeresen hozzáadva ' num2str(N_rand) ' szintetikus adatpont.']);

disp('Szimuláció sikeresen befejeződött!');

% =========================================================================
% SEGÉDFÜGGVÉNY: VALÓS GUMIABRONCS FIZIKA (Dinamikus Kerékpármodell)
% =========================================================================
function x_real_new = dynamic_car_step(x_real, u_k, Ts)
% x_real = [X, Y, Vx, Vy]
% u_k = [ax, ay] (A CasADi MPC által kért ideális gyorsulások)

% 1. Autó fizikai paraméterei
m = 1500; Iz = 2500; Lf = 1.2; Lr = 1.6; L = Lf + Lr;
Cf = 80000; Cr = 80000; % Kanyarodási merevség (Cornering stiffness)
mu = 0.9; g = 9.81; % Súrlódási együttható (Tapadás)

% 2. Jelenlegi állapotok
X = x_real(1); Y = x_real(2);
Vx = x_real(3); Vy = x_real(4);
V = max(sqrt(Vx^2 + Vy^2), 0.1); % Jármű sebessége
psi = atan2(Vy, Vx); % Jármű tényleges haladási iránya

% 3. Mit kér az MPC? (Átfordítjuk a globális ax,ay-t autó-specifikus gázra és kormányra)
ax_global = u_k(1); ay_global = u_k(2);
a_lon_req = ax_global * cos(psi) + ay_global * sin(psi); % Gáz/Fék pedál
a_lat_req = -ax_global * sin(psi) + ay_global * cos(psi); % Kívánt keresztgyorsulás

% Kinematikai kormányszög számítása az MPC kérése alapján
delta = atan((L * a_lat_req) / (V^2));
delta = max(min(delta, 0.6), -0.6); % Kormány limitálása (+- ~35 fok)

% 4. VALÓS GUMIABRONCS CSÚSZÁS SZÁMÍTÁSA (Slip Angles)
% Kiszámoljuk, mennyire csúszik meg a kocsi fara és orra az adott kormányszögnél
beta = delta * (Lr / L); % Oldalcsúszási szög
r_kin = (V / L) * tan(delta); % Perdület

alpha_f = delta - atan((V * sin(beta) + Lf * r_kin) / (V * cos(beta)));
alpha_r = -atan((V * sin(beta) - Lr * r_kin) / (V * cos(beta)));

% 5. Nemlineáris Gumi Erők (Tapadás elvesztése)
Fz_f = m * g * (Lr / L);
Fz_r = m * g * (Lf / L);

% Tanh() limitálja az erőt: Ha túl nagy a csúszás, a gumi nem tapad jobban!
F_yf = mu * Fz_f * tanh((Cf * alpha_f) / (mu * Fz_f));
F_yr = mu * Fz_r * tanh((Cr * alpha_r) / (mu * Fz_r));

% Valós keresztirányú gyorsulás (Ez az, ami MIATT az autó "kiesik" az ívről!)
a_lat_real = (F_yf * cos(delta) + F_yr) / m;

% 6. Visszatranszformálás globális koordinátarendszerbe
ax_real = a_lon_req * cos(psi) - a_lat_real * sin(psi);
ay_real = a_lon_req * sin(psi) + a_lat_real * cos(psi);

% Kiegészítő külső szél / turbulencia (csak egy pici, hogy az is legyen)
szel_x = 0.5 * sin(X/20); szel_y = 0.5 * cos(Y/20);

% Új állapotok integrálása
Vx_new = Vx + (ax_real + szel_x) * Ts;
Vy_new = Vy + (ay_real + szel_y) * Ts;
X_new = X + Vx_new * Ts;
Y_new = Y + Vy_new * Ts;

x_real_new = [X_new; Y_new; Vx_new; Vy_new];
end
