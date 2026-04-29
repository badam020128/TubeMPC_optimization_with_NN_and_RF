function [Observation, Reward, IsDone, LoggedSignals] = step(Action, LoggedSignals)
    try
        % A worker saját globális változóit hívjuk be!
        global RefMatrix opti x0_param ref_param U A B Ts K_lqr x_real z_nom MPC_k tube_scale_param
        
        tube_scale = Action; 
        N = 15; 
        
        % --- PÁLYA VÉGÉNEK ELLENŐRZÉSE ---
        if MPC_k + N > size(RefMatrix, 1)
            IsDone = true;
            Reward = 10; 
            Observation = [0; 0; x_real(3)/10.0; x_real(4)/10.0; 0];
            LoggedSignals = []; 
            return;
        end
        
        % --- CASADI PARAMÉTEREK FRISSÍTÉSE ---
        current_ref = RefMatrix(MPC_k : MPC_k+N, :)';
        opti.set_value(x0_param, z_nom);       
        opti.set_value(ref_param, current_ref);
        opti.set_value(tube_scale_param, tube_scale); 
        
        % --- CASADI OPTIMALIZÁCIÓ ---
        try
            sol = opti.solve(); 
            v_k_opt = sol.value(U(:, 1)); 
        catch
            v_k_opt = opti.debug.value(U(:, 1)); 
        end
        
        % --- LQR ÉS FIZIKA ---
        u_k = v_k_opt - K_lqr * (x_real - z_nom);
        u_k = max(min(u_k, 25), -25); 
        
        x_predicted = A * x_real + B * u_k; 
        
        drag_x = -0.02 * x_real(3) * abs(x_real(3)) * Ts;
        drag_y = -0.02 * x_real(4) * abs(x_real(4)) * Ts;
        const_wind_x = 0.8 * Ts; 
        const_wind_y = 0.5 * Ts;
        zaj_x = randn() * 0.5;
        zaj_y = randn() * 0.5;
        
        w_k = [0; 0; const_wind_x + drag_x + zaj_x; const_wind_y + drag_y + zaj_y];
        
        x_real_next = A * x_real + B * u_k + w_k;
        z_nom_next  = A * z_nom + B * v_k_opt;
        
        residual_error = norm(x_real_next - x_predicted); 
        e_x = x_real_next(1) - z_nom_next(1);             
        e_y = x_real_next(2) - z_nom_next(2);             
        
        % --- JUTALOM (Reward Shaping a túlélésért) ---
        tracking_penalty = - (e_x^2 + e_y^2) * 0.1;      
        aggressiveness_bonus = (2.5 - tube_scale) * 0.1; 
        survival_bonus = 0.5; % +0.5 pont minden sikeres lépésért a pályán!
        
        Reward = tracking_penalty + aggressiveness_bonus + survival_bonus;
        IsDone = false;
        
        if norm([e_x, e_y]) > 5
            Reward = Reward - 50; % Fájdalmas -50 pontos büntetés, ha leesik!
            IsDone = true;
        end
        
        % --- NORMALIZÁLÁS ---
        obs_e_x = e_x / 5.0;            
        obs_e_y = e_y / 5.0;
        obs_v_x = x_real_next(3) / 10.0; 
        obs_v_y = x_real_next(4) / 10.0;
        obs_err = residual_error / 2.0;  
        
        Observation = [obs_e_x; obs_e_y; obs_v_x; obs_v_y; obs_err];
        LoggedSignals = []; 
        
        x_real = x_real_next;
        z_nom  = z_nom_next;
        MPC_k  = MPC_k + 1;

    catch ME
        disp('!!! KÓD ÖSSZEOMLÁS A step függvényben !!!');
        disp(ME.message);
        Observation = zeros(5, 1);
        Reward = 0;
        IsDone = true;
        LoggedSignals = [];
    end
end