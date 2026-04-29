function [InitialObservation, LoggedSignals] = reset()
    % A szál (worker) saját globális változói
    global RefMatrix opti x0_param ref_param U A B Ts K_lqr x_real z_nom MPC_k tube_scale_param worker_initialized

    % =====================================================================
    % INICIALIZÁLÁS A WORKEREN (Csak a legelső epizódnál fut le!)
    % =====================================================================
    if isempty(worker_initialized)
        import casadi.*
        
        % 1. Pálya betöltése a fájlból (amit a fő kód mentett el)
        data = load('track_data.mat');
        RefMatrix = data.RefMatrix;
        
        % 2. Dinamika és LQR
        Ts = 0.1;
        A = [1 0 Ts 0; 0 1 0 Ts; 0 0 1 0; 0 0 0 1];
        B = [0 0; 0 0; Ts 0; 0 Ts];
        Q_lqr = diag([10, 10, 1, 1]);
        R_lqr = diag([0.5, 0.5]);
        K_lqr = dlqr(A, B, Q_lqr, R_lqr);
        
        % 3. CasADi Felépítése (Minden worker csinál magának egyet)
        N = 15;
        opti = casadi.Opti();
        X = opti.variable(4, N+1); 
        U = opti.variable(2, N);   

        x0_param = opti.parameter(4, 1);         
        ref_param = opti.parameter(4, N+1);      
        tube_scale_param = opti.parameter(1, 1);
        
        Q_mpc = diag([20, 20, 0, 0]); 
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

        u_tight_base = 20; 
        opti.subject_to( -(u_tight_base * tube_scale_param) <= U <= (u_tight_base * tube_scale_param) );

        % Gyorsított solver a párhuzamosításhoz! (50 iteráció)
        p_opts = struct('expand', true); 
        s_opts = struct('max_iter', 50, 'print_level', 0, 'acceptable_tol', 1e-3); 
        opti.solver('ipopt', p_opts, s_opts);
        
        % Jelöljük, hogy a worker kész van, többé nem kell felépíteni
        worker_initialized = true;
    end

    % =====================================================================
    % EPIZÓD VISSZAÁLLÍTÁSA (Minden epizód elején lefut)
    % =====================================================================
    x_real = RefMatrix(1, :)'; 
    z_nom  = x_real;
    MPC_k  = 1;
    
    InitialObservation = [0; 0; x_real(3)/10.0; x_real(4)/10.0; 0];
    LoggedSignals = [];
end