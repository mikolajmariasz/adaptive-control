function lqr_with_observer
    % ==================================================
    % 1) Definicja układu w przestrzeni stanów
    %    z transmitancji K(s)=(s-7)/[(s+9)(s+5)].
    % ==================================================
    
    % Macierze stanu:
    A = [  0   -45
           1   -14 ];
    B = [ -7
           1 ];
    C = [  0    1 ];
    D = 0;             
    
    % Zadana wartość wyjścia w stanie ustalonym:
    y_ref = 1;
    
    % Warunki początkowe:
    x0 = [0; 0];
    
    % Zakres symulacji w sekundach:
    tspan = [0 3];
    
    % ==================================================
    % 2) Punkt pracy (x_ref, u_ref), by w stanie ustalonym y=1
    % ==================================================
    M   = [A, B; 
           C, D];
    rhs = [0; 0; 
           y_ref];
    
    XU_ref = M \ rhs;  % rozwiązuje układ liniowy M * [x_ref; u_ref] = rhs
    
    n = size(A,1);     
    x_ref = XU_ref(1:n);
    u_ref = XU_ref(n+1:end);
    
    % ==================================================
    % 3) Obserwator Luenbergera
    % ==================================================
    % Dobór wzmocnienia obserwatora (L):
    L = [0; 1]; 
    
    % ==================================================
    % 4) Regulacja LQR
    % ==================================================
    % Macierze wag dla LQR:
    Q = eye(2);
    R = 1;
    
    % Wzmocnienie K:
    K = lqr(A, B, Q, R);
    
    % ==================================================
    % 5) Symulacja układu z obserwatorem i LQR
    % ==================================================
    % Łączny wektor stanu: [x rzeczywiste; x estymowane]
    state0 = [x0; [1; 1]]; % Początkowy stan rzeczywisty i estymowany
    
    % Definicja równania różniczkowego:
    odefun = @(t, state) [
        A * state(1:2) + B * (u_ref - K * (state(3:4) - x_ref));                % Rzeczywisty stan
        A * state(3:4) + B * (u_ref - K * (state(3:4) - x_ref)) + L * (C * state(1:2) - C * state(3:4)) % Obserwator
    ];
    
    % Symulacja za pomocą ode45:
    [t, state] = ode45(odefun, tspan, state0);
    
    % Rozdzielenie stanów:
    x_real = state(:, 1:2);   % Rzeczywisty stan
    x_est = state(:, 3:4);    % Stan estymowany
    
    % Wyjścia układu:
    y_real = (C * x_real.').';  % Wyjście rzeczywiste
    y_est = (C * x_est.').';    % Wyjście estymowane
    
    % Sterowanie:
    u = zeros(size(t));
    for i = 1:length(t)
        u(i) = u_ref - K * (x_est(i, :)' - x_ref);
    end
    
    % Błąd estymacji:
    estimation_error = x_real - x_est;
    
    % ==================================================
    % 6) Wykresy
    % ==================================================
    
    % Wyjścia rzeczywiste i estymowane:
    figure('Name', 'Wyjście układu i obserwatora', 'NumberTitle', 'off');
    hold on; grid on;
    plot(t, y_real, 'b-', 'LineWidth', 2);
    plot(t, y_est, 'r--', 'LineWidth', 2);
    plot(t, y_ref * ones(size(t)), 'k:', 'LineWidth', 1.5);
    legend('Wyjście rzeczywiste', 'Wyjście estymowane', 'y_{ref}=1', 'Location', 'best');
    xlabel('Czas [s]');
    ylabel('Wyjście y(t)');
    title('Porównanie wyjść układu rzeczywistego i obserwatora');
    
    % Sterowanie:
    figure('Name', 'Sterowanie', 'NumberTitle', 'off');
    plot(t, u, 'g-', 'LineWidth', 2);
    grid on;
    xlabel('Czas [s]');
    ylabel('Sterowanie u(t)');
    title('Sterowanie LQR');
    
    % Błąd estymacji:
    figure('Name', 'Błąd estymacji', 'NumberTitle', 'off');
    plot(t, estimation_error(:, 1), 'r-', 'LineWidth', 2);
    hold on;
    plot(t, estimation_error(:, 2), 'b-', 'LineWidth', 2);
    grid on;
    xlabel('Czas [s]');
    ylabel('Błąd estymacji');
    legend('Błąd x_1', 'Błąd x_2', 'Location', 'best');
    title('Błąd estymacji stanu');
end
