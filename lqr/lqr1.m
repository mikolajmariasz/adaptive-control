function lqr_1
    % ==================================================
    % 1) Definicja układu w postaci przestrzeni stanów
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
    % 3) Regulacja P:  u(t) = y_ref - y(t) = 1 - C*x(t)
    % ==================================================
    
    % Definiujemy funkcję do ode45: 
    %  dx/dt = A*x + B*u,
    %  gdzie u = (y_ref - C*x).
    
    odefunP = @(t, x) A*x + B*( y_ref - C*x );
    
    % Symulacja regulatora P:
    [tP, xP] = ode45(odefunP, tspan, x0);
    yP = (C * xP.').';          % y(t) = C*x(t)
    uP = y_ref - yP;            % u(t) = 1 - y(t)
    
    % ==================================================
    % 4) Regulacja LQR z przesunięciem do (x_ref, u_ref)
    % ==================================================
    %
    %  Standardowe LQR minim prowadzi do x=0.
    %  Jeżeli chcemy mieć y=1, trzeba "przesunąć" układ do (x_ref, u_ref).
    %
    %  Definiujemy: x_t = x - x_ref. Wtedy x_t'= A_cl*x_t, z A_cl = (A - B*K).
    
    % Macierze wag:
    Q = eye(2);
    R = 1;
    
    % Wzmocnienie K:
    K = lqr(A, B, Q, R);
    
    % Macierz w pętli zamkniętej (dla odchyłki x_t):
    A_cl = A - B*K; 
    
    % Funkcja stanu dla x_t:
    %   d(x_t)/dt = A_cl * x_t,
    % bo w pętli mamy u = u_ref - K(x - x_ref) => u - u_ref = -K x_t.
    odefunLQR = @(t, x_t) A_cl*x_t;
    
    % Stan początkowy w sensie odchyłki: x_t(0) = x0 - x_ref
    x_t0 = x0 - x_ref;
    
    % Symulacja regulatora LQR:
    [tLQR, x_t_LQR] = ode45(odefunLQR, tspan, x_t0);
    
    % Pełny stan: x(t) = x_ref + x_t(t)
    xLQR = x_t_LQR + x_ref.';
    
    % Wyjście:
    yLQR = (C * xLQR.').';
    
    % Sterowanie:
    %   u(t) = u_ref - K*(x - x_ref) = u_ref - K*x_t
    %         = u_ref - K*( x_t )
    uLQR = zeros(size(tLQR));
    for i=1:length(tLQR)
       uLQR(i) = u_ref - K*x_t_LQR(i,:)';
    end
    
    % ==================================================
    % 5) Wykres z wynikiem
    % ==================================================
    
    figure('Name','Porównanie P vs LQR','NumberTitle','off');
    
    hold on; grid on;
    plot(tP,   yP,   'r--','LineWidth',2);
    plot(tLQR, yLQR, 'b-','LineWidth',2);
    plot(tLQR, y_ref*ones(size(tLQR)), 'k:','LineWidth',1.5);
    legend('Regulacja P','Regulacja LQR','y_{ref}=1','Location','best');
    xlabel('Czas [s]');  ylabel('y(t)');
    title('Wyjście układu');
    
end
