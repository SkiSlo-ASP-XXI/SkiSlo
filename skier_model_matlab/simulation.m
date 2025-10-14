clear;
close all;
%% ---- Nuvola di Punti ----
[pts, Xg, Yg, Zg, meta] = buildSlopePointCloud('TerrainType' ,3);

%% ----- Interpolante z = h(x,y) -----
Fz = scatteredInterpolant(pts(:,1), pts(:,2), pts(:,3), 'natural','nearest');
% plotSurface(Fz, meta)

%% ---- Func. per gradienti e z ----
h = @(x,y) Fz(x,y);
grad_h = @(x,y) compute_grad_h(Fz, x, y, 1e-3);
alpha = @(x,y)  compute_slope_angle(Fz, x, y);

%% ---- Parametri fisici ----
m = 80;            % massa [kg]
g = 9.81;          % gravità [m/s^2]
mu = 0.04;         % coeff. attrito radente neve
rho = 1.2;         % densità aria [kg/m^3]
CdA = 0.5 * 0.7;   % 0.5 * Cd * Area (valore esemplificativo)


%% ---- Condizioni iniziali ----
x0 = 0; y0 = 8;             % posizione di partenza (sull'area generata)
z0 = h(x0,y0);
speed0 = 0.5;                  % velocità iniziale lungo tangente [m/s]
% direzione iniziale: verso discesa locale 
beta = deg2rad(80);
tangent_dir = [cos(beta); sin(beta)];  % Non tiene conto di z
tangent_dir = tangent_dir / norm(tangent_dir);
vx0 = speed0 * tangent_dir(1);
vy0 = speed0 * tangent_dir(2);

state0 = [x0; y0; vx0; vy0];

%% ---- angolo beta ----
% TODO: decidere come aggiornare angolo beta per fare la curva

%% ---- integrazione ODE ---
tspan = [0, 60];  % tempo di simulazione [s]
options = odeset('RelTol',1e-6,'AbsTol',1e-7);
[tt, YY] = ode45(@(t, state) odefun(t, state, h, grad_h, beta, m, g, mu, rho, CdA), ...
                 tspan, state0, options);

% costruisci traiettoria 3D
X = YY(:,1); Y = YY(:,2);
Z = arrayfun(@(i) h(X(i), Y(i)), 1:length(X))';

%% ---- Visualizzazione Traiettoria ----
figure('Position',[100 100 1100 600]);

% --- Pannello sinistro: terreno e traiettoria 3D ---
subplot(1,2,1);
plotSurface(Fz, meta);      % usa la tua funzione per disegnare il terreno
hold on;

% traiettoria calcolata
plot3(X, Y, Z, 'r-', 'LineWidth', 2);

% punto iniziale
scatter3(X(1), Y(1), Z(1), 80, 'g', 'filled');

% dettagli grafici
title('Terreno e traiettoria');
view(45,35); axis tight; 
camlight; lighting gouraud;

% --- Pannello destro: velocità tangente ---
subplot(1,2,2);
plot(tt, sqrt( YY(:,3).^2 + YY(:,4).^2 ), 'LineWidth', 1.5);
xlabel('Tempo [s]'); ylabel('Velocità tangente [m/s]');
title('Velocità tangente nel tempo');
grid on;

% ---- Finestra aggiuntiva: vista libera del terreno e traiettoria ----
figure('Name','Vista 3D completa','Position',[200 150 900 700]);
plotSurface(Fz, meta);
hold on;

% traiettoria e punto iniziale
plot3(X, Y, Z, 'r-', 'LineWidth', 2);
scatter3(X(1), Y(1), Z(1), 80, 'g', 'filled');

% impostazioni visive
title('Vista 3D - Terreno e traiettoria');
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
view(40,30);  % cambia angolo se vuoi (azimuth,elevation)
axis tight; camlight; lighting gouraud; grid on;


%% ---- Funzioni Locali ----

function dst = odefun(~, state, h, grad_h, beta, m, g, mu, rho, CdA)
    % state = [x;y;vx;vy]
    x = state(1); y = state(2); vx = state(3); vy = state(4);

    z = h(x,y);
    [hx, hy] = grad_h(x,y);  % gradiente alla superficie              
    n_unnorm = [-hx; -hy; 1];  % normale alla superficie
    n = n_unnorm / norm(n_unnorm);

    vz = hx*vx + hy*vy;  % vincolo di superficie
    v3 = [vx; vy; vz];
    vmag = norm(v3);

    % Forze
    Fg = [0; 0; -m*g];

    if vmag > 1e-6
        t_hat = v3 / vmag;
        F_fric = - mu * m * g * t_hat;
        F_drag = - 0.5 * rho * CdA * vmag * v3;
    else
        F_fric = [0;0;0];
        F_drag = [0;0;0];
    end

    % Proiezione sul piano tangente
    P = eye(3) - n*n.';
    F_tan = P * (Fg + F_fric + F_drag);

    ubeta = [cos(beta); sin(beta); hx*cos(beta) + hy*sin(beta)];
    t_beta = ubeta / norm(ubeta);
    F_beta = (F_tan.' * t_beta) * t_beta;

    a3 = F_beta / m;
    ax = a3(1); 
    ay = a3(2);

    % Derivata di stato
    dst = [vx;        % dx/dt
           vy;        % dy/dt
           ax;        % dvx/dt
           ay];       % dvy/dt
end

function [hx, hy] = compute_grad_h(Fzfun, xq, yq, eps)
    % derivata centrale a due punti (finite difference)
    zpx = Fzfun(xq+eps, yq);
    zmx = Fzfun(xq-eps, yq);
    zpy = Fzfun(xq, yq+eps);
    zmy = Fzfun(xq, yq-eps);
    hx = (zpx - zmx) / (2*eps);
    hy = (zpy - zmy) / (2*eps);
end

function alpha = compute_slope_angle(Fz, x, y, hstep, units)
% alpha = slope_angle(Fz, x, y, hstep, units)
% Calcola l'inclinazione alpha(x,y) della superficie z=Fz(x,y).
%   Fz    : function handle z=Fz(x,y)
%   x, y  : scalare o matrici di uguali dimensioni
%   hstep : (opz.) passo per differenze finite, default 1e-3
%   units : (opz.) 'rad' (default) o 'deg'
%
% Output:
%   alpha : inclinazione locale (stessa shape di x,y)

    if nargin < 4 || isempty(hstep), hstep = 1e-3; end
    if nargin < 5 || isempty(units), units = 'rad'; end

    % --- gradiente numerico (usa la tua funzione) ---
    [gx, gy] = compute_grad_h(Fz, x, y, hstep);

    % modulo del gradiente = tan(alpha)
    t = hypot(gx, gy);

    alpha = atan2(t, 1);

    if startsWith(lower(units), 'deg')
        alpha = rad2deg(alpha);
    end
end

  