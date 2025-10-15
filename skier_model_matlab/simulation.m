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
alpha_fun = @(x,y)  compute_slope_angle(Fz, x, y);

%% ---- Parametri fisici ----
m = 70;            % massa [kg]
g = 9.81;          % gravità [m/s^2]
mu = 0.02;         % coeff. attrito radente neve
rho = 1.2;         % densità aria [kg/m^3]
CdA = 0.5 * 0.4 *0.5;   % 0.5 * Cd * Area (valore esemplificativo)


%% ---- Condizioni iniziali ----
x0 = 0; y0 = 8;             % posizione di partenza (sull'area generata)
z0 = h(x0,y0);
speed0 = 0;                  % modulo velocità iniziale [m/s] 
% direzione iniziale: verso discesa locale 
beta = deg2rad(80);
alpha = alpha_fun(x0, y0);
tangent_dir = [cos(beta); sin(beta)*cos(alpha); sin(beta)*sin(alpha)];  
tangent_dir = tangent_dir / norm(tangent_dir);
vx0 = speed0 * tangent_dir(1);
vy0 = speed0 * tangent_dir(2);
vz0 = speed0 * tangent_dir(3);

state0 = [x0; y0; vx0; vy0; vz0]; % sistema di riferimento 

%% ---- angolo beta ----
% TODO: decidere come aggiornare angolo beta per fare la curva

%% ---- integrazione ODE ---
tspan = [0, 60];  % tempo di simulazione [s]
events = @(t,state) skiEvents(t, state, meta);
options = odeset('RelTol',1e-6,'AbsTol',1e-7, 'Events',events);
[tt, YY] = ode45(@(t, state) odefun(t, state, h, grad_h, beta, alpha_fun, m, g, mu, rho, CdA), ...
                 tspan, state0, options);

% costruisci traiettoria 3D
X = YY(:,1); Y = YY(:,2);
Z = arrayfun(@(i) h(X(i), Y(i)), 1:length(X))';

%% ---- Visualizzazione Traiettoria ----
figure('Position',[100 100 1200 700]);

% --- Pannello sinistro: terreno e traiettoria 3D ---
subplot(2,2,[1 3]);  % occupa due righe
plotSurface(Fz, meta);      % funzione per disegnare il terreno
hold on;

% traiettoria calcolata
plot3(X, Y, Z, 'r-', 'LineWidth', 2);

% punto iniziale
scatter3(X(1), Y(1), Z(1), 80, 'g', 'filled');

% dettagli grafici
title('Terreno e traiettoria');
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
view(45,35); axis tight; 
camlight; lighting gouraud; grid on;

% --- Pannello in alto a destra: velocità tangente ---
subplot(2,2,2);
plot(tt, sqrt(YY(:,3).^2 + YY(:,4).^2 + YY(:,5).^2), 'LineWidth', 1.5);
xlabel('Tempo [s]'); ylabel('Velocità tangente [m/s]');
title('Velocità tangente nel tempo');
grid on;

% --- Pannello in basso a destra: posizione (modulo) nel tempo ---
subplot(2,2,4);
pos_mod = sqrt(X.^2 + Y.^2 + Z.^2);
plot(tt, pos_mod, 'LineWidth', 1.5);
xlabel('Tempo [s]'); ylabel('Posizione |r(t)| [m]');
title('Posizione (modulo) nel tempo');
grid on;

% ---- Finestra aggiuntiva: vista 3D libera ----
figure('Name','Vista 3D completa','Position',[200 150 900 700]);
plotSurface(Fz, meta);
hold on;

% traiettoria e punto iniziale
plot3(X, Y, Z, 'r-', 'LineWidth', 2);
scatter3(X(1), Y(1), Z(1), 80, 'g', 'filled');

% impostazioni visive
title('Vista 3D - Terreno e traiettoria');
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
view(40,30);  
axis tight; camlight; lighting gouraud; grid on;


%% ---- Funzioni Locali ----

function dst = odefun(~, state, h, grad_h, beta, alpha_fun, m, g, mu, rho, CdA)
    % state = [x;y;vx;vy]
    x = state(1); y = state(2); vx = state(3); vy = state(4); vz= state(5);

    z = h(x,y);
    [hx, hy] = grad_h(x,y);  % gradiente alla superficie              
    n_unnorm = [-hx; -hy; 1];  % normale alla superficie
    n = n_unnorm / norm(n_unnorm);

    v3 = [vx; vy; vz];  
    vmag = norm(v3);


    % Forze
    F_p = m * g * sin(alpha_fun(x,y))*sin(beta);
    F_fric = - mu * m * g *sqrt(((cos(alpha_fun(x,y))).^2+ (sin(alpha_fun(x,y))*cos(beta)).^2 ));
    F_drag = -CdA *rho*0.5* vmag.^2;

    F_tot = (F_p+ F_fric + F_drag)/m;

    tangent_dir = [cos(beta); sin(beta)*cos(alpha_fun(x,y)); sin(beta)*sin(alpha_fun(x,y))]; 

    a = F_tot*tangent_dir;

    ax= a(1);
    ay = a(2);
    az = a(3);

    % Derivata di stato
    dst = [vx;        % dx/dt
           vy;        % dy/dt
           ax;        % dvx/dt
           ay;        % dvy/dt
           az];       % dvz/dt
end

function [value, isterminal, direction] = skiEvents(~, state, meta)
    % state = [x;y;vx;vy]
    x = state(1);  y = state(2);

    % --- condizioni di stop ---
    % 1) fondo pista: y = L
    value1 = meta.L - y;      % -> zero quando y = L (poi diventa negativo)
    % 2) uscita laterale: |x| = W/2
    value2 = meta.W/2 - abs(x);

    % output eventi 
    value      = [value1; value2];        % L'evento scatta quando il valore raggiunge lo 0 
    isterminal = [1; 1];                   % 1 = ferma integrazione
    direction  = [-1; -1];                 % zero raggiunto dall'alto
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

  