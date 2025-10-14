function [pts, Xg, Yg, Zg, meta] = buildSlopePointCloud(opts)

    arguments
        opts.TerrainType= 1
        % Griglia
        opts.L        (1,1) double = 150     % lunghezza (asse Y, verso il basso)
        opts.W        (1,1) double = 38       % larghezza (asse X)
        opts.dx       (1,1) double = 1
        opts.dy       (1,1) double = 1
        % Override opzionali 
        opts.SlopeDeg double = NaN % pendenza media in gradi
        opts.Turns    double = NaN % S lungo la pista
        opts.A_rel    double = NaN %
        opts.Camber   double = NaN
        opts.BankStrength double = NaN
        opts.K_center double = NaN
        opts.SigmaX_rel double = NaN
        opts.MogulsAmp double = NaN
        opts.MogulsLambdaY double = NaN
        opts.MogulsLambdaX double = NaN
        opts.Phase    double = pi/6 % fase iniziale della serpentina
        % Rugosità/rumore a bassa frequenza
        opts.RoughAmp double = NaN          % ampiezza rumore frattale
        opts.RoughScaleY double = NaN       % scala di correlazione lungo Y
        opts.RoughScaleX double = NaN       % scala di correlazione lungo X
        % Riproducibilità
        opts.Seed     double = NaN
    end

    % --------- Normalizza TerrainType ---------
    if ~isfield(opts,'TerrainType'), opts.TerrainType = 1; end
    tv = opts.TerrainType;
    if isstring(tv) || ischar(tv)
        key = lower(string(tv));
        switch key
            case "t1", tsel = 1;
            case "t2", tsel = 2;
            case "t3", tsel = 3;
            otherwise, error('TerrainType deve essere "t1","t2","t3" o 1/2/3.');
        end
    elseif isnumeric(tv) && isscalar(tv) && ismember(tv,[1 2 3])
        tsel = tv;
    else
        error('TerrainType deve essere "t1","t2","t3" o 1/2/3.');
    end


    % --------- Preset di base ---------
    switch tsel
        case 1  
            P = struct( ...
                'SlopeDeg',     11, ...
                'Turns',        12, ...
                'A_rel',        1, ...
                'Camber',       0.08, ...
                'BankStrength', 0.06, ...
                'K_center',     2.0, ...
                'SigmaX_rel',   0.22, ...
                'MogulsAmp',    0.00, ...
                'MogulsLambdaY',12, ...
                'MogulsLambdaX', 8, ...
                'RoughAmp',     0.10, ...
                'RoughScaleY',  45, ...
                'RoughScaleX',  18);
        case 2  
            P = struct( ...
                'SlopeDeg',     18, ...
                'Turns',        18, ...
                'A_rel',        0.38, ...
                'Camber',       0.12, ...
                'BankStrength', 0.10, ...
                'K_center',     1.6, ...
                'SigmaX_rel',   0.18, ...
                'MogulsAmp',    0.28, ...
                'MogulsLambdaY',10, ...
                'MogulsLambdaX', 6, ...
                'RoughAmp',     0.16, ...
                'RoughScaleY',  35, ...
                'RoughScaleX',  14);
        % case 3
        %     P = struct( ...
        %         'SlopeDeg',      7, ...
        %         'Turns',         8, ...
        %         'A_rel',        0.30, ...
        %         'Camber',       0.05, ...
        %         'BankStrength', 0.03, ...
        %         'K_center',     1.2, ...
        %         'SigmaX_rel',   0.28, ...
        %         'MogulsAmp',    0.06, ...
        %         'MogulsLambdaY',22, ...
        %         'MogulsLambdaX',12, ...
        %         'RoughAmp',     0.06, ...
        %         'RoughScaleY',  60, ...
        %         'RoughScaleX',  22);
        case 3  % Piano inclinato semplice (solo pendenza media)
            P = struct( ...
                'SlopeDeg',      12, ...   % usa il tuo default; resta sovrascrivibile da opts.SlopeDeg
                'Turns',          0, ...
                'A_rel',          0, ...
                'Camber',         0, ...
                'BankStrength',   0, ...
                'K_center',       0, ...
                'SigmaX_rel',     0.2, ... % irrilevante qui, ma teniamo un valore valido
                'MogulsAmp',      0, ...
                'MogulsLambdaY', 10, ...   % irrilevanti
                'MogulsLambdaX', 10, ...
                'RoughAmp',       0, ...
                'RoughScaleY',   40, ...
                'RoughScaleX',   16);
    end

    % --------- Override da opts (se forniti) ---------
    fields = fieldnames(P);
    for i = 1:numel(fields)
        f = fields{i};
        if isfield(opts, f) && ~isnan(opts.(f))
            P.(f) = opts.(f);
        end
    end

    % Griglia
    xvec = -opts.W/2 : opts.dx :  opts.W/2;
    yvec = 0         : opts.dy :  opts.L;
    [Xg, Yg] = meshgrid(xvec, yvec);

    % Seed opzionale
    if ~isnan(opts.Seed)
        rng(opts.Seed, 'twister');
    end

    % Pendenza media
    S  = tand(P.SlopeDeg);
    Z0 = -S .* Yg;

    % Serpentina centrale con ampiezza leggermente modulata
    base_freq = 2*pi*(P.Turns/opts.L);
    A_base    = P.A_rel*(opts.W/2);
    A_y       = A_base .* (0.8 + 0.2*sin(3*base_freq*Yg));
    x_c       = A_y .* sin(base_freq*Yg + opts.Phase);

    % Canale centrale
    sigma_x   = P.SigmaX_rel * opts.W;
    Z_center  = -P.K_center .* exp(-((Xg - x_c).^2)/(2*sigma_x^2));

    % Camber laterale
    Z_camber  = P.Camber * ((2*Xg/opts.W).^2 - 1);

    % Banking interno curva
    dxcdY = A_y .* base_freq .* cos(base_freq*Yg + opts.Phase);
    denom = max(abs(dxcdY(:)));
    if denom < eps, denom = 1; end
    Z_bank = P.BankStrength * (Xg - x_c) .* (dxcdY / denom);

    % Moguls (se attive)
    if P.MogulsAmp ~= 0
        Z_moguls = P.MogulsAmp .* ...
                   sin(2*pi*Yg/P.MogulsLambdaY) .* ...
                   sin(2*pi*Xg/P.MogulsLambdaX);
    else
        Z_moguls = 0;
    end

    % Rugosità a bassa frequenza (per evitare superficie troppo “pulita”)
    Z_rough = 0;
    if P.RoughAmp > 0
        % perlin-like semplice: filtra rumore gaussiano con kernel gaussiano anisotropo
        szY = max(3, round(P.RoughScaleY/opts.dy));  % ampiezza filtro in celle
        szX = max(3, round(P.RoughScaleX/opts.dx));
        R   = randn(size(Xg));
        gY  = max(1, round(szY/3));
        gX  = max(1, round(szX/3));
        ky  = fspecial('gaussian', [2*gY+1, 1], szY/6);
        kx  = fspecial('gaussian', [1, 2*gX+1], szX/6);
        Rf  = conv2(conv2(R, ky, 'same'), kx, 'same');
        Rf  = Rf / max(1e-9, std(Rf(:)));
        Z_rough = P.RoughAmp * Rf;
    end

    % Superficie finale
    Zg = Z0 + Z_center + Z_camber + Z_bank + Z_moguls + Z_rough;

    % Nuvola di punti
    pts = [Xg(:), Yg(:), Zg(:)];

    % Meta info (parametri effettivi)
    meta = struct('TerrainType', tsel, 'Params', P, ...
                  'L', opts.L, 'W', opts.W, 'dx', opts.dx, 'dy', opts.dy);

end