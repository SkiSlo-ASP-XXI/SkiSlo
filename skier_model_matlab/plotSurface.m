function plotSurface(Fz, meta)
    res = max(meta.dx, meta.dy);
    [X,Y] = meshgrid(-meta.W/2:res:meta.W/2, 0:res:meta.L);
    Z = Fz(X,Y);
    surf(X,Y,Z, 'EdgeColor','none'); shading interp; axis equal
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]'); colorbar; view(35,35); grid on
end