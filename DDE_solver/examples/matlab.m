function dde_example
    % Delays
    lags = [1, 2, 3, 4];

    % Solve with dde23
    sol = dde23(@dde_rhs, lags, @history, [0, 10]);  % solve from 0 to 10

    % Plot solution
    figure;
    plot(sol.x, sol.y, 'LineWidth', 2);
    xlabel('t');
    ylabel('y(t)');
    title('Solution of the DDE');
    grid on;

    % Nested: RHS of the DDE
    function dydt = dde_rhs(t, y, Z)
        % Z(:,k) corresponds to y(t - lag(k))
        y1 = Z(1,1);   % y(t-1)
        y2 = Z(1,2);   % y(t-2)
        y3 = Z(1,3);   % y(t-3)
        y4 = Z(1,4);   % y(t-4)

        dydt = -y1 + y2 - y3*y4;
    end

    % Nested: history function
    function s = history(t)
        if t < 0
            s = 1;
        elseif abs(t) < 1e-12
            s = 0;   % enforce y(0)=0
        else
            s = 1;
        end
    end
end
