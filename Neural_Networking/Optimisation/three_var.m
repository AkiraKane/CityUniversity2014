function b = three_var(v)
    x = v(1);
    y = v(2);
    z = v(3);
    b = x.^2 + 2.5*sin(y) - z^2*x^2*y^2;