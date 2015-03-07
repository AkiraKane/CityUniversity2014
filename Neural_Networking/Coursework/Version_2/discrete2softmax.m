function s_m = discrete2softmax(inp,nmax)
    % Convert discrete data into softmax
    sNum = size(inp,1);
    s_m = zeros(sNum,nmax);
    x = inp' - 1;
    x( x < 0 )= 0; 
    s_m([1:sNum] + (x)*sNum) = 1;
    size(s_m);
end