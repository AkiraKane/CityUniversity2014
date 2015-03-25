function y = activation(x)
    % Return the Log-sigmoid transfer function
    %y = 1 / (1 + exp(-x)); 
       
    % Return the Hyperbolic tangent sigmoid transfer function
    y = 2/(1+exp(-2*x))-1;
    
    % TO DO STRING TO FUNCTION AS SHOWN IN ARTURs SCRIPT