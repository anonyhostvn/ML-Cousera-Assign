function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% disp(y) ; 

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    % disp(X) ;
    % disp(size(theta)) ; break ; 
    h = X * theta ;

    h = h - y ; 
    % disp(size(h)) ; break ;  

    h = X .* h ; 
    h = sum(h) ; 
    
    h = alpha * (1 / m) * (h') ; 
    theta = theta - h ; 
    % disp (theta) ; break  ; 

    % theta = theta - h' ; 




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
