for t = 1:m
	% Get t-th training example
	a1_t = X(t, :);
	
	% Compute a2
	z2_t = Theta1 * a1';
	a2_t = sigmoid(z2_t);
	
	% Add bias element to a2_t
	a2_t = [ones(1, columns(a2_t)); a2_t];
	
	% Compute a3
	z3_t = Theta2 * a2_t;
	a3_t = sigmoid(z3_t);
	
	% Get t-th element of y - should be a 10 x 1 vector
	y_t = y(t, :)';
	
	% Compute d3 (for output layer)
	d3 = a3_t - y_t;
	
	% Compute g'(z2) (hidden layer)
	g_z2 = sigmoidGradient(Theta1 * a1');
	g_z2 = [ones(1, columns(g_z2)); g_z2];
	
	% Compute d2, removing first term
	d2 = (Theta2' * d3) .* g_z2;
	d2 = d2(2:end); 
		
	% Compute and accumulate deltas
	DeltaSum = 0;
	
	% Delta1 = d2 * a1_t;
	% Delta2 = d3 * a2_t';
	% DeltaSum += Delta1 + Delta2;
endfor;
