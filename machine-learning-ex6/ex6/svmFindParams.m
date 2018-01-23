function [C, sigma] = svmFindParams(C_opt, sigma_opt, X, y, Xval, yval)

min_error = realmax;
C = 0;
sigma = 0;

for i = 1:length(C_opt)
	for j = 1:length(sigma_opt)
		
		C_temp = C_opt(1, i);
		sigma_temp = sigma_opt(1, j);
		
		% Train the model using svmTrain with temp variables
		model_temp = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
		pred_temp = svmPredict(model_temp, Xval);
		error_temp = mean(double(pred_temp ~= yval))
		
		if (error_temp < min_error)
			min_error = error_temp;
			C = C_temp;
			sigma = sigma_temp;
		endif
	end
end

end

