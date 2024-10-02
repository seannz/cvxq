%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Cumulative distribution function of the        %
%   Laplace distribution with MATLAB Implementation    %
%                                                      %
% Author: Ph.D. Eng. Hristo Zhivomirov        06/06/20 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fx = cdflaplace(x, mu, sigma)
% function: fx = cdflaplace(x, mu, sigma)
% x - values at which to evaluate the Laplace CDF;
% mu - mean of the Laplace distribution;
% sigma - standard deviation of the Laplace distribution. If the  
%         scale parameter b is given, then sigma = sqrt(2)*b;
% fx - Laplace CDF values.
% calculate the Laplace CDF
fx = 0.5*(1 + sign(x-mu).*(1 - exp( -abs(x-mu)/(sigma/sqrt(2)) )));
end