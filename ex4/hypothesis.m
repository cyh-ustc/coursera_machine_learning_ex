function  h  = hypothesis( theta, X )
%HYPOTHESIS 此处显示有关此函数的摘要
%   此处显示详细说明

h = sigmoid(X * theta);
end