clear all;
close all; 

activations = load('activations_10.mat').data;
weight = load('weights10.mat').weight;
biases = load('weights10.mat').bias';

output = permute(reshape(weight * reshape(permute(activations,[3,1, ...
                    2]),[],32*2048) + biases, [], 32, 2048),[2,3,1]);
rected = max(0, output);


histogram(reshape((load('weights10.mat').weight * ...
                   reshape(load('activations_10.mat').data,32*2048,768)')', 32, 2048, []))
