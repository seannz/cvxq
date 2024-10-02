clear all;
close all; 

layer = 5;

activations = load(sprintf('../activations_%02d.mat', layer)).data;
weight = load(sprintf('../weights%02d.mat', layer)).weight;
weight = weight - quantize(weight, 8, -11);

biases = load(sprintf('../weights%02d.mat', layer)).bias';

output = permute(reshape(weight * reshape(permute(activations,[3,1,2]),[],32*2048), [], 32, 2048),[2,3,1]);

figure;
histogram(output);

function y = quantize(weight, bitdepth, stepsize)
    y = 2^stepsize * min(2^(bitdepth-1) - 1, max(-2^(bitdepth-1), weight * (2^-stepsize)));
end
