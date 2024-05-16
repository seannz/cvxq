clear all; 
close all;

load('../bdcurves/curves_1_2048.mat');
of_layer = 6;
at_layer_1 = 2;
at_layer_2 = 12;

figure();
plot(squeeze(data(of_layer,at_layer_1,1,:,:,1))', ...
     squeeze(log2(data(of_layer,at_layer_1,1,:,:,2)))');

figure();
plot(squeeze(data(of_layer,at_layer_2,1,:,:,1))', ...
     squeeze(log2(data(of_layer,at_layer_2,1,:,:,2)))');


figure();
plot(min(squeeze(log2(data(of_layer,at_layer_1,1,:,:,2))),[],2));
hold on; 
plot(min(squeeze(log2(data(of_layer,at_layer_2,1,:,:,2))),[],2));
