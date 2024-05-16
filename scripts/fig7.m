% close all;
clear all;

curves = load('../bdcurves/facebook_opt-125m_curves_32_log_comp_12.mat').data;

l = 4;
c = 9;

m = floor((l - 1) / 6) + 1;
step_sizes = squeeze(curves(l,m,c,:,:,1))';
dist_final = squeeze(curves(l,end,c,:,:,2))';
dist_block = squeeze(curves(l,m,c,:,:,2))';
dist_weight = squeeze(curves(l,m,c,:,:,3))';

figure();
[~,ind] = min(log2(dist_weight),[],1);

plot(min(0.5 * log2(dist_weight(ind,:)),[],1));
hold on;
plot(min(0.5 * log2(dist_block(ind,:)),[],1));
hold on;
plot(min(0.5 * log2(dist_final(ind,:)),[],1));
