clear all;
close all;

latent_t = dir('../losses_latent_t/model.*.mat');
[~, idx] = sort([latent_t.datenum]);
latent_t = {latent_t.name};
latent_t = {latent_t{idx}};

latent = dir('../losses_latent/model.*.mat');
[~, idx] = sort([latent.datenum]);
latent = {latent.name};
latent = {latent{idx}};

% grads = cell(length(latent_t), 1);
% datas = cell(length(latent_t), 1);
% batch_size = 64;

for l = 1:length(latent)
    temp = load(['../losses_latent_t/', latent_t{l}]);
    latent_t_datas{l} = temp.data;
    temp = load(['../losses_latent/', latent{l}]);
    latent_datas{l} = temp.data;

    savingC{l} = mean(0.5 * log2(mean(latent_t_datas{l})) - 0.5 * log2(latent_t_datas{l}));
    savingR{l} = mean(0.5 * log2(mean(latent_datas{l})) - 0.5 * log2(latent_datas{l}));

    linesC{l} = (0. * log2(mean(latent_t_datas{l})) - 0.5 * log2(latent_t_datas{l}));
    linesR{l} = (0. * log2(mean(latent_datas{l})) - 0.5 * log2(latent_datas{l}));

    temp = load(sprintf('../gradsq%02d_fp16.mat', l - 1));
    gradsC{l} = 0.5 * log2(mean(temp.weight.^2,2) .* mean(temp.grad,2));
    gradsR{l} = 0.5 * log2(mean(temp.weight.^2,1) .* mean(temp.grad,1));
end

figure(1);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf']);
bar(0.5:59.5, [savingR(:)';savingC(:)';]',1,'stacked', 'FaceAlpha', 0.8);
set(gca,'YGrid','on');
xticks(2:10:59);
yticks(0:4);
xticklabels(0:2:11);
xlim([0,54]);

figure(2);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf']);
h = area(sort(linesR{2}));
h.FaceAlpha = 0.8;
xlim([0,768]);
ylim([-2,6]);
xticks(0:192:768);
yticks(-6:2:6);
xtickangle(0);
grid on;
