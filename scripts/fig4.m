clear all;
close all;

% layers = dir('../params/model.decoder.layers.10.self_attn.*_proj.mat');
% layers = dir('../params/model.*.mat');
% layers = {layers([3,1,4]).name};
latent_t = dir('../losses_latent_t/model.*.self_attn.*.mat');
latent_t = {latent_t.name};

latent = dir('../losses_latent/model.*.self_attn.*.mat');
latent = {latent.name};

% grads = cell(length(latent_t), 1);
datas = cell(length(latent_t), 1);
% batch_size = 64;

for l = 1:length(latent)
    temp = load(['../losses_latent_t/', latent_t{l}]);
    latent_t_datas{l} = temp.data;
    temp = load(['../losses_latent/', latent{l}]);
    latent_datas{l} = temp.data;

    linesC{l} = (0. * log2(mean(latent_t_datas{l})) - 0.5 * log2(latent_t_datas{l}));
    linesR{l} = (0. * log2(mean(latent_datas{l})) - 0.5 * log2(latent_datas{l}));
    savingC{l} = mean(0.5 * log2(mean(latent_t_datas{l})) - 0.5 * log2(latent_t_datas{l}));
    savingR{l} = mean(0.5 * log2(mean(latent_datas{l})) - 0.5 * log2(latent_datas{l}));
end

savingC = [reshape(cell2mat(savingC), 4, []); zeros(1,12)];
savingC = savingC([1,4,3,2,5],:);

savingR = [reshape(cell2mat(savingR), 4, []); zeros(1,12)];
savingR = savingR([1,4,3,2,5],:);

figure(1);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
bar(0.5:59.5, [savingR(:)';savingC(:)';]',1,'stacked', 'FaceAlpha', 0.8);
set(gca,'YGrid','on')
xticks(2:10:59);
yticks(0:4);
xticklabels(0:2:11);
xlim([0,54]);

figure(2);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
h = area(sort(linesR{2}));
h.FaceAlpha = 0.8;
xlim([0,768]);
ylim([-2,6]);
xticks(0:192:768);
yticks(-6:2:6);
xtickangle(0);
grid on;

figure(3);
colororder(['#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
h = area(sort(linesC{2}));
h.FaceAlpha = 0.8;
xlim([0,768]);
ylim([-2,6]);
xticks(0:192:768);
yticks(-6:2:6);
xtickangle(0);
grid on;
