clear all;
close all;

% layers = dir('../params/model.decoder.layers.10.self_attn.*_proj.mat');
% layers = dir('../params/model.*.mat');
% layers = {layers([3,1,4]).name};
layers = dir('../params/model.*.self_attn.*.mat');
layers = {layers.name};

grads = cell(length(layers), 1);
datas = cell(length(layers), 1);
batch_size = 64;

for l = 1:length(layers)
    layer_data = load(['../params/', layers{l}]);
    datas{l} = layer_data.data;
    grads{l} = layer_data.grad;
    datas{l}(:, ~any(grads{l},1)) = [];  %columns
    grads{l}(:, ~any(grads{l},1)) = [];  %columns

    gradsqR{l} = diag(grads{l}' * grads{l});
    datasqR{l} = diag(datas{l}' * datas{l});
    prodsqR{l} = (gradsqR{l} .* datasqR{l});
    savingR{l} = 0.5 * log2(mean(prodsqR{l})) - 0.5 * log2(prodsqR{l});

    gradsqC{l} = diag(grads{l} * grads{l}');
    datasqC{l} = diag(datas{l} * datas{l}');
    prodsqC{l} = (gradsqC{l} .* datasqC{l});
    savingC{l} = 0.5 * log2(mean(prodsqC{l})) - 0.5 * log2(prodsqC{l});
end

figure(2);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
h = area(sort(savingC{13},'asc'));
h.FaceAlpha = 0.6;
hold on;
plot([1,768],[mean(savingC{13}),mean(savingC{13})], 'k');
mean(savingR{13});
xlim([0,768]);
ylim([-4,4]);
xticks(0:192:768);
yticks(-4:2:4);
xtickangle(0);
grid on;
pdfprint('fig3c.pdf', 'Width', 11, 'Height', 9, 'Position', [2.5, 1.5, 7.5, 7]);

figure(3);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
plot(log2(sort(cell2mat(prodsqR),'desc')./max(cell2mat(prodsqR),[],1)));
xlim([0,768]);
ylim([-16,0]);
xticks(0:192:768);
yticks(-16:4:0);
yticklabels({"$2^{-16}$","$2^{-12}$","$2^{-8}$","$2^{-4}$","$2^{-0}$"});
xtickangle(0);
grid on;
% pdfprint('fig3c.pdf', 'Width', 11, 'Height', 9, 'Position', [2.5, 1.5, 7.5, 7]);

% savings = -[mean(reshape(cell2mat(savingR),3,16),1); mean(reshape(cell2mat(savingC),3,16),1)];
% colororder(['#1f77b4';'#ff7f0e']);
% h = bar(savings','BarWidth',1);
% for i = 1:length(h)
%     h(i).FaceColor = colors(mod(i, length(colors)));
% end

figure(3);
colororder(['#1f77b4';'#ff7f0e';'#2ca02c';'#d62728';'#9467bd';'#8c564b';'#e377c2';'#7f7f7f';'#bcbd22';'#17becf'])
h = area(sort(savingR{13},'asc'));
h.FaceAlpha = 0.6;
hold on;
plot([1,768],[mean(savingR{13}),mean(savingR{13})], 'k');
mean(savingR{13});
xlim([0,768]);
ylim([-4,4]);
xticks(0:192:768);
yticks(-4:2:4);
xtickangle(0);
grid on;
pdfprint('fig3c.pdf', 'Width', 11, 'Height', 9, 'Position', [2.5, 1.5, 7.5, 7]);

