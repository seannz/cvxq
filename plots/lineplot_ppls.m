close all;
clear all;

filereg = {'../results/facebook-opt-125m_group_2_batch_size_16_bit_rate_3.00_pca_1024_no_taper_reset_fixed_lcna40.txt',...
           '../results/facebook-opt-125m_group_2_batch_size_16_bit_rate_4.00_pca_1024_no_taper_reset_fixed_lcna40.txt',...
           '../results/facebook-opt-2.7b_group_8_batch_size_16_bit_rate_3.00_pca_1024_no_taper_reset_fixed_lcna40_rand_p.txt',...
           '../results/facebook-opt-2.7b_group_8_batch_size_16_bit_rate_4.00_pca_1024_no_taper_reset_fixed_lcna40_rand_p.txt',...
           '../results/facebook-opt-6.7b_group_8_batch_size_16_bit_rate_3.00_pca_1024_no_taper_reset_fixed_lcna40.txt',...
           '../results/facebook-opt-6.7b_group_8_batch_size_16_bit_rate_4.00_pca_1024_no_taper_reset_fixed_lcna40.txt',...
           '../results/facebook-opt-30b_group_16_batch_size_16_bit_rate_3_pca_1024_no_taper_reset_fixed.txt',...
          };


% linecol = ['#1f77b4'; '#ff7f0e'; '#7f7f7f'; '#999999'; '#999999'];
linecol = ['#0000ff'; '#000000'; '#7f7f7f'; '#999999'; '#999999'];
linesty = {'-','-','-','-', '-'};
% linecol = {{'m.-'}, {'g.-'}, {'b.-'}, {'r.-'}, {'k.-'}, {'c.-'}};

alpha = 1;
tables = cell(1,1);
for f = 1:length(filereg)
    filename = filereg{f}; %, alphaset{f}{g}, deltaset{f}{g});
    disp(filename)
    recs = readtable(filename, 'Delimiter', ',', 'TrimNonNumeric', true);
    tables{f,1} = recs;
end

figure(1);
% yyaxis left;
colororder(linecol);
plot(tables{6,1}{2:end,1} / tables{6,1}{2,1}, smooth(tables{6,1}{2:end,[2]}, 0.05, 'lowess'), 'k',...
     tables{6,1}{2:end,1} / tables{6,1}{2,1}, smooth(tables{6,1}{2:end,[3]}, 0.05, 'lowess'), 'k');

% xlabel('Epoch ($10^3$ steps)');
% ylabel('PPL (WikiText2)');
axis([0,40,10,14]);
xticks(0:10:40);
% xticklabels({"0", "64", "128", "192", "256"});
xtickangle(0);
yticks(10:1:14);
xticklabels({' ',' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' ',' '});

% yyaxis right;
% plot(tables{6,1}{2:end,1}, smooth(tables{6,1}{2:end,[2]}, 0.1, 'lowess'));

grid on;
filename = 'ppls_1.pdf';
pdfprint(filename, 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(2);
%colororder(linecol);
plot(tables{4,1}{2:end,1} / tables{4,1}{2,1}, smooth(tables{4,1}{2:end,[2]}, 0.05, 'lowess'), 'k',...
     tables{4,1}{2:end,1} / tables{4,1}{2,1}, smooth(tables{4,1}{2:end,[3]}, 0.05, ...
                                  'lowess'), 'k');
% xlabel('Epoch ($10^3$ steps)');
% ylabel('PPL (WikiText2)');
axis([0,40,12,16]);
xticks(0:10:40);
% xticklabels({"0", "64", "128", "192", "256"});
xtickangle(0);
yticks(12:1:16);
xticklabels({' ',' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' ',' '});

grid on;
filename = 'ppls_2.pdf';
pdfprint(filename, 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(3);
colororder(linecol);
plot(tables{7,1}{2:end,1} / tables{7,1}{2,1}, smooth(tables{7,1}{2:end,[2]}, 0.05, 'lowess'), 'k',...
     tables{7,1}{2:end,1} / tables{7,1}{2,1}, smooth(tables{7,1}{2:end,[3]}, 0.05, 'lowess'), 'k');

% xlabel('Epoch ($10^3$ steps)');
% ylabel('PPL (WikiText2)');
axis([0,40,9,13]);
xticks(0:10:40);
% xticklabels({"0", "64", "128", "192", "256"});
xtickangle(0);
yticks(9:1:13);
xticklabels({' ',' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' ',' '});

grid on;
filename = 'ppls_3.pdf';
pdfprint(filename, 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);
