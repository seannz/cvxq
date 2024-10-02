clear all;
close all;

figure(1);
% plot(0:0.01:4, 2 * 2 .^ (-2*(0:0.01:4)), 'k', ...
%      0:0.01:4, 8 * 2 .^ (-2*(0:0.01:4)), 'k');

newcolors = [1,1,1; 0.9,0.9,0.9; 0.8,0.8,0.8];
colororder(newcolors)

area(0:0.01:4, [2 * 2 .^ (-2*(0:0.01:4)); 6 * 2 .^ (-2*(0:0.01:4)); ...
                10 * 1 + 0 * (0:0.01:4)]' ,'LineWidth', 0.8);
hold on;

% slope = 3.8 * 2.^(-2 * 1.2) * -2 * log(2);
% offset = 3.8 * 2.^(-2 * 1.2) - slope * 1.2;
% plot(0:0.01:4, slope * (0:0.01:4) + offset + 1.04, 'b', ...
%      0:0.01:4, slope * (0:0.01:4) + offset + -0.825, 'b');

ax = gca;
xtickangle(0);
xticks(0:4);
yticks(0:4);
axis([0,4,0,4]);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' ',' '});
grid on;
pdfprint('cond-1.pdf', 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(2);
semilogy(0:0.01:4, (2 * log(2) * 2 * 2 .^ (-2*(0:0.01:4))), 'k', ...
         0:0.01:4, (2 * log(2) * 8 * 2 .^ (-2*(0:0.01:4))), 'k');

ax = gca;
% ax.MinorGridLineStyle = '-';
xtickangle(0);
xticks(0:4);
% yticks(0:4);
axis([0,4,0.1,10]);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' '});
grid on;
pdfprint('cond-2.pdf', 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(3);
semilogy((0:0.01:4), (2 * log(2) * 8 * 2 .^ (-2*(0:0.01:4))), 'LineColor', '#AAAAAA', ...
         round(0:0.001:4), (2 * log(2) * 8 * 2 .^ (-2*(0:0.001:4))), 'k');

ax = gca;
ax.MinorGridLineStyle = '-';
xtickangle(0);
xticks(0:4);
% yticks(0:4);
axis([0,4,0.1,10]);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' '});
grid on;
pdfprint('cond-3.pdf', 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);
