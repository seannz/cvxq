clear all;
close all;

rng(0);
weight = randn(4096,1);

figure(1);
yyaxis left;
ax = gca;
ax = gca;
ax.YAxis(1).Color = 'k';
histogram(weight, 'BinEdges', -4:8/16:4, 'Normalization', 'probability', ...
          'FaceColor', '#AAAAAA', 'FaceAlpha', 0.4, 'EdgeAlpha', 0.4);
xtickangle(0);
xticks(-4:2:4);
yticks(0:0.05:0.2);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' '});
axis([-4,4,0,0.2]);

yyaxis right;
ax = gca;
ax = gca;
ax.YAxis(2).Color = 'k';
plot(-4:0.0001:3.9999, 1/2 * (floor(2 * (-4:0.0001:3.9999)) + 0.5), ...
     'Color', 'blue');
yticks(-4:2:4);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' '});
grid on;
pdfprint('compand-1.pdf', 'Width', 11.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(2);
plot(-6:0.01:6, cdflaplace(-6:0.01:6, 0, 3), 'b');
ax = gca;
xtickangle(0);
xticks(-6:3:6);
yticks(0:0.25:1);
axis([-6,6,0,1]);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' ',' '});
grid on;
pdfprint('compand-2.pdf', 'Width', 10.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);

figure(3);
yyaxis left;
ax = gca;
ax.YAxis(1).Color = 'k';
histogram(weight, 'BinEdges', icdflaplace(0.06:0.88/16:0.94, 0, 3), ...
          'Normalization', 'probability', 'FaceColor', '#AAAAAA', ...
          'FaceAlpha', 0.4, 'EdgeAlpha', 0.4);
xtickangle(0);
xticks(-4:2:4);
yticks(0:0.05:0.2);
axis([-4,4,0,0.2]);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' '});

yyaxis right;
ax = gca;
ax.YAxis(2).Color = 'k';
plot(icdflaplace(0.06:0.0001:0.94, 0, 3), ...
     icdflaplace(0.88/16*(floor(((0.06:0.0001:0.94) - 0.06)/(0.88/16)) + 0.5)+0.06, 0, 3), 'Color', 'blue');
yticks(-4:2:4);
xticklabels({' ',' ',' ',' ',' '});
yticklabels({' ',' ',' ',' ',' '});
grid on;
pdfprint('compand-3.pdf', 'Width', 11.5, 'Height', 10, 'Position', [2, 1.5, 8, 8]);
