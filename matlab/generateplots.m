tr = load('../output/sceneresults_training.txt');
plot(tr(:,1), tr(:,3), 'r:x')
hold on
plot(tr(:,1), tr(:,2), 'b:x')
xlabel('beam size')
ylabel('loss')
title('Beam search for inference on tag order <3,5,0,2,1,4>')
ts = load('../output/sceneresults_test.txt');
plot(ts(:,1), ts(:,3), 'r-x')
plot(ts(:,1), ts(:,2), 'b-x')
ylim([0 0.45])
legend('Training Subset loss', 'Training Hamming loss', 'Test Subset loss', 'Test Hamming loss')

% gof
gofgraph()

%%%%%
% inference graphs
%%%%%

% 0-1 loss

plot01losses('scene-test-fixedpcc.txt', 'scene-01', 0.530936454849, 0.338628762542, [0.32,0.59])
plot01losses('yeast-test-fixedpcc.txt', 'yeast-01', 0.846237731734, 0.741548527808, [0.72,0.89])
plot01losses('emotions-test-fixedpcc.txt', 'emotions-01', 0.792079207921, 0.683168316832, [0.67,0.82])


%%% hamming loss

plothamminglosses('scene-test-fixedpcc.txt', 'scene-hamming', 0.108556298774, 0.10019509476, [0.099,0.1105])
plothamminglosses('yeast-test-fixedpcc.txt', 'yeast-hamming', 0.198862751207, 0.203302695124, [0.198, 0.206])
plothamminglosses('emotions-test-fixedpcc.txt', 'emotions-hamming', 0.226072607261, 0.23597359736, [0.22,0.25])


% ranking loss

plotrankingloss('scene-test-fixedpcc.txt', 'scene-ranking', 0.454849498328, 0.591137123746, [0.43,0.64])
plotrankingloss('yeast-test-fixedpcc.txt', 'yeast-ranking', 6.42093784079, 7.30207197383, [6,8])
plotrankingloss('emotions-test-fixedpcc.txt', 'emotions-ranking', 1.28217821782, 1.41089108911, [1.25,1.5])


%%%%%
% tag ordering graphs
%%%%%

plotlosses('scene-tagordering-nll.txt', 'scene-tagordering-nll', [2340 2500], 6, 'Beam size for tag ordering', 'Unnormalized negative log-likelihood', 'PCC')

%%%%
% inference graphs all in one
%%%%

clf
set(gca, 'Color', 'none'); % Sets axes background


subplot(7,3,[4 7]); subplot01losses('emotions-test-fixedpcc.txt', 'emotions-01', 0.792079207921, 0.683168316832, [0.67,0.82], 1)
subplot(7,3,[5 8]); subplot01losses('scene-test-fixedpcc.txt', 'scene-01', 0.530936454849, 0.338628762542, [0.32,0.59],0)
subplot(7,3,[6 9]); subplot01losses('yeast-test-fixedpcc.txt', 'yeast-01', 0.846237731734, 0.741548527808, [0.72,0.89],0)

subplot(7,3,[10 13]); subplothamminglosses('emotions-test-fixedpcc.txt', 'emotions-hamming', 0.226072607261, 0.23597359736, [0.22,0.25], 1)
subplot(7,3,[11 14]); subplothamminglosses('scene-test-fixedpcc.txt', 'scene-hamming', 0.108556298774, 0.10019509476, [0.099,0.1105],0)
subplot(7,3,[12 15]); subplothamminglosses('yeast-test-fixedpcc.txt', 'yeast-hamming', 0.198862751207, 0.203302695124, [0.198, 0.206],0)

subplot(7,3,[16 19]); subplotrankingloss('emotions-test-fixedpcc.txt', 'emotions-ranking', 1.28217821782, 1.41089108911, [1.25,1.5],1)
subplot(7,3,[17 20]); subplotrankingloss('scene-test-fixedpcc.txt', 'scene-ranking', 0.454849498328, 0.591137123746, [0.43,0.64],0)
subplot(7,3,[18 21]); subplotrankingloss('yeast-test-fixedpcc.txt', 'yeast-ranking', 6.42093784079, 7.30207197383, [6,8],0)


%%%%
% tag ordering graphs, all in one
%%%%


clf
set(gca, 'Color', 'none'); % Sets axes background

subplot(9,3,[4 7]); subplot01losses_tagordering('emotions-tagordering-nll.txt', 0.792079207921, [0.64,0.82], 1)
subplot(9,3,[5 8]); subplot01losses_tagordering('scene-tagordering-nll.txt', 0.530936454849, [0.32,0.57],0)
subplot(9,3,[6 9]); subplot01losses_tagordering('yeast-tagordering-nll.txt', 0.846237731734, [0.73,0.86],0)

subplot(9,3,[10 13]); subplothamminglosses_tagordering('emotions-tagordering-nll.txt', 0.226072607261, [0.20,0.24], 1)
subplot(9,3,[11 14]); subplothamminglosses_tagordering('scene-tagordering-nll.txt', 0.108556298774, [0.1,0.117],0)
subplot(9,3,[12 15]); subplothamminglosses_tagordering('yeast-tagordering-nll.txt', 0.198862751207, [0.197, 0.215],0)

subplot(9,3,[16 19]); subplotrankingloss_tagordering('emotions-tagordering-nll.txt', 1.28217821782, [1.22,1.6],1)
subplot(9,3,[17 20]); subplotrankingloss_tagordering('scene-tagordering-nll.txt', 0.454849498328, [0.43,0.68],0)
subplot(9,3,[18 21]); subplotrankingloss_tagordering('yeast-tagordering-nll.txt', 6.42093784079, [6,8],0)

subplot(9,3,[22 25]); subplotnll_tagordering('emotions-tagordering-nll.txt', 1.28217821782, [590,605],1)
subplot(9,3,[23 26]); subplotnll_tagordering('scene-tagordering-nll.txt', 0.454849498328, [2300,2500],0)
subplot(9,3,[24 27]); subplotnll_tagordering('yeast-tagordering-nll.txt', 6.42093784079, [8400,8600],0)



%%%%
% tag ordering graphs, all losses in one
%%%%

clf
set(gca, 'Color', 'none'); % Sets axes background
subplot(3,3,[4 7]); subplot_tagordering('emotions-tagordering-nll.txt', [0.8 1.01], 1)
subplot(3,3,[5 8]); subplot_tagordering('scene-tagordering-nll.txt', [0.89 1.01],0)
subplot(3,3,[6 9]); subplot_tagordering('yeast-tagordering-nll.txt', [0.98 1.001],0)
