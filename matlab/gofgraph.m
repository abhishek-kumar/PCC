function [ retval ] = gofgraph( )

    %%%%
    % goodness of fit graph
    %%%%

    clf
    pcc1 = load('scene-fixedpcc-allinstances.txt');
    pcc1hl = pcc1(:,1);
    pcc5 = load('scene-fixedpcc-allinstances5.txt');
    pcc5hl = pcc5(:,1);
    br = load('scene-br-allinstances.txt');
    brhl = br(:,1);

    pb = polyfit(c', sort(brhl), 3);
    pp1 = polyfit(c', sort(pcc1hl), 3);
    pp5 = polyfit(c', sort(pcc5hl), 3);

    for i=1:1196
        c(i) = i;
    end

    hold on
    plot(c, sort(brhl), '-.r','LineWidth',2)
    plot(c, sort(pcc1hl), '--b','LineWidth',2)
    %plot(c, sort(pcc5hl), '--','Color', [0 0.7 0], 'LineWidth',2)

    plot(c, polyval(pb, c), '-.r');
    plot(c, polyval(pp1, c), '--b')
    %plot(c, polyval(pp5, c), '--', 'Color',[0 0.7 0])

    h=legend('Binary Relevance.......', 'PCC with beam size 1')
    xlabel('Instances in test set')
    ylabel('Proportion of wrongly predicted tags')
    ylim([-0.05 0.6])
    h_xlabel = get(gca,'XLabel');
    set(h_xlabel,'FontSize',16);
    h_ylabel = get(gca,'YLabel');
    set(h_ylabel,'FontSize',16);

    h2 = findobj(get(h,'Children'),'String','Binary Relevance.......');
    set(h2,'String','$\mbox{\ Binary\ Relevance}$','Interpreter','latex')

    h3 = findobj(get(h,'Children'),'String','PCC with beam size 1');
    set(h3,'String','$\mbox{\ PCC\ with}\ \ \ \ b=1$','Interpreter','latex')

    ti = get(gca,'TightInset')
    set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
    set(gca,'units','centimeters')
    pos = get(gca,'Position');
    ti = get(gca,'TightInset');

    set(gcf, 'PaperUnits','centimeters');
    set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    saveas(gcf,'scene-lossfunctions','pdf');
    retval = 1;
end

