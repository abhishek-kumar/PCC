function [ retval ] = plothamminglosses( inputfile, outputfile, BRvalue, EXvalue, ylimits )
    scene = load(inputfile);
    clf;
    plot(scene(1:15,1), scene(1:15,2), '-bx', 'LineWidth',2,'MarkerSize',10)
    for i=1:15
    scenebr(i,1) = i;
    scenebr(i,2) = BRvalue;
    end
    hold on
    plot(scenebr(:,1), scenebr(:,2), '-.r', 'LineWidth',2,'MarkerSize',10)
    for i=1:15
    sceneex(i,1) = i;
    sceneex(i,2) = EXvalue;
    end
    plot(sceneex(:,1), sceneex(:,2), '--', 'Color',[0 0.7 0], 'LineWidth',2,'MarkerSize',10)
    ylim(ylimits)
    xlim([0,16])
    xlabel('Beam size for inference')
    ylabel('Hamming loss')
    h=legend('PCC', 'Binary Relevance........', 'PCC with b=infty')
    % Find the first text object (alpha) and change it.
      h1 = findobj(get(h,'Children'),'String','PCC');
      set(h1,'String','$\mbox{\ PCC}$','Interpreter','latex')

      % Find the second text object (W) and change it.
      h2 = findobj(get(h,'Children'),'String','Binary Relevance........');
      set(h2,'String','$\mbox{\ Binary\ Relevance}$','Interpreter','latex')
    % Find the second text object (W) and change it.
      h3 = findobj(get(h,'Children'),'String','PCC with b=infty');
      set(h3,'String','$\mbox{\ PCC\ with}\ \ \ \ b=\infty$','Interpreter','latex')

      h_xlabel = get(gca,'XLabel');
      set(h_xlabel,'FontSize',16);
      h_ylabel = get(gca,'YLabel');
      set(h_ylabel,'FontSize',16);

    ti = get(gca,'TightInset')
    set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
    set(gca,'units','centimeters')
    pos = get(gca,'Position');
    ti = get(gca,'TightInset');

    set(gcf, 'PaperUnits','centimeters');
    set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    saveas(gcf,outputfile,'pdf');
    retval = 1;

end

