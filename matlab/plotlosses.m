function [ output_args ] = plotlosses( inputfile, outputfile, ylimits, idx, xlbl, ylbl, lgnd )
    scene = load(inputfile);
    clf;
    plot(scene(:,1), scene(:,idx), '-bx', 'LineWidth',2,'MarkerSize',10)
    
    ylim(ylimits)
    xlim([0,13])
    xlabel(xlbl)
    ylabel(ylbl)
    h=legend(lgnd)
    
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

