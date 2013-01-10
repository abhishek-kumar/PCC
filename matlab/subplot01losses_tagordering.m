function [ retval ] = subplot01losses_tagordering( inputfile, BRvalue, ylimits, ylbl )
    scene = load(inputfile);
    %clf;
    plot(scene(1:15,1), scene(1:15,3), '-bx')
    for i=1:15
    scenebr(i,1) = i;
    scenebr(i,2) = BRvalue;
    end
    hold on
    plot(scenebr(:,1), scenebr(:,2), '--r')
    
    ylim(ylimits)
    xlim([0,16])
    %xlabel('Beam width for inference')
    
    if ylbl==1
        ylabel('Subset 0-1 loss')
    end
    
    h=legend('PCC', 'Binary Relevance........')
    % Find the first text object (alpha) and change it.
      h1 = findobj(get(h,'Children'),'String','PCC');
      set(h1,'String','$\mbox{\ PCC}$','Interpreter','latex')

      % Find the second text object (W) and change it.
      h2 = findobj(get(h,'Children'),'String','Binary Relevance........');
      set(h2,'String','$\mbox{\ Binary\ Relevance}$','Interpreter','latex')
   
    retval = 1;
end

