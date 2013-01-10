function [ retval ] = subplotrankingloss_tagordering( inputfile, BRvalue, ylimits, ylbl )
    scene = load(inputfile);
    %clf;
    plot(scene(1:15,1), scene(1:15,5), '-bx')
    for i=1:15
    scenebr(i,1) = i;
    scenebr(i,2) = BRvalue;
    end
    %hold on
    %plot(scenebr(:,1), scenebr(:,2), ':r')
    
    ylim(ylimits)
    xlim([0,16])
    
    xlabel('Beam width for ordering tags')
    
    if ylbl == 1
        ylabel('Negative log-likelihood')
    end
    
    retval = 1;

end

