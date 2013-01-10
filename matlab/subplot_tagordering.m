function [ retval ] = subplot_tagordering( inputfile, ylimits, ylbl )
    scene = load(inputfile);
    ms = max(scene);
    %clf;
    hold on
    plot(scene(1:15,1), scene(1:15,4)/ms(4), ':m')
    plot(scene(1:15,1), scene(1:15,2)/ms(2), '-.rs')
    plot(scene(1:15,1), scene(1:15,3)/ms(3), '--v', 'Color', [0 0.7 0])
    plot(scene(1:15,1), scene(1:15,5)/ms(5), '-bp')
    legend('Ranking loss', 'Hamming loss', 'Subset 0/1 loss', 'Negative log-likelihood');
    
    ylim(ylimits)
    xlim([0,16])
    
    xlabel('Beam width for ordering tags')
    
    if ylbl == 1
        ylabel('Loss')
    end
    
    retval = 1;

end

