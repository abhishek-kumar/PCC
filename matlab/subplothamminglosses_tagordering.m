function [ retval ] = subplothamminglosses_tagordering( inputfile, BRvalue, ylimits, ylbl )
    scene = load(inputfile);
    plot(scene(1:15,1), scene(1:15,2), '-bx')
    for i=1:15
    scenebr(i,1) = i;
    scenebr(i,2) = BRvalue;
    end
    hold on
    plot(scenebr(:,1), scenebr(:,2), '--r')
    
    ylim(ylimits)
    xlim([0,16])
    %xlabel('Beam size for inference')
    
    if ylbl==1
        ylabel('Hamming loss')
    end
    
    retval = 1;

end

