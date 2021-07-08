function [label,VdevMax,fdevMax, VLLUnbalanceMax] = runMetrics(dataLocation, sampleNum, figLocation)
    fprintf('Evaluating sample at %s\n',dataLocation);
    
    %% The preparation of the data for the metrics run:
    if exist('y','var')==1
        clear y;
    end
    
    fileName = dir(sprintf('%s/%d/*.mat', dataLocation, sampleNum)).name;
    absFileName = sprintf('%s/%d/%s', dataLocation, sampleNum, fileName);
    fprintf('Loading file: %s\n', absFileName);
    load(absFileName,'y');
    figFolder = sprintf('%s/%d', figLocation, sampleNum);
    fprintf('Figures location: %s\n', figFolder); 
    
    if ~exist(figFolder, 'dir')
        mkdir(figFolder);
    end
    
    %% Extracting the data vectors from the loaded .mat file: 
    timeVec = y.glog_time;
    Vbase = 4.16; % kV
    Fbase = 60;
    Va = y.GA1WA;
    Vb = y.GA1WB;
    Vc = y.GA1WC;

    Vab = Va - Vb;
    Vbc = Vb - Vc;
    Vca = Vc - Va;
    
    %% Calling the metrics function: 
    [status,vv,f,A,theta,r,vals]=adherence1399_680_source(timeVec, Vab, Vbc,Vca,...
        'Vbase',Vbase,'Fbase',Fbase,'plot',1,'Nfig',1);
     
    %% Saving the figures for observation:
    figure(1)
    figName = 'VoltageMagnitude';
    title(sprintf('Voltage Magnitude (sample %d)', sampleNum));
    saveas(gcf,sprintf('%s/%s.png', figFolder, figName));
    savefig(gcf,sprintf('%s/%s', figFolder, figName));

    figure(2);
    figName = 'VoltageFrequency';
    title(sprintf('Voltage Frequency (sample %d)', sampleNum));
    saveas(gcf,sprintf('%s/%s.png', figFolder, figName));
    savefig(gcf,sprintf('%s/%s', figFolder, figName));

    figure(3);
    figName = 'VoltageUnbalance';
    title(sprintf('Voltage L-L Imbalance (sample %d)', sampleNum));
    saveas(gcf,sprintf('%s/%s.png', figFolder, figName));
    savefig(gcf,sprintf('%s/%s', figFolder, figName));

    %% Preparation of the output:
    label = status;
    fdevMax = vals.fdevMax;
    VLLUnbalanceMax = vals.VLLUnbalanceMax;
    VdevMax = vals.VdevMax;
    
    % Closing all the plots:
    close all;
end