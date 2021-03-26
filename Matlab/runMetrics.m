function [label,fdevMax, VLLUnbalanceMax,VdevMax] = runMetrics(dataLocation, sampleNum, figLocation)
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
    
    timeVec = y.glog_time;
    Vbase = .45;
    Fbase = 60;
    Va = y.GA1WA;
    Vb = y.GA1WB;
    Vc = y.GA1WC;

    Vab = Va - Vb;
    Vbc = Vb - Vc;
    Vca = Vc - Va;
    
    [status,vv,f,A,theta,r,vals]=adherence1399_680_source(timeVec, Vab, Vbc,Vca,...
        'Vbase',Vbase,'Fbase',Fbase,'plot',1,'Nfig',1);
     
%% Saving the figures for observation:
    figure(1)
    figName = sprintf('%s/VoltageMagnitude.png',figFolder);
    saveas(gcf, figName);
    figure(2);
    figName = sprintf('%s/VoltageFrequency.png',figFolder);
    saveas(gcf, figName);
    figure(3);
    figName = sprintf('%s/VoltageUnbalance.png',figFolder);
    saveas(gcf, figName); 

%% Preparation of the output:
    label = status;
    fdevMax = vals.fdevMax;
    VLLUnbalanceMax = vals.VLLUnbalanceMax;
    VdevMax = vals.VdevMax;
    
    % Closing all the plots:
    close all;
end