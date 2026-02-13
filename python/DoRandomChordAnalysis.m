function DoRandomChordAnalysis(datafolder, sl, seg)
% function DoRandomChordAnalysis(datafolder, sl, seg)

% Do a better spectrotemporal analysis and save the data

%datafolder = slashappend(pwd); % given
%seg = 2; % given
%sl = stimlab(2);

load([datafolder 'asdf'], 'asdf_raw');
load([datafolder 'segttls'], 'segttls');
load([datafolder, 'segmentlengths'], 'segmentseparations');
load([datafolder, 'EncoderLocomotion'], 'asdf_loco');

nNeu = asdf_raw{end}(1);

sn = sl.stimname;

if contains(sn, '4810')
    tn = 48; % tonenums
    tonedur = 10;
    fname = 'RandomChordStim_purerand4810';
elseif contains(sn, '4820')
    tn = 48;
    tonedur = 20;
    fname = 'RandomChordStim_purerand4820';
elseif contains(sn, '4805')
    tn = 48;
    tonedur = 20;
    fname = 'RandomChordStim_purerand4805';
else
    tn = 24;
    tonedur = 25;
    fname = 'RandomChordStim_purerand2405';
end
pt = load(['~/SpectroTemporalRFStimuli/' fname] , 'picked_tones');

nreps = length(sl.multipliers);
segst = segmentseparations(seg - 1);
segen = segmentseparations(seg);
asdf_seg = ASDFChooseTime(asdf_raw, segst, segen);
asdf_loco_seg = ASDFChooseTime(asdf_loco, segst, segen); % to save time
ttls = segttls{seg};
[asdf_seg_run, asdf_seg_stat] = getRunningSpikes(asdf_seg, asdf_loco_seg, 500);

[stas, stasigs, nSpikes, meanpic] = stacalc_sub(asdf_seg);
[stas_run, stasigs_run, nSpikes_run] = stacalc_sub(asdf_seg_run);
[stas_stat, stasigs_stat, nSpikes_stat] = stacalc_sub(asdf_seg_stat);

save([datafolder 'RandomChordResults_' num2str(seg)], 'stas', 'stasigs', 'stas_run', 'stasigs_run', 'stas_stat', 'stasigs_stat', 'nreps', 'tonedur', 'fname', 'nSpikes', 'nSpikes_run', 'nSpikes_stat', 'meanpic')


% here's a nested function. it might be a bad idea as it shares a bunch of
% variables with the main function. watch out.
    function [stas, stasigs, nSpikes, meanpic] = stacalc_sub(asdf)
        patnum = length(ttls) / nreps;
        inittimes = ttls(1:patnum:end) - segst;
        inittimes = [inittimes; segttls{seg}(end) + 25];
        
        pat_ttls = {};
        for i = 1:nreps
            offset = patnum * (i - 1);
            pat_ttls{i} = segttls{seg}((1:patnum) + offset) - segst - inittimes(i);
        end
        
        asdf_minis = cell(length(inittimes) - 1, 1);
        for i = 1:(length(inittimes)-1)
            asdf_minis{i} = ASDFChooseTime(asdf, inittimes(i), inittimes(i+1));
        end
        
        %% prepare STA picture
        
        fullpic = zeros(tn, patnum, 2);
        for j = 1:2
            for i = 1:patnum
                fullpic(pt.picked_tones{i, j}, i, j) = 1;
            end
        end
        %figure(33553);
        %imagesc(fullpic);
        picoffset = 9;
        fullpic = [ones(tn, picoffset, 2)/2, fullpic, ones(tn, 9-picoffset, 2)/2];
        partpic = zeros(tn, 10, patnum, 2);
        for i = 1:patnum
            partpic(:, :, i, :) = fullpic(:, i:i+9, :);
        end
        meanpic = mean(partpic, 3);
        
        
        %% let's do analysis per neuron
        %shortlist = [62 68 71 72 74 76 90 92 94 97 98 105 107 112 124 127 141 143 151 162 163 167];
        
        %figure(5000 + seg);
        %clf
        
        % separate plot and calculation. Do calculation first.
        stas = cell(nNeu, 1);
        stasigs = cell(nNeu, 1);
        nSpikes = zeros(nNeu, 1);
        tic;
        for neu = 1:nNeu
            fprintf('*');
            if mod(neu, 50) == 0
                fprintf(' %d\n', neu);
            end
            % 1. get spike timing % done by asdf_minis
            % 2. make correspondence with segment TTLs
            hc = zeros(1, patnum);
            for i = 1:nreps
                hc = hc + histcounts(asdf_minis{i}{neu}, [pat_ttls{i}; pat_ttls{i}(end) + 25]);
            end
            % 3. calculate a weighted average
            fhc = find(hc);
            fhcv = zeros(1, 1, length(fhc));
            fhcv(:) = hc(fhc);
            nSpike = sum(fhcv(:));
            nSpikes(neu) = nSpike;
            sizep = size(partpic);
            staraw = sum(partpic(:, :, fhc, :) .* repmat(fhcv, [sizep(1), sizep(2), 1, 2]), 3);
            stas{neu} = staraw / nSpike;
            
            % get significance calculation. takes about a second per neuron.
            stasigs{neu} = binocdf_unique(staraw, nSpike, meanpic);
        end
        fprintf(' done!\n');
        toc
    end
end

% %%
% for neu = 1:nNeu
%     if mod(neu-1, 25) == 0
%         figure(10000*seg + floor(neu/25));
%         setsize(15, 15);
%         clf
%     end
%     subplot(5, 5, mod(neu-1, 25)+1);
%
%     sta = stas{neu};
%     imagesc([sta(:, 6:end, 1, 1), sta(:, 6:end, 1, 2)]);
%     set(gca, 'ydir', 'normal');
%     set(gca, 'clim', [0.3 0.7]);
%     colormap viridis
%     colorbar
%     hold on
%     % vertical line in the middle
%     plot(5.5 * [1, 1], ylim, 'r');
%     xlabel('Time to spike (ms)');
%     xticks([0.5, 5.5 10.5]);
%     xticklabels({'-100', '0/-100', '0'});
%     yticks([0.5:12:49]);
%     yticklabels([5, 10, 20, 40, 80]);
%     ylabel('Frequency (kHz)');
%     ylim([24.5 48.5]);
%
%     fprintf('Neu:%3d, nSp: %5d\n', neu, sum(fhcv(:)));
%     title(neu)
% end

