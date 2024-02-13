# DoAllAnalysis (mainpath=os.path.abspath(), doanalysis = , debug = False) in python

if nargin < 2
    doanalysis = 1:100; % doing all by default.
    % kind of stupid implementation, but I don't know a nice way of
    % getting length of all the stimuli.
    % beware when you have more than 100 stimuli
end

mainpath = slashappend(mainpath);


% a function that identifies what analysis is available, and does all the analysis
% in that folder.
%skipanalysis = [1]; % analysis that is ignored % movie now.

# load StimParams

## look for and create a copy
pind = strfind(mainpath, 'processed700');
if ~isempty(pind)
    localparampath = [mainpath(1:pind+25) 'StimParams'];
    parampath = [mainpath([1:pind-1 pind+13:pind+25]) 'StimParams' filesep];
else
    pind = strfind(lower(mainpath), 'processedl5');
    if ~isempty(pind)
        localparampath = [mainpath(1:pind+24) 'StimParams'];
        parampath = [mainpath([1:pind-1 pind+12:pind+24]) 'StimParams' filesep];
    else
        pind = strfind(mainpath, 'processed');
        localparampath = [mainpath(1:pind+22) 'StimParams'];
        parampath = [mainpath([1:pind-1 pind+10:pind+22]) 'StimParams' filesep];
    end
end

% if local one exists, use it, otherwise, copy it and use it
if exist(localparampath, 'dir')
    parampath = slashappend(localparampath);
else
    if exist(parampath, 'dir')
        copyfile(parampath, localparampath);
        parampath = slashappend(localparampath);
    else
        parampath = [];
    end
end

## load stimParams
load([mainpath 'segmentlengths']);

## n datasets
nData = length(segmentlengths); % number of data available

# basic analysis 


## makeEIsummary
makeEISummary(mainpath); % so far, only one

## load ttl times, save in processed data
ttlTimes = getTTLTimes(mainpath); % in ms
save([mainpath 'ttlTimes'], 'ttlTimes'); % for future use

## load order of stimulations
% Stimulus related analysis
% load([mainpath 'segmentstims']);

## add a zero to segmentseparations
ssep = [0 segmentseparations];

## set up parameters/datasets
stimfilenames = {};
segmentstims = {}; % name of stimulation for each segment
sf = {};
movement = {}; % mouse trackball movement information
isawake = 0; % 1 for awake recording (if a trackball file exists, this is set to 1)

for i = 1:length(timestamps)
    [stimfilenames{i} trackballnames{i}] = FindStimFile(timestamps(i), parampath);
        
    sf{i} = load(stimfilenames{i}); % all the stim info saved in this variable
    if ~isfield(sf{i}, 'intanSyncData') % old version
        try
            intfile = FindStimFile(timestamps(i), parampath, 'I'); % look for I files
            load(intfile);
            sf{i}.intanSyncData = intanSyncData;
        catch
            disp('Intan file may be broken?')
        end
    end
    if trackballnames{i} % if track ball is available
        tbr{i} = load(trackballnames{i});
        movement{i} = getMouseTrajectory_intan(tbr{i});
        isawake = 1;
    else
        tbr{i} = 0;
    end
    segmentstims{i} = rmSpace(sf{i}.StimulusStr{sf{i}.StimulusNum});
end

## Save all loaded arguments
save([mainpath 'segmentstims'], 'segmentstims', 'isawake', 'movement');

# Pre-procesing and basic analysis

## Load data and assess level of contamination
load([mainpath 'asdf'])
contamination = ASDFContaminationIndex(asdf_raw);
asdf = ASDFChangeBinning(asdf_raw, 0.05);

## Assess firing rates
acfine = ASDFCC(asdf, 1:200, 1);
asdf = ASDFChangeBinniACng(asdf_raw, 1);
accoarse = ASDFCC(asdf, 1:200, 1);
nNeu = asdf_raw{end}(1);

timebins = 0:5000:asdf_raw{end}(2); % in ms
FRhist = zeros(nNeu, length(timebins));
for i = 1:nNeu
    FRhist(i,:) = histc(asdf_raw{i}, timebins);
end

## Movement vectors
if isempty(movement)
    try
        movement = getMouseTrajectory_rotencoder(mainpath);
        tbr = movement;
    catch
    end
end

% determine segment firing rate for each neuron,
% and also firing rate of the first and last 5 mins
segfr = zeros(nNeu, length(segmentstims));
startfr = zeros(nNeu, length(segmentstims));
endfr = zeros(nNeu, length(segmentstims));
frevalduration = 5 * 60 * 1000; % 5 minues in ms
for i = 1:length(segmentstims)
    asdf_c = ASDFChooseTime(asdf_raw, ssep(i), ssep(i+1));
    segfr(:,i) = ASDFGetfrate(asdf_c) * 1000; % to get Hz
    asdf_s = ASDFChooseTime(asdf_raw, ssep(i), ssep(i) + frevalduration);
    startfr(:,i) = ASDFGetfrate(asdf_s) * 1000; % to get Hz
    asdf_e = ASDFChooseTime(asdf_raw, ssep(i+1)-frevalduration, ssep(i+1));
    endfr(:,i) = ASDFGetfrate(asdf_e) * 1000; % to get Hz

end

## deterimine which data segments are bad
badsegments = segfr < 0.1;
frbadneurons = sum(badsegments, 2);
badsegments_startend = startfr < 0.1 | endfr < 0.1;

%[frbadneurons, subfrhz] = PartFREvaluation(mainpath, 0.1); % bad neurons have <0.1 Hz at least in one segment

## Save this initial pre-processing step
save([mainpath 'FRACresults'], 'contamination', 'acfine', 'accoarse', 'FRhist',...
    'segfr', 'frbadneurons', 'startfr', 'endfr', 'badsegments_startend');

# LoadVision2 why?
## Load data.spikes (not sure what this is)
sfile = LoadVisionFiles([mainpath 'data*.spikes']);
arrayID = sfile.getHeader.arrayID;
sfile.close;

## Separate TTLs based on data segments
segttls = {};
for i = 1:length(segmentstims)
    segttl = ttlTimes(ttlTimes > ssep(i) & ttlTimes < ssep(i+1));
    segttl = TTLGlitchRemoval(segttl, sf{i});
    segttls{i} = segttl;
end

## save basicinfor file
save([mainpath 'basicinfo'], 'arrayID', 'movement', 'segttls', 'sf', '-append');

## make duplicate classification file
makeDupClassification(mainpath);

%%
for i = 1:length(segmentstims)
    if ~ismember(i, doanalysis)
        fprintf('Skipping %sAnalysis...\n', segmentstims{i});
        continue
    end
    fprintf('Running %sAnalysis...\n', segmentstims{i});
    
    %infodir = [mainpath segmentstims{i} 'Info'];
    %fh = eval(['@' segmentstims{i} 'Analysis']); % getting appropreate handle
    fh = str2func([segmentstims{i} 'Analysis']); % making handle, better than eval
    % get local TTL times
    %segttl = ttlTimes(ttlTimes>ssep(i) & ttlTimes<ssep(i+1));
    %segttl = TTLGlitchRemoval(segttl, sf{i});
    
    
    
    if debug
        fh(mainpath, sf{i}, segttls{i}, tbr{i}, i); % stim info is passed to it
        % 6/2/2016, added 5th argument i
    else
        try
            fh(mainpath, sf{i}, segttls{i}, tbr{i}, i); % stim info is passed to it
        catch e
            %warning(e);
            warning('Analysis execution failed on %s: %s\n%s\n', segmentstims{i}, e.identifier, e.message);
            e.stack(1)
            e.stack(2)
            %rethrow(e)
        end
    end
end

### NOT NEEDED?

    %% Additional Analysis for V1 

    load([mainpath 'basicinfo'], 'recordingregion')
    if exist('recordingregion', 'var') && strcmp(recordingregion, 'V1')
        EIAnalysis(mainpath)
    end

disp('Analysis Finished')


# DoAuditoryAnalysis(datafolder, doind)
% function DoAuditoryAnalysis(datafolder, doind)
% Do Auditory Analysis by automatic stimulus detection
if nargin < 1
    datafolder = pwd;
end
if nargin < 2
    doind = 0; % do all
end
datafolder = slashappend(datafolder);


% this analysis requires the user to put their analysis files in
auditorystimpath = '~/AuditoryAnalysisParams/';

% get the time stamp from the current directory.
load([datafolder, 'segmentlengths.mat']);
% this load 'timestamps', 'segmentlengths', 'segmentseparations'
sttlname = [datafolder, 'segttls'];
if ~exist(sttlname) % if not there, make it.
     makesegttls(datafolder);
end
load([datafolder, 'segttls']) % load it.

stimfiles = arrayfun(@(x) FindStimFile(x, auditorystimpath, 'A'),...
    timestamps, 'UniformOutput', false);
stimlab = cellfun(@ReadAuditoryLabViewFile, stimfiles);


if doind == 0
    doind = 1:length(stimlab);
end

for i = doind
    if contains(stimlab(i).stimname, 'RandomChord')

## DoRandomChordAnalysis

        DoRandomChordAnalysis (datafolder, stimlab(i), i);
    else

## makeAuditorySpotSummary

        makeAuditorySpotSummary (datafolder, stimlab(i), i);
    end
end