% Generates the matRad reference tables of test/test_scenarios (run from MATLAB, matRad dev checkout)
%   matlab -batch "cd('D:\MATLAB\matRad'); matRad_rc; run('<this file>')"
% Output: one .mat (-v7) per configuration next to this script.
outDir = fileparts(mfilename('fullpath'));
if isempty(outDir), outDir = pwd; end

ct.numOfCtScen = 2;
ctProbs = {[1 1], [1 0.3; 2 0.7]};
combs = {'none', 'shift', 'all'};

cases = {};
for model = {'wcScen', 'impScen'}
    for c = 1:numel(combs)
        for cr = [true false]
            for p = 1:numel(ctProbs)
                s = struct('model', model{1}, 'combinations', combs{c}, 'combineRange', cr, ...
                    'ctScenProb', ctProbs{p});
                if strcmp(model{1}, 'impScen')
                    s.numOfSetupGridPoints = 3;
                    s.numOfRangeGridPoints = 3;
                end
                s.name = sprintf('%s_%s_cr%d_ct%d', model{1}, combs{c}, cr, p);
                cases{end + 1} = s; %#ok<AGROW>
            end
        end
    end
end

% default impScen (9 x 9 grid points)
cases{end + 1} = struct('model', 'impScen', 'combinations', 'none', 'combineRange', true, ...
    'ctScenProb', [1 1], 'name', 'impScen_default_none');
cases{end + 1} = struct('model', 'impScen', 'combinations', 'shift', 'combineRange', false, ...
    'ctScenProb', [1 0.3; 2 0.7], 'name', 'impScen_default_shift');
% non-default grid points, SDs and wcSigma
cases{end + 1} = struct('model', 'impScen', 'combinations', 'all', 'combineRange', true, ...
    'ctScenProb', [1 0.3; 2 0.7], 'numOfSetupGridPoints', 5, 'numOfRangeGridPoints', 3, ...
    'shiftSD', [1 2 3], 'rangeAbsSD', 0.5, 'rangeRelSD', 2, 'wcSigma', 2.5, ...
    'name', 'impScen_custom_all');
% odd number of points with inexact linspace middle, even number of points without nominal
cases{end + 1} = struct('model', 'impScen', 'combinations', 'none', 'combineRange', false, ...
    'ctScenProb', [1 1], 'numOfSetupGridPoints', 7, 'numOfRangeGridPoints', 5, ...
    'shiftSD', [1.1 2.3 0.7], 'rangeAbsSD', 0.3, 'rangeRelSD', 1.7, 'wcSigma', 3, ...
    'name', 'impScen_odd7_none');
cases{end + 1} = struct('model', 'impScen', 'combinations', 'shift', 'combineRange', true, ...
    'ctScenProb', [1 1], 'numOfSetupGridPoints', 4, 'numOfRangeGridPoints', 2, ...
    'name', 'impScen_even_shift');
% worst case with a single grid point
cases{end + 1} = struct('model', 'wcScen', 'combinations', 'all', 'combineRange', false, ...
    'ctScenProb', [1 1], 'numOfSetupGridPoints', 1, 'numOfRangeGridPoints', 3, ...
    'name', 'wcScen_one_setup_point');
% worst case with non-default SDs
cases{end + 1} = struct('model', 'wcScen', 'combinations', 'shift', 'combineRange', true, ...
    'ctScenProb', [1 0.3; 2 0.7], 'shiftSD', [1 2 3], 'rangeAbsSD', 0.5, 'rangeRelSD', 2, ...
    'wcSigma', 2, 'name', 'wcScen_custom_shift');

for k = 1:numel(cases)
    p = cases{k};
    switch p.model
        case 'wcScen', m = matRad_WorstCaseScenarios(ct);
        case 'impScen', m = matRad_ImportanceScenarios(ct);
    end
    f = fieldnames(p);
    isGrid = startsWith(f, 'numOf');
    f = [f(isGrid); f(~isGrid)];
    for i = 1:numel(f)
        if any(strcmp(f{i}, {'model', 'name'})), continue; end
        m.(f{i}) = p.(f{i});
    end
    save_fixture(outDir, p, m, ct);
end

% random scenarios: only the deterministic structure can be compared
m = matRad_RandomScenarios(ct);
m.nSamples = 4;
m.includeNominalScenario = true;
m.ctScenProb = [1 0.3; 2 0.7];
save_fixture(outDir, struct('model', 'rndScen', 'nSamples', 4, 'includeNominalScenario', true, ...
    'ctScenProb', [1 0.3; 2 0.7], 'name', 'rndScen_structure'), m, ct);

function save_fixture(outDir, p, m, ct)
s = struct();
s.model = m.shortName;
s.combinations = '';
s.combineRange = 1;
s.numOfSetupGridPoints = 0;
s.numOfRangeGridPoints = 0;
s.nSamples = 0;
s.includeNominalScenario = 0;
if isprop(m, 'combinations')
    s.combinations = m.combinations;
    s.combineRange = double(m.combineRange);
    s.numOfSetupGridPoints = m.numOfSetupGridPoints;
    s.numOfRangeGridPoints = m.numOfRangeGridPoints;
end
if isprop(m, 'nSamples')
    s.nSamples = m.nSamples;
    s.includeNominalScenario = double(m.includeNominalScenario);
end
s.rangeRelSD = m.rangeRelSD;
s.rangeAbsSD = m.rangeAbsSD;
s.shiftSD = m.shiftSD;
s.wcSigma = m.wcSigma;
s.ctScenProb = m.ctScenProb;
s.numOfAvailableCtScen = ct.numOfCtScen;
s.scenForProb = m.scenForProb;
s.scenProb = m.scenProb;
s.scenWeight = m.scenWeight;
s.linearMask = m.linearMask;
s.scenMask = double(m.scenMask);
s.scenMaskSize = size(m.scenMask);
s.totNumShiftScen = m.totNumShiftScen;
s.totNumRangeScen = m.totNumRangeScen;
s.totNumScen = m.totNumScen;
save(fullfile(outDir, [p.name '.mat']), '-struct', 's', '-v7');
end
