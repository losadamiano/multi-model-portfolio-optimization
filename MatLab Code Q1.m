%% --- MASTER SETUP
clear; clc; close all;

filename = 'dataexam_jan26.xlsx';

UB_VAL = 0.30;

opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
raw_nas_d = readtable(filename, 'Sheet', 'Nasdaq daily', 'VariableNamingRule', 'preserve');
raw_nas_m = readtable(filename, 'Sheet', 'Nasdaq monthly', 'VariableNamingRule', 'preserve');
raw_nys_d = readtable(filename, 'Sheet', 'Nyse daily', 'VariableNamingRule', 'preserve');
raw_nys_m = readtable(filename, 'Sheet', 'Nyse monthly', 'VariableNamingRule', 'preserve');

if height(raw_nas_d) > 2, raw_nas_d(1:2,:) = []; end
if height(raw_nas_m) > 2, raw_nas_m(1:2,:) = []; end
if height(raw_nys_d) > 2, raw_nys_d(1:2,:) = []; end
if height(raw_nys_m) > 2, raw_nys_m(1:2,:) = []; end

extract_dates = @(tbl) datetime(table2array(tbl(:,1)), 'InputFormat', 'dd/MM/yyyy');

dates_nas_d = extract_dates(raw_nas_d);
dates_nas_m = extract_dates(raw_nas_m);
dates_nys_d = extract_dates(raw_nys_d);
dates_nys_m = extract_dates(raw_nys_m);

parse_prices = @(tbl) str2double(strrep(string(table2cell(tbl(:, 2:end))), ',', '.'));

prices_nas_d = parse_prices(raw_nas_d);
prices_nas_m = parse_prices(raw_nas_m);
prices_nys_d = parse_prices(raw_nys_d);
prices_nys_m = parse_prices(raw_nys_m);

stock_names_nas = raw_nas_d.Properties.VariableNames(2:end);
stock_names_nys = raw_nys_d.Properties.VariableNames(2:end);

returns_nas_d = 100 * diff(log(prices_nas_d));
returns_nas_m = 100 * diff(log(prices_nas_m));
returns_nys_d = 100 * diff(log(prices_nys_d));
returns_nys_m = 100 * diff(log(prices_nys_m));

dates_ret_nas_d = dates_nas_d(2:end);
dates_ret_nas_m = dates_nas_m(2:end);
dates_ret_nys_d = dates_nys_d(2:end);
dates_ret_nys_m = dates_nys_m(2:end);

disp('>>> MASTER SETUP COMPLETED.');
disp(['Nasdaq Daily: ' num2str(size(returns_nas_d,1)) ' obs, ' num2str(size(returns_nas_d,2)) ' assets.']);
disp(['Nyse Daily:   ' num2str(size(returns_nys_d,1)) ' obs, ' num2str(size(returns_nys_d,2)) ' assets.']);

%% --- NASDAQ STATISTICS ---

calc_stats = @(x) [mean(x, 'omitnan'); std(x, 'omitnan'); var(x, 'omitnan'); skewness(x); kurtosis(x)];

stats_nas_d = array2table(calc_stats(returns_nas_d)', ...
    'VariableNames', {'Mean','Std','Var','Skew','Kurt'}, 'RowNames', stock_names_nas);

stats_nas_m = array2table(calc_stats(returns_nas_m)', ...
    'VariableNames', {'Mean','Std','Var','Skew','Kurt'}, 'RowNames', stock_names_nas);

fprintf('\n===========================================================\n');
fprintf('POINT 1: NASDAQ DAILY STATISTICS (ALL STOCKS)\n');
fprintf('===========================================================\n');
disp(stats_nas_d); 

fprintf('\n===========================================================\n');
fprintf('POINT 1: NASDAQ MONTHLY STATISTICS (ALL STOCKS)\n');
fprintf('===========================================================\n');
disp(stats_nas_m);

%% --- POINT 2: NYSE STATISTICS ---

stats_nys_d = array2table(calc_stats(returns_nys_d)', ...
    'VariableNames', {'Mean','Std','Var','Skew','Kurt'}, 'RowNames', stock_names_nys);

stats_nys_m = array2table(calc_stats(returns_nys_m)', ...
    'VariableNames', {'Mean','Std','Var','Skew','Kurt'}, 'RowNames', stock_names_nys);

fprintf('\n===========================================================\n');
fprintf('POINT 2: NYSE DAILY STATISTICS (ALL STOCKS)\n');
fprintf('===========================================================\n');
disp(stats_nys_d);

fprintf('\n===========================================================\n');
fprintf('POINT 2: NYSE MONTHLY STATISTICS (ALL STOCKS)\n');
fprintf('===========================================================\n');
disp(stats_nys_m);

%% --- COVARIANCE & CORRELATION ---
get_valid_assets = @(ret, names) ret(:, sum(~isnan(ret)) > (height(ret)*0.5));

r_nas_d_clean = get_valid_assets(returns_nas_d, stock_names_nas);
r_nas_m_clean = get_valid_assets(returns_nas_m, stock_names_nas);
r_nys_d_clean = get_valid_assets(returns_nys_d, stock_names_nys);
r_nys_m_clean = get_valid_assets(returns_nys_m, stock_names_nys);

Cov_nas_d = cov(r_nas_d_clean, 'omitrows');
Cov_nas_m = cov(r_nas_m_clean, 'omitrows');
Cov_nys_d = cov(r_nys_d_clean, 'omitrows');
Cov_nys_m = cov(r_nys_m_clean, 'omitrows');

Corr_nas_d = corr(r_nas_d_clean, 'rows', 'pairwise');
Corr_nas_m = corr(r_nas_m_clean, 'rows', 'pairwise');
Corr_nys_d = corr(r_nys_d_clean, 'rows', 'pairwise');
Corr_nys_m = corr(r_nys_m_clean, 'rows', 'pairwise');

fprintf('\n===========================================================\n');
fprintf('POINT 3: CORRELATION MATRICES (PREVIEW)\n');
fprintf('===========================================================\n');

n_show = 5;
disp('NASDAQ Daily Corr (Top 5 Valid):');
disp(array2table(Corr_nas_d(1:n_show,1:n_show)));

disp('NYSE Daily Corr (Top 5 Valid):');
disp(array2table(Corr_nys_d(1:n_show,1:n_show)));

plot_heatmap = @(mat, title_str) ...
    heatmap(mat, 'Title', title_str, 'Colormap', parula, 'ColorLimits', [-1 1], ...
    'GridVisible', 'off', 'CellLabelColor', 'none', ...
    'XDisplayLabels', repmat({''},size(mat,1),1), 'YDisplayLabels', repmat({''},size(mat,1),1));

figure('Name', 'Heatmap: Nasdaq Daily', 'Color', 'w', 'Position', [50, 50, 800, 700]);
plot_heatmap(Corr_nas_d, 'Nasdaq Daily Correlation (Valid Assets)');
xlabel('Assets'); ylabel('Assets');

figure('Name', 'Heatmap: Nasdaq Monthly', 'Color', 'w', 'Position', [100, 100, 800, 700]);
plot_heatmap(Corr_nas_m, 'Nasdaq Monthly Correlation (Valid Assets)');
xlabel('Assets'); ylabel('Assets');

figure('Name', 'Heatmap: NYSE Daily', 'Color', 'w', 'Position', [150, 150, 800, 700]);
plot_heatmap(Corr_nys_d, 'NYSE Daily Correlation (Valid Assets)');
xlabel('Assets'); ylabel('Assets');

figure('Name', 'Heatmap: NYSE Monthly', 'Color', 'w', 'Position', [200, 200, 800, 700]);
plot_heatmap(Corr_nys_m, 'NYSE Monthly Correlation (Valid Assets)');
xlabel('Assets'); ylabel('Assets');

%% --- NASDAQ SAMPLE SELECTION ---
target_stocks = { ...
    'APPLE', ...
    'MICROSOFT', ...
    'NVIDIA', ...
    'AMAZON.COM', ...
    'ALPHABET ''A''', ...
    'TESLA', ...
    'META PLATFORMS A', ...
    'BROADCOM', ...
    'COSTCO WHOLESALE', ...
    'NETFLIX', ...
    'CISCO SYSTEMS', ...
    'T-MOBILE US' ...
    };

idx_best_nas = [];
found_names = {};
fprintf('Searching for NASDAQ stocks...\n');
for i = 1:length(target_stocks)
    match = strcmpi(stock_names_nas, target_stocks{i});
    if any(match)
        idx = find(match, 1);
        idx_best_nas = [idx_best_nas; idx];
        found_names = [found_names; stock_names_nas(idx)];
    else
        fprintf('WARNING: "%s" not found in dataset.\n', target_stocks{i});
    end
end
sel_names_nas = stock_names_nas(idx_best_nas);
n_sel = length(sel_names_nas);
disp('Selected Assets:');
disp(sel_names_nas');

raw_sel_rets_d = returns_nas_d(:, idx_best_nas);
valid_rows_d = ~any(isnan(raw_sel_rets_d), 2);
sel_rets_nas_d = raw_sel_rets_d(valid_rows_d, :);
fprintf('Daily Observations: %d\n', length(sel_rets_nas_d));

raw_sel_rets_m = returns_nas_m(:, idx_best_nas);
valid_rows_m = ~any(isnan(raw_sel_rets_m), 2);
sel_rets_nas_m = raw_sel_rets_m(valid_rows_m, :);
fprintf('Monthly Observations: %d\n', length(sel_rets_nas_m));

figure('Name', 'P4: Nasdaq Correlation Structure', 'Color', 'w', 'Position', [50, 50, 1400, 600]);

subplot(1, 2, 1);
h1 = heatmap(sel_names_nas, sel_names_nas, corr(sel_rets_nas_d));
h1.Title = 'Correlation: DAILY';
h1.Colormap = parula;
h1.ColorLimits = [-1 1];

subplot(1, 2, 2);
h2 = heatmap(sel_names_nas, sel_names_nas, corr(sel_rets_nas_m));
h2.Title = 'Correlation: MONTHLY';
h2.Colormap = parula;
h2.ColorLimits = [-1 1];

sgtitle('Point 4: Nasdaq Correlation Analysis (Frequency Effect)', 'FontSize', 14);

%% --- NYSE SAMPLE SELECTION ---
n_sel = 12;

market_cap_nys = zeros(1, length(stock_names_nys));
target_stocks_nys = { ...
    'JP MORGAN CHASE & CO.', ...
    'BERKSHIRE', ...
    'LILLY', ...          
    'VISA', ...
    'CATERPILLAR', ...
    'GOLDMAN SACHS GP.', ...
    'EXXON', ...
    'JOHNSON & JOHNSON', ... 
    'PROCTER', ...
    'MASTERCARD', ...            
    'HOME DEPOT', ...
    'PFIZER' ...
    };

fprintf('Selecting NYSE stocks...\n');
for i = 1:length(target_stocks_nys)
    idx = find(contains(upper(stock_names_nys), target_stocks_nys{i}));
    if ~isempty(idx)
        market_cap_nys(idx(1)) = 1000 - (i*10); 
    else
        fprintf('WARNING: "%s" not found in NYSE dataset.\n', target_stocks_nys{i});
    end
end

[~, sorted_idx_cap] = sort(market_cap_nys, 'descend');
idx_best_nys = sorted_idx_cap(1:n_sel); 
sel_names_nys = stock_names_nys(idx_best_nys);

fprintf('\n===========================================================\n');
fprintf('POINT 5: SELECTED NYSE STOCKS (Top 12)\n');
fprintf('===========================================================\n');
disp(sel_names_nys');

raw_sel_rets_d = returns_nys_d(:, idx_best_nys);
valid_rows_d = ~any(isnan(raw_sel_rets_d), 2); 
sel_rets_nys_d = raw_sel_rets_d(valid_rows_d, :);
fprintf('Daily Observations: %d\n', length(sel_rets_nys_d));

if exist('returns_nys_m', 'var')
    raw_sel_rets_m = returns_nys_m(:, idx_best_nys);
    valid_rows_m = ~any(isnan(raw_sel_rets_m), 2);
    sel_rets_nys_m = raw_sel_rets_m(valid_rows_m, :);
    fprintf('Monthly Observations: %d\n', length(sel_rets_nys_m));
else
    warning('Variabile returns_nys_m non trovata. Impossibile estrarre dati mensili.');
end

figure('Name', 'P5: NYSE Correlation Structure', 'Color', 'w', 'Position', [150, 150, 1400, 600]);

subplot(1, 2, 1);
h1 = heatmap(sel_names_nys, sel_names_nys, corr(sel_rets_nys_d));
h1.Title = 'Correlation: DAILY (NYSE)';
h1.Colormap = parula;
h1.ColorLimits = [-1 1];

subplot(1, 2, 2);
if exist('sel_rets_nys_m', 'var')
    h2 = heatmap(sel_names_nys, sel_names_nys, corr(sel_rets_nys_m));
    h2.Title = 'Correlation: MONTHLY (NYSE)';
    h2.Colormap = parula;
    h2.ColorLimits = [-1 1];
end

sgtitle('Point 5: NYSE Correlation Analysis (Frequency Effect)', 'FontSize', 14);

%% --- PRICE EVOLUTION PLOTS ---
plot_data_nas_d = fillmissing(prices_nas_d(:, idx_best_nas), 'nearest');
plot_data_nas_m = fillmissing(prices_nas_m(:, idx_best_nas), 'nearest');

prices_nys_filled_d = fillmissing(prices_nys_d(:, idx_best_nys), 'nearest');
prices_nys_filled_m = fillmissing(prices_nys_m(:, idx_best_nys), 'nearest');

prices_nys_norm_d = zeros(size(prices_nys_filled_d));
for i = 1:size(prices_nys_filled_d, 2)
    col = prices_nys_filled_d(:, i);
    val_start = col(find(col > 0, 1)); 
    if isempty(val_start), val_start = 1; end
    prices_nys_norm_d(:, i) = 100 * (col / val_start);
end

prices_nys_norm_m = zeros(size(prices_nys_filled_m));
for i = 1:size(prices_nys_filled_m, 2)
    col = prices_nys_filled_m(:, i);
    val_start = col(find(col > 0, 1));
    if isempty(val_start), val_start = 1; end
    prices_nys_norm_m(:, i) = 100 * (col / val_start);
end

figure('Name', 'P6: Nasdaq Price Evolution', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

ax1 = subplot(2,1,1);
plot(dates_nas_d, plot_data_nas_d, 'LineWidth', 1.2);
title('Nasdaq Giants - Daily Prices'); 
ylabel('Price ($)'); grid on; axis tight;
legend(sel_names_nas, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 8);

ax2 = subplot(2,1,2);
plot(dates_nas_m, plot_data_nas_m, 'LineWidth', 1.5);
title('Nasdaq Giants - Monthly Prices'); 
ylabel('Price ($)'); grid on; axis tight;
legend(sel_names_nas, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 8);

linkaxes([ax1, ax2], 'x');

figure('Name', 'P6: NYSE Price Evolution', 'Color', 'w', 'Position', [150, 150, 1200, 800]);

ax3 = subplot(2,1,1);
plot(dates_nys_d, prices_nys_norm_d, 'LineWidth', 1.2);
title('NYSE Giants - Daily Normalized Prices (Base 100)'); 
ylabel('Rebased Price (Start=100)'); 
grid on; axis tight;
legend(sel_names_nys, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 8);

ax4 = subplot(2,1,2);
plot(dates_nys_m, prices_nys_norm_m, 'LineWidth', 1.5);
title('NYSE Giants - Monthly Normalized Prices (Base 100)'); 
ylabel('Rebased Price (Start=100)'); 
grid on; axis tight;
legend(sel_names_nys, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 8);

linkaxes([ax3, ax4], 'x');

%% --- NYSE MEAN-VARIANCE OPTIMAL PORTFOLIO ---

r_sel_nys_d = returns_nys_d(:, idx_best_nys);
r_sel_nys_d(any(isnan(r_sel_nys_d), 2), :) = [];
r_sel_nys_m = returns_nys_m(:, idx_best_nys);
r_sel_nys_m(any(isnan(r_sel_nys_m), 2), :) = [];

mu_d = mean(r_sel_nys_d)';      
Sigma_nys_d = cov(r_sel_nys_d); 

mu_m = mean(r_sel_nys_m)';      
Sigma_nys_m = cov(r_sel_nys_m); 

n = length(idx_best_nys);
ones_vec = ones(n, 1);

invSigma_d = Sigma_nys_d \ eye(n);
Z_d = invSigma_d * mu_d;      
w_nys_d = Z_d / sum(Z_d);       

invSigma_m = Sigma_nys_m \ eye(n);
Z_m = invSigma_m * mu_m;
w_nys_m = Z_m / sum(Z_m);

T_weights = table(sel_names_nys', round(w_nys_d, 4), round(w_nys_m, 4), ...
    'VariableNames', {'Asset', 'Weight_Daily_Tangency', 'Weight_Monthly_Tangency'});

fprintf('\n===========================================================\n');
fprintf('POINT 7: NYSE MEAN-VARIANCE OPTIMAL PORTFOLIO (TANGENCY)\n');
fprintf('===========================================================\n');
disp(T_weights);

figure('Name', 'P7: NYSE Tangency Allocation', 'Color', 'w', 'Position', [100, 100, 1200, 400]);

subplot(1,2,1);
bar(w_nys_d, 'FaceColor', [0 0.4470 0.7410]); 
title({'NYSE Optimal Portfolio (Daily)', 'Max Sharpe (Tangency)'});
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45); 
ylabel('Weight'); grid on; yline(0, 'k-', 'LineWidth', 1.5);
ylim([min([w_nys_d; w_nys_m])*1.1, max([w_nys_d; w_nys_m])*1.1]);

subplot(1,2,2);
bar(w_nys_m, 'FaceColor', [0.8500 0.3250 0.0980]); 
title({'NYSE Optimal Portfolio (Monthly)', 'Max Sharpe (Tangency)'});
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45);
ylabel('Weight'); grid on; yline(0, 'k-', 'LineWidth', 1.5);
ylim([min([w_nys_d; w_nys_m])*1.1, max([w_nys_d; w_nys_m])*1.1]);

sgtitle('Point 7: NYSE Mean-Variance Allocation Comparison', 'FontSize', 14);

%% --- NASDAQ MEAN-VARIANCE OPTIMAL PORTFOLIO ---
r_sel_nas_d = returns_nas_d(:, idx_best_nas);
r_sel_nas_d(any(isnan(r_sel_nas_d), 2), :) = [];
r_sel_nas_m = returns_nas_m(:, idx_best_nas);
r_sel_nas_m(any(isnan(r_sel_nas_m), 2), :) = [];

mu_d = mean(r_sel_nas_d)';
Sigma_nas_d = cov(r_sel_nas_d);

mu_m = mean(r_sel_nas_m)';
Sigma_nas_m = cov(r_sel_nas_m);

n = length(idx_best_nas);

invSigma_d = Sigma_nas_d \ eye(n);
Z_d = invSigma_d * mu_d;
w_nas_d = Z_d / sum(Z_d);

invSigma_m = Sigma_nas_m \ eye(n);
Z_m = invSigma_m * mu_m;
w_nas_m = Z_m / sum(Z_m);

T_weights_nas = table(sel_names_nas', round(w_nas_d, 4), round(w_nas_m, 4), ...
    'VariableNames', {'Asset', 'Weight_Daily_Tangency', 'Weight_Monthly_Tangency'});

fprintf('\n===========================================================\n');
fprintf('POINT 8: NASDAQ MEAN-VARIANCE OPTIMAL PORTFOLIO (TANGENCY)\n');
fprintf('===========================================================\n');
disp(T_weights_nas);

figure('Name', 'P8: Nasdaq Tangency Allocation', 'Color', 'w', 'Position', [150, 150, 1200, 400]);

subplot(1,2,1);
bar(w_nas_d, 'FaceColor', [0.4660 0.6740 0.1880]); 
title({'Nasdaq Optimal Portfolio (Daily)', 'Max Sharpe (Tangency)'});
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Weight'); grid on; yline(0, 'k-', 'LineWidth', 1.5);
ylim([min([w_nas_d; w_nas_m])*1.1, max([w_nas_d; w_nas_m])*1.1]);

subplot(1,2,2);
bar(w_nas_m, 'FaceColor', [0.9290 0.6940 0.1250]); 
title({'Nasdaq Optimal Portfolio (Monthly)', 'Max Sharpe (Tangency)'});
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Weight'); grid on; yline(0, 'k-', 'LineWidth', 1.5);
ylim([min([w_nas_d; w_nas_m])*1.1, max([w_nas_d; w_nas_m])*1.1]);

sgtitle('Point 8: Nasdaq Mean-Variance Allocation Comparison', 'FontSize', 14);

%% --- NASDAQ CONSTRAINED MEAN-VARIANCE (LONG-ONLY) ---

n = length(idx_best_nas);
opt = optimoptions('quadprog', 'Display', 'off');

lb = zeros(n, 1);    
ub = [];             
Aeq = ones(1, n);    
beq = 1;

r_sel_nas_d = returns_nas_d(:, idx_best_nas);
r_sel_nas_d(any(isnan(r_sel_nas_d), 2), :) = [];
Sigma_nas_d = cov(r_sel_nas_d);
mu_nas_d = mean(r_sel_nas_d)';

target_ret_d = mean(mu_nas_d); 
A_d = -mu_nas_d';
b_d = -target_ret_d;

w_nas_9_d = quadprog(Sigma_nas_d, zeros(n,1), A_d, b_d, Aeq, beq, lb, ub, [], opt);

if isempty(w_nas_9_d)
    disp('Target Daily aggressivo: Ottimizzazione adattata.');
    w_nas_9_d = quadprog(Sigma_nas_d, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
end

r_sel_nas_m = returns_nas_m(:, idx_best_nas);
r_sel_nas_m(any(isnan(r_sel_nas_m), 2), :) = [];
Sigma_nas_m = cov(r_sel_nas_m);
mu_nas_m = mean(r_sel_nas_m)';

target_ret_m = mean(mu_nas_m);
A_m = -mu_nas_m';
b_m = -target_ret_m;

w_nas_9_m = quadprog(Sigma_nas_m, zeros(n,1), A_m, b_m, Aeq, beq, lb, ub, [], opt);

if isempty(w_nas_9_m)
    w_nas_9_m = quadprog(Sigma_nas_m, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
end

T_weights_nas_constr = table(sel_names_nas', round(w_nas_9_d, 4), round(w_nas_9_m, 4), ...
    'VariableNames', {'Asset', 'Weight_Daily_MeanVar', 'Weight_Monthly_MeanVar'});

fprintf('\n===========================================================\n');
fprintf('POINT 9: NASDAQ CONSTRAINED MEAN-VARIANCE (Long-Only)\n');
fprintf('===========================================================\n');
disp(T_weights_nas_constr);

figure('Name', 'P9: Nasdaq Mean-Variance (Long-Only)', 'Color', 'w', 'Position', [100, 100, 1200, 450]);

subplot(1,2,1);
bar(w_nas_9_d, 'FaceColor', [0.6350 0.0780 0.1840]); 
title({'Nasdaq Mean-Variance (Daily)', 'Long-Only (No Cap)'});
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Weight'); grid on; 
top_lim_d = min(1, max(w_nas_9_d) * 1.1); 
if top_lim_d == 0, top_lim_d = 1; end
ylim([0 top_lim_d]); 

subplot(1,2,2);
bar(w_nas_9_m, 'FaceColor', [0.8500 0.3250 0.0980]); 
title({'Nasdaq Mean-Variance (Monthly)', 'Long-Only (No Cap)'});
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Weight'); grid on; 
top_lim_m = min(1, max(w_nas_9_m) * 1.1);
if top_lim_m == 0, top_lim_m = 1; end
ylim([0 top_lim_m]);

sgtitle('Point 9: Nasdaq Constrained Mean-Variance Allocation', 'FontSize', 14);

%% --- NYSE CONSTRAINED MEAN-VARIANCE (LONG-ONLY) ---

n = length(idx_best_nys);
opt = optimoptions('quadprog', 'Display', 'off');

lb = zeros(n, 1);   
ub = [];             
Aeq = ones(1, n);   
beq = 1;

r_sel_nys_d = returns_nys_d(:, idx_best_nys);
r_sel_nys_d(any(isnan(r_sel_nys_d), 2), :) = [];
Sigma_nys_d = cov(r_sel_nys_d);
mu_nys_d = mean(r_sel_nys_d)';

target_ret_d = mean(mu_nys_d); 

A_d = -mu_nys_d';
b_d = -target_ret_d;

w_nys_10_d = quadprog(Sigma_nys_d, zeros(n,1), A_d, b_d, Aeq, beq, lb, ub, [], opt);

if isempty(w_nys_10_d)
    disp('Target NYSE Daily aggressivo: Ottimizzazione adattata.');
    w_nys_10_d = quadprog(Sigma_nys_d, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
end

r_sel_nys_m = returns_nys_m(:, idx_best_nys);
r_sel_nys_m(any(isnan(r_sel_nys_m), 2), :) = [];
Sigma_nys_m = cov(r_sel_nys_m);
mu_nys_m = mean(r_sel_nys_m)';

target_ret_m = mean(mu_nys_m);
A_m = -mu_nys_m';
b_m = -target_ret_m;

w_nys_10_m = quadprog(Sigma_nys_m, zeros(n,1), A_m, b_m, Aeq, beq, lb, ub, [], opt);

if isempty(w_nys_10_m)
    w_nys_10_m = quadprog(Sigma_nys_m, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
end

T_weights_nys_constr = table(sel_names_nys', round(w_nys_10_d, 4), round(w_nys_10_m, 4), ...
    'VariableNames', {'Asset', 'Weight_Daily_MeanVar', 'Weight_Monthly_MeanVar'});

fprintf('\n===========================================================\n');
fprintf('POINT 10: NYSE CONSTRAINED MEAN-VARIANCE (Long-Only)\n');
fprintf('===========================================================\n');
disp(T_weights_nys_constr);

figure('Name', 'P10: NYSE Mean-Variance (Long-Only)', 'Color', 'w', 'Position', [150, 150, 1200, 450]);

subplot(1,2,1);
bar(w_nys_10_d, 'FaceColor', [0.4940 0.1840 0.5560]); 
title({'NYSE Mean-Variance (Daily)', 'Long-Only (No Cap)'});
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45);
ylabel('Weight'); grid on; 

top_lim_d = min(1, max(w_nys_10_d) * 1.15);
if top_lim_d <= 0, top_lim_d = 1; end
ylim([0 top_lim_d]);

subplot(1,2,2);
bar(w_nys_10_m, 'FaceColor', [0.7170 0.2740 1.0000]); 
title({'NYSE Mean-Variance (Monthly)', 'Long-Only (No Cap)'});
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45);
ylabel('Weight'); grid on; 

top_lim_m = min(1, max(w_nys_10_m) * 1.15);
if top_lim_m <= 0, top_lim_m = 1; end
ylim([0 top_lim_m]);

sgtitle('Point 10: NYSE Constrained Mean-Variance Allocation', 'FontSize', 14);

%% --- POINT 11: STATISTICS OF OPTIMAL NYSE PORTFOLIOS ---

r_sel_nys_d = returns_nys_d(:, idx_best_nys);
r_sel_nys_d(any(isnan(r_sel_nys_d), 2), :) = []; 
port_ret_nys_d = r_sel_nys_d * w_nys_10_d;

r_sel_nys_m = returns_nys_m(:, idx_best_nys);
r_sel_nys_m(any(isnan(r_sel_nys_m), 2), :) = [];
port_ret_nys_m = r_sel_nys_m * w_nys_10_m;

calc_stats = @(x) [mean(x); std(x); var(x); skewness(x); kurtosis(x)];

stats_vec_d = calc_stats(port_ret_nys_d);
stats_vec_m = calc_stats(port_ret_nys_m);

T_stats_nys = table(stats_vec_d, stats_vec_m, ...
    'VariableNames', {'NYSE_Optimal_Daily', 'NYSE_Optimal_Monthly'}, ...
    'RowNames', {'Mean', 'StdDev', 'Variance', 'Skewness', 'Kurtosis'});

fprintf('\n===========================================================\n');
fprintf('POINT 11: STATISTICS OF OPTIMAL NYSE PORTFOLIOS\n');
fprintf('===========================================================\n');
disp(T_stats_nys);

figure('Name', 'P11: NYSE Portfolio Return Distribution', 'Color', 'w', 'Position', [100, 100, 1000, 500]);

subplot(1,2,1);
histfit(port_ret_nys_d, 50, 'normal');
title('Daily Returns Distribution vs Normal');
xlabel('Return (%)'); ylabel('Frequency');
grid on; legend('Portfolio Returns', 'Normal Curve');
subtitle(['Kurtosis: ' num2str(stats_vec_d(5), '%.2f')]);

subplot(1,2,2);
histfit(port_ret_nys_m, 20, 'normal');
title('Monthly Returns Distribution vs Normal');
xlabel('Return (%)'); ylabel('Frequency');
grid on; legend('Portfolio Returns', 'Normal Curve');
subtitle(['Kurtosis: ' num2str(stats_vec_m(5), '%.2f')]);

sgtitle('Point 11: Analysis of Optimal NYSE Portfolio Distributions', 'FontSize', 14);

%% --- STATISTICS OF OPTIMAL NASDAQ PORTFOLIOS ---
r_sel_nas_d = returns_nas_d(:, idx_best_nas);
r_sel_nas_d(any(isnan(r_sel_nas_d), 2), :) = []; 
port_ret_nas_d = r_sel_nas_d * w_nas_9_d;

r_sel_nas_m = returns_nas_m(:, idx_best_nas);
r_sel_nas_m(any(isnan(r_sel_nas_m), 2), :) = []; 
port_ret_nas_m = r_sel_nas_m * w_nas_9_m;

calc_stats = @(x) [mean(x); std(x); var(x); skewness(x); kurtosis(x)];

stats_vec_nas_d = calc_stats(port_ret_nas_d);
stats_vec_nas_m = calc_stats(port_ret_nas_m);

T_stats_nas = table(stats_vec_nas_d, stats_vec_nas_m, ...
    'VariableNames', {'Nasdaq_Optimal_Daily', 'Nasdaq_Optimal_Monthly'}, ...
    'RowNames', {'Mean', 'StdDev', 'Variance', 'Skewness', 'Kurtosis'});

fprintf('\n===========================================================\n');
fprintf('POINT 12: STATISTICS OF OPTIMAL NASDAQ PORTFOLIOS\n');
fprintf('===========================================================\n');
disp(T_stats_nas);

figure('Name', 'P12: Nasdaq Portfolio Return Distribution', 'Color', 'w', 'Position', [150, 150, 1000, 500]);

subplot(1,2,1);
h_d = histfit(port_ret_nas_d, 50, 'normal');
set(h_d(1), 'FaceColor', [0.4660 0.6740 0.1880], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
set(h_d(2), 'Color', [0.2 0.2 0.2], 'LineWidth', 2);
title('Daily Returns Distribution vs Normal');
xlabel('Return (%)'); ylabel('Frequency');
grid on; legend(h_d, 'Portfolio Returns', 'Normal Curve');
subtitle(['Kurtosis: ' num2str(stats_vec_nas_d(5), '%.2f')]);

subplot(1,2,2);
h_m = histfit(port_ret_nas_m, 20, 'normal');
set(h_m(1), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
set(h_m(2), 'Color', [0.2 0.2 0.2], 'LineWidth', 2); 
title('Monthly Returns Distribution vs Normal');
xlabel('Return (%)'); ylabel('Frequency');
grid on; legend(h_m, 'Portfolio Returns', 'Normal Curve');
subtitle(['Kurtosis: ' num2str(stats_vec_nas_m(5), '%.2f')]);

sgtitle('Point 12: Analysis of Optimal Nasdaq Portfolio Distributions', 'FontSize', 14);

%% --- EFFICIENT FRONTIER - NASDAQ (Long-Only) ---
mu_d = mean(returns_nas_d(:, idx_best_nas), 'omitnan')';
Sigma_d = cov(returns_nas_d(:, idx_best_nas), 'omitrows');
mu_m = mean(returns_nas_m(:, idx_best_nas), 'omitnan')';
Sigma_m = cov(returns_nas_m(:, idx_best_nas), 'omitrows');

n = 12; 
lb = zeros(n, 1);
ub = ones(n, 1);
opt = optimoptions('quadprog', 'Display', 'off');
n_pts = 50;

w_gmv_d = quadprog(Sigma_d, zeros(n,1), [], [], ones(1,n), 1, lb, ub, [], opt);
mu_min_d = w_gmv_d' * mu_d;
targets_d = linspace(mu_min_d, max(mu_d), n_pts);
vols_d = zeros(size(targets_d));

for i = 1:n_pts
    w = quadprog(Sigma_d, zeros(n,1), [], [], [ones(1,n); mu_d'], [1; targets_d(i)], lb, ub, [], opt);
    if ~isempty(w)
        vols_d(i) = sqrt(w' * Sigma_d * w);
    else
        vols_d(i) = NaN;
    end
end

w_gmv_m = quadprog(Sigma_m, zeros(n,1), [], [], ones(1,n), 1, lb, ub, [], opt);
mu_min_m = w_gmv_m' * mu_m;

targets_m = linspace(mu_min_m, max(mu_m), n_pts);
vols_m = zeros(size(targets_m));

for i = 1:n_pts
    w = quadprog(Sigma_m, zeros(n,1), [], [], [ones(1,n); mu_m'], [1; targets_m(i)], lb, ub, [], opt);
    if ~isempty(w)
        vols_m(i) = sqrt(w' * Sigma_m * w);
    else
        vols_m(i) = NaN;
    end
end

figure('Name', 'P13: Nasdaq Efficient Frontiers', 'Color', 'w', 'Position', [100, 100, 800, 800]);

subplot(2,1,1); 
plot(vols_d, targets_d, 'b-', 'LineWidth', 2); hold on;
scatter(sqrt(diag(Sigma_d)), mu_d, 30, 'r', 'filled', 'MarkerEdgeColor', 'k');
title('Nasdaq Daily Efficient Frontier (Truncated at GMV)'); 
xlabel('Risk (Std)'); ylabel('Return'); grid on;

subplot(2,1,2); 
plot(vols_m, targets_m, 'Color', [0.85 0.33 0.1], 'LineWidth', 2); hold on;
scatter(sqrt(diag(Sigma_m)), mu_m, 30, 'b', 'filled', 'MarkerEdgeColor', 'k');
title('Nasdaq Monthly Efficient Frontier (Truncated at GMV)'); 
xlabel('Risk (Std)'); ylabel('Return'); grid on;

%% --- EFFICIENT FRONTIER - NYSE (Long-Only) ---
mu_dn = mean(returns_nys_d(:, idx_best_nys), 'omitnan')';
Sigma_dn = cov(returns_nys_d(:, idx_best_nys), 'omitrows');
mu_mn = mean(returns_nys_m(:, idx_best_nys), 'omitnan')';
Sigma_mn = cov(returns_nys_m(:, idx_best_nys), 'omitrows');

n = length(idx_best_nys);
lb = zeros(n, 1);
ub = ones(n, 1);
opt = optimoptions('quadprog', 'Display', 'off');
n_pts = 50;

w_gmv_d = quadprog(Sigma_dn, zeros(n,1), [], [], ones(1,n), 1, lb, ub, [], opt);
min_ret_d = w_gmv_d' * mu_dn;
max_ret_d = max(mu_dn);

targets_dn = linspace(min_ret_d, max_ret_d, n_pts);
vols_dn = zeros(size(targets_dn));

for i = 1:n_pts
    w = quadprog(Sigma_dn, zeros(n,1), [], [], [ones(1,n); mu_dn'], [1; targets_dn(i)], lb, ub, [], opt);
    if ~isempty(w)
        vols_dn(i) = sqrt(w' * Sigma_dn * w);
    else
        vols_dn(i) = NaN;
    end
end

w_gmv_m = quadprog(Sigma_mn, zeros(n,1), [], [], ones(1,n), 1, lb, ub, [], opt);
min_ret_m = w_gmv_m' * mu_mn;
max_ret_m = max(mu_mn);

targets_mn = linspace(min_ret_m, max_ret_m, n_pts);
vols_mn = zeros(size(targets_mn));

for i = 1:n_pts
    w = quadprog(Sigma_mn, zeros(n,1), [], [], [ones(1,n); mu_mn'], [1; targets_mn(i)], lb, ub, [], opt);
    if ~isempty(w)
        vols_mn(i) = sqrt(w' * Sigma_mn * w);
    else
        vols_mn(i) = NaN;
    end
end

figure('Name', 'P14: NYSE Efficient Frontiers', 'Color', 'w', 'Position', [150, 150, 800, 800]);

subplot(2,1,1);
plot(vols_dn, targets_dn, 'Color', [0.5 0 0.5], 'LineWidth', 2); hold on;
scatter(sqrt(diag(Sigma_dn)), mu_dn, 30, 'r', 'filled', 'MarkerEdgeColor', 'k');
title('NYSE Daily Efficient Frontier');
xlabel('Risk (Std)'); ylabel('Return'); grid on;

subplot(2,1,2);
plot(vols_mn, targets_mn, 'Color', [0.7 0.3 1.0], 'LineWidth', 2); hold on;
scatter(sqrt(diag(Sigma_mn)), mu_mn, 30, 'b', 'filled', 'MarkerEdgeColor', 'k');
title('NYSE Monthly Efficient Frontier');
xlabel('Risk (Std)'); ylabel('Return'); grid on;

alpha_95 = 0.05;
alpha_99 = 0.01;
z_95 = norminv(1 - alpha_95);
z_99 = norminv(1 - alpha_99);

mu_p_nas = mean(port_ret_nas_d);
std_p_nas = std(port_ret_nas_d);
VaR_95_nas = -(mu_p_nas - z_95 * std_p_nas);
VaR_99_nas = -(mu_p_nas - z_99 * std_p_nas);

mu_p_nys = mean(port_ret_nys_d);
std_p_nys = std(port_ret_nys_d);
VaR_95_nys = -(mu_p_nys - z_95 * std_p_nys);
VaR_99_nys = -(mu_p_nys - z_99 * std_p_nys);

T_VaR = table([VaR_95_nas; VaR_99_nas], [VaR_95_nys; VaR_99_nys], ...
    'VariableNames', {'Nasdaq_VaR', 'NYSE_VaR'}, ...
    'RowNames', {'VaR 95% (Daily)', 'VaR 99% (Daily)'});

disp(T_VaR);

figure('Name', 'P15: Frontier Comparison', 'Color', 'w', 'Position', [100, 100, 800, 600]);
plot(vols_d, targets_d, 'b-', 'LineWidth', 2); hold on;
plot(vols_dn, targets_dn, 'm-', 'LineWidth', 2);
legend('Nasdaq Frontier', 'NYSE Frontier', 'Location', 'best');
title('Comparison: Nasdaq vs NYSE Efficient Frontiers');
xlabel('Risk (Std Deviation)'); ylabel('Expected Return');
grid on;

%% --- MARKET INDEX ANALYSIS & DISCUSSION ---

row_labels = {'Mean', 'Std_Dev', 'Variance', 'Skewness', 'Kurtosis'};
freqs = {'Daily', 'Monthly'};
filename = 'dataexam_jan26.xlsx';

raw_idx_d = readtable(filename, 'Sheet', 'indexes daily', 'VariableNamingRule', 'preserve');
raw_idx_m = readtable(filename, 'Sheet', 'indexes monthly', 'VariableNamingRule', 'preserve');

if height(raw_idx_d) > 2, raw_idx_d(1:2, :) = []; end
if height(raw_idx_m) > 2, raw_idx_m(1:2, :) = []; end

parse_idx = @(tbl) str2double(strrep(string(table2cell(tbl(:, 2:end))), ',', '.'));
P_idx_d = parse_idx(raw_idx_d);
P_idx_m = parse_idx(raw_idx_m);
names_idx = raw_idx_d.Properties.VariableNames(2:end);

r_idx_d = 100 * diff(log(P_idx_d));
r_idx_m = 100 * diff(log(P_idx_m));

calc_stats = @(x) [mean(x, 'omitnan'); std(x, 'omitnan'); var(x, 'omitnan'); skewness(x); kurtosis(x)];

stats_mat_d = NaN(5, length(names_idx));
for i = 1:length(names_idx)
    col = r_idx_d(:, i); col(isnan(col)) = [];
    if length(col) > 10, stats_mat_d(:, i) = calc_stats(col); end
end
T_idx_d = rmmissing(array2table(stats_mat_d', 'VariableNames', row_labels, 'RowNames', names_idx));

stats_mat_m = NaN(5, length(names_idx));
for i = 1:length(names_idx)
    col = r_idx_m(:, i); col(isnan(col)) = [];
    if length(col) > 2, stats_mat_m(:, i) = calc_stats(col); end
end
T_idx_m = rmmissing(array2table(stats_mat_m', 'VariableNames', row_labels, 'RowNames', names_idx));

fprintf('\n===========================================================\n');
fprintf('POINT 15: MARKET INDEX STATISTICS\n');
fprintf('===========================================================\n');
disp('--- DAILY INDEX STATS ---'); disp(T_idx_d);
disp('--- MONTHLY INDEX STATS ---'); disp(T_idx_m);

find_bench = @(names, keyword) find(contains(upper(names), upper(keyword)), 1);

idx_sp = find_bench(names_idx, 'S&P'); if isempty(idx_sp), idx_sp = 1; end
name_sp = names_idx{idx_sp};

fprintf('\n=== DISCUSSION A: NYSE PORTFOLIO vs %s ===\n', upper(name_sp));
if exist('stats_vec_d', 'var')
    T_comp_nys_d = table(stats_vec_d, stats_mat_d(:, idx_sp), ...
        'VariableNames', {'NYSE_Portfolio', ['Benchmark_' name_sp]}, 'RowNames', row_labels);
    disp('--- DAILY COMPARISON ---'); disp(T_comp_nys_d);
    
    diff_ret = stats_vec_d(1) - stats_mat_d(1, idx_sp);
    diff_vol = stats_vec_d(2) - stats_mat_d(2, idx_sp);
    fprintf('Analysis: Return Diff: %.4f%% | Volatility Diff: %.4f%%\n', diff_ret, diff_vol);
    if diff_vol < 0, fprintf('Result: Portfolio is LESS volatile than the market (Defensive).\n'); end
end
if exist('stats_vec_m', 'var')
    T_comp_nys_m = table(stats_vec_m, stats_mat_m(:, idx_sp), ...
        'VariableNames', {'NYSE_Portfolio', ['Benchmark_' name_sp]}, 'RowNames', row_labels);
    disp('--- MONTHLY COMPARISON ---'); disp(T_comp_nys_m);
end

idx_nas = find_bench(names_idx, 'Nasdaq'); if isempty(idx_nas), idx_nas = 2; end
name_nas = names_idx{idx_nas};

fprintf('\n=== DISCUSSION B: NASDAQ PORTFOLIO vs %s ===\n', upper(name_nas));
if exist('stats_vec_nas_d', 'var')
    T_comp_nas_d = table(stats_vec_nas_d, stats_mat_d(:, idx_nas), ...
        'VariableNames', {'Nasdaq_Portfolio', ['Benchmark_' name_nas]}, 'RowNames', row_labels);
    disp('--- DAILY COMPARISON ---'); disp(T_comp_nas_d);
    
    diff_ret_nas = stats_vec_nas_d(1) - stats_mat_d(1, idx_nas);
    diff_vol_nas = stats_vec_nas_d(2) - stats_mat_d(2, idx_nas);
    fprintf('Analysis: Return Diff: %.4f%% | Volatility Diff: %.4f%%\n', diff_ret_nas, diff_vol_nas);
end
if exist('stats_vec_nas_m', 'var')
    T_comp_nas_m = table(stats_vec_nas_m, stats_mat_m(:, idx_nas), ...
        'VariableNames', {'Nasdaq_Portfolio', ['Benchmark_' name_nas]}, 'RowNames', row_labels);
    disp('--- MONTHLY COMPARISON ---'); disp(T_comp_nas_m);
end

%% --- BETA CALCULATION - NYSE ---
if ~exist('w_nys_10_d', 'var') || ~exist('r_idx_d', 'var')
    error('Esegui prima P10 e P15.');
end

idx_bench = find(contains(names_idx, 'S&P', 'IgnoreCase', true), 1);
if isempty(idx_bench), idx_bench = 1; end

r_stocks_d = returns_nys_d(:, idx_best_nys);
r_mkt_d = r_idx_d(:, idx_bench);
min_d = min(size(r_stocks_d, 1), size(r_mkt_d, 1));
data_d = [r_stocks_d(1:min_d, :), r_mkt_d(1:min_d)];
valid_d = ~any(isnan(data_d), 2);
r_s_d_c = data_d(valid_d, 1:end-1);
r_m_d_c = data_d(valid_d, end);

n = size(r_s_d_c, 2);
betas_d = zeros(n, 1);
v_m_d = var(r_m_d_c);
for i = 1:n
    c = cov(r_s_d_c(:, i), r_m_d_c);
    betas_d(i) = c(1, 2) / v_m_d;
end
beta_p_d = w_nys_10_d' * betas_d;

r_stocks_m = returns_nys_m(:, idx_best_nys);
r_mkt_m = r_idx_m(:, idx_bench);
min_m = min(size(r_stocks_m, 1), size(r_mkt_m, 1));
data_m = [r_stocks_m(1:min_m, :), r_mkt_m(1:min_m)];
valid_m = ~any(isnan(data_m), 2);
r_s_m_c = data_m(valid_m, 1:end-1);
r_m_m_c = data_m(valid_m, end);

betas_m = zeros(n, 1);
v_m_m = var(r_m_m_c);
for i = 1:n
    c = cov(r_s_m_c(:, i), r_m_m_c);
    betas_m(i) = c(1, 2) / v_m_m;
end
beta_p_m = w_nys_10_m' * betas_m;

T_beta_nys = table(sel_names_nys', round(betas_d, 4), round(betas_m, 4), ...
    'VariableNames', {'Security', 'Beta_Daily', 'Beta_Monthly'});
fprintf('\n==================================================\n');
fprintf('POINT 16: NYSE BETA ANALYSIS\n');
fprintf('==================================================\n');
disp(T_beta_nys);
fprintf('PORTFOLIO BETA (DAILY):   %.4f\n', beta_p_d);
fprintf('PORTFOLIO BETA (MONTHLY): %.4f\n', beta_p_m);
fprintf('==================================================\n');

figure('Name', 'P16: NYSE Daily Beta Analysis', 'Color', 'w', 'Position', [100, 100, 900, 450]);
b1 = bar(betas_d, 'FaceColor', [0.4940 0.1840 0.5560]);
hold on; grid on;
yline(1, 'k--', 'Market (1.0)', 'LineWidth', 1.2);
yline(beta_p_d, 'r', ['Portf Beta: ' num2str(beta_p_d, '%.3f')], 'LineWidth', 2);
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45);
ylabel('Beta Coefficient'); title('NYSE: Individual Betas (DAILY)');
ylim([0 max([betas_d; 1.2])*1.2]);
xtips1 = b1.XEndPoints; ytips1 = b1.YEndPoints;
text(xtips1, ytips1, string(round(betas_d, 2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

figure('Name', 'P16: NYSE Monthly Beta Analysis', 'Color', 'w', 'Position', [150, 150, 900, 450]);
b2 = bar(betas_m, 'FaceColor', [0.7170 0.2740 1.0000]);
hold on; grid on;
yline(1, 'k--', 'Market (1.0)', 'LineWidth', 1.2);
yline(beta_p_m, 'r', ['Portf Beta: ' num2str(beta_p_m, '%.3f')], 'LineWidth', 2);
xticks(1:n); xticklabels(sel_names_nys); xtickangle(45);
ylabel('Beta Coefficient'); title('NYSE: Individual Betas (MONTHLY)');
ylim([0 max([betas_m; 1.2])*1.2]);
xtips2 = b2.XEndPoints; ytips2 = b2.YEndPoints;
text(xtips2, ytips2, string(round(betas_m, 2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

T_beta_nys_daily = table(sel_names_nys', round(betas_d, 4), ...
    'VariableNames', {'Security', 'Beta_Daily'});

T_beta_nys_monthly = table(sel_names_nys', round(betas_m, 4), ...
    'VariableNames', {'Security', 'Beta_Monthly'});

fprintf('\n==================================================\n');
fprintf('POINT 16: NYSE BETA ANALYSIS - DAILY\n');
fprintf('==================================================\n');
disp(T_beta_nys_daily);
fprintf('PORTFOLIO BETA (DAILY): %.4f\n', beta_p_d);

fprintf('\n==================================================\n');
fprintf('POINT 16: NYSE BETA ANALYSIS - MONTHLY\n');
fprintf('==================================================\n');
disp(T_beta_nys_monthly);
fprintf('PORTFOLIO BETA (MONTHLY): %.4f\n', beta_p_m);

%% --- BETA CALCULATION - NASDAQ ---
if ~exist('w_nas_9_d', 'var') || ~exist('r_idx_d', 'var')
    error('Esegui prima P9 e P15.');
end

idx_bench = find(contains(names_idx, 'Nasdaq', 'IgnoreCase', true), 1);
if isempty(idx_bench), idx_bench = 2; end

r_stocks_d = returns_nas_d(:, idx_best_nas);
r_mkt_d = r_idx_d(:, idx_bench);
min_d = min(size(r_stocks_d, 1), size(r_mkt_d, 1));
data_d = [r_stocks_d(1:min_d, :), r_mkt_d(1:min_d)];
valid_d = ~any(isnan(data_d), 2);
r_s_d_c = data_d(valid_d, 1:end-1);
r_m_d_c = data_d(valid_d, end);

n = size(r_s_d_c, 2);
betas_d = zeros(n, 1);
v_m_d = var(r_m_d_c);
for i = 1:n
    c = cov(r_s_d_c(:, i), r_m_d_c);
    betas_d(i) = c(1, 2) / v_m_d;
end
beta_p_d = w_nas_9_d' * betas_d;

r_stocks_m = returns_nas_m(:, idx_best_nas);
r_mkt_m = r_idx_m(:, idx_bench);
min_m = min(size(r_stocks_m, 1), size(r_mkt_m, 1));
data_m = [r_stocks_m(1:min_m, :), r_mkt_m(1:min_m)];
valid_m = ~any(isnan(data_m), 2);
r_s_m_c = data_m(valid_m, 1:end-1);
r_m_m_c = data_m(valid_m, end);

betas_m = zeros(n, 1);
v_m_m = var(r_m_m_c);
for i = 1:n
    c = cov(r_s_m_c(:, i), r_m_m_c);
    betas_m(i) = c(1, 2) / v_m_m;
end
beta_p_m = w_nas_9_m' * betas_m;

T_beta_nas = table(sel_names_nas', round(betas_d, 4), round(betas_m, 4), ...
    'VariableNames', {'Security', 'Beta_Daily', 'Beta_Monthly'});
fprintf('\n==================================================\n');
fprintf('POINT 17: NASDAQ BETA ANALYSIS\n');
fprintf('==================================================\n');
disp(T_beta_nas);
fprintf('PORTFOLIO BETA (DAILY):   %.4f\n', beta_p_d);
fprintf('PORTFOLIO BETA (MONTHLY): %.4f\n', beta_p_m);
fprintf('==================================================\n');

figure('Name', 'P17: Nasdaq Daily Beta Analysis', 'Color', 'w', 'Position', [100, 100, 900, 450]);
b1 = bar(betas_d, 'FaceColor', [0.4660 0.6740 0.1880]);
hold on; grid on;
yline(1, 'k--', 'Market (1.0)', 'LineWidth', 1.2);
yline(beta_p_d, 'r', ['Portf Beta: ' num2str(beta_p_d, '%.3f')], 'LineWidth', 2);
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Beta Coefficient'); title('Nasdaq: Individual Betas (DAILY)');
ylim([0 max([betas_d; 1.2])*1.2]);
xtips1 = b1.XEndPoints; ytips1 = b1.YEndPoints;
text(xtips1, ytips1, string(round(betas_d, 2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

figure('Name', 'P17: Nasdaq Monthly Beta Analysis', 'Color', 'w', 'Position', [150, 150, 900, 450]);
b2 = bar(betas_m, 'FaceColor', [0.9290 0.6940 0.1250]);
hold on; grid on;
yline(1, 'k--', 'Market (1.0)', 'LineWidth', 1.2);
yline(beta_p_m, 'r', ['Portf Beta: ' num2str(beta_p_m, '%.3f')], 'LineWidth', 2);
xticks(1:n); xticklabels(sel_names_nas); xtickangle(45);
ylabel('Beta Coefficient'); title('Nasdaq: Individual Betas (MONTHLY)');
ylim([0 max([betas_m; 1.2])*1.2]);
xtips2 = b2.XEndPoints; ytips2 = b2.YEndPoints;
text(xtips2, ytips2, string(round(betas_m, 2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

T_beta_nas_daily = table(sel_names_nas', round(betas_d, 4), ...
    'VariableNames', {'Security', 'Beta_Daily'});

T_beta_nas_monthly = table(sel_names_nas', round(betas_m, 4), ...
    'VariableNames', {'Security', 'Beta_Monthly'});

fprintf('\n==================================================\n');
fprintf('POINT 17: NASDAQ BETA ANALYSIS - DAILY\n');
fprintf('==================================================\n');
disp(T_beta_nas_daily);
fprintf('PORTFOLIO BETA (DAILY): %.4f\n', beta_p_d);

fprintf('\n==================================================\n');
fprintf('POINT 17: NASDAQ BETA ANALYSIS - MONTHLY\n');
fprintf('==================================================\n');
disp(T_beta_nas_monthly);
fprintf('PORTFOLIO BETA (MONTHLY): %.4f\n', beta_p_m);

%% --- SML VERIFICATION - NYSE ---

Rf_annual = 2.0; 
Rf_daily = Rf_annual / 252;
Rf_monthly = Rf_annual / 12;

idx_check = [1, 2]; 
names_check = sel_names_nys(idx_check);

r_mkt_d = r_idx_d(:, 1); 
min_d = min(size(returns_nys_d(:, idx_best_nys), 1), size(r_mkt_d, 1));
data_d = [returns_nys_d(1:min_d, idx_best_nys), r_mkt_d(1:min_d)];
valid_d = ~any(isnan(data_d), 2);

r_asset_d = data_d(valid_d, 1:end-1);
r_mkt_d_clean = data_d(valid_d, end);

mu_assets_d = mean(r_asset_d)';
mu_mkt_d = mean(r_mkt_d_clean);
mu_port_d = mean(r_asset_d * w_nys_10_d);

betas_d = zeros(12, 1);
var_mkt_d = var(r_mkt_d_clean);
for i = 1:12
    c = cov(r_asset_d(:, i), r_mkt_d_clean);
    betas_d(i) = c(1, 2) / var_mkt_d;
end
beta_port_d = w_nys_10_d' * betas_d;

r_mkt_m = r_idx_m(:, 1); 
min_m = min(size(returns_nys_m(:, idx_best_nys), 1), size(r_mkt_m, 1));
data_m = [returns_nys_m(1:min_m, idx_best_nys), r_mkt_m(1:min_m)];
valid_m = ~any(isnan(data_m), 2);

r_asset_m = data_m(valid_m, 1:end-1);
r_mkt_m_clean = data_m(valid_m, end);

mu_assets_m = mean(r_asset_m)';
mu_mkt_m = mean(r_mkt_m_clean);
mu_port_m = mean(r_asset_m * w_nys_10_m);

betas_m = zeros(12, 1);
var_mkt_m = var(r_mkt_m_clean);
for i = 1:12
    c = cov(r_asset_m(:, i), r_mkt_m_clean);
    betas_m(i) = c(1, 2) / var_mkt_m;
end
beta_port_m = w_nys_10_m' * betas_m;

fprintf('\n===========================================================\n');
fprintf('POINT 18: SML VERIFICATION (NYSE - 2 Chosen Securities)\n');
fprintf('===========================================================\n');

print_verif = @(freq, name, beta, mu_real, rf, mu_mkt) ...
    fprintf('%s | %-12s | Beta: %.3f | Real: %.4f%% | CAPM: %.4f%% | Alpha: %.4f%%\n', ...
    freq, name, beta, mu_real, (rf + beta*(mu_mkt - rf)), mu_real - (rf + beta*(mu_mkt - rf)));

fprintf('--- DAILY FREQUENCY ---\n');
for k = idx_check
    print_verif('Daily  ', names_check{k==idx_check}, betas_d(k), mu_assets_d(k), Rf_daily, mu_mkt_d);
end
print_verif('Daily  ', 'PORTFOLIO', beta_port_d, mu_port_d, Rf_daily, mu_mkt_d);

fprintf('\n--- MONTHLY FREQUENCY ---\n');
for k = idx_check
    print_verif('Monthly', names_check{k==idx_check}, betas_m(k), mu_assets_m(k), Rf_monthly, mu_mkt_m);
end
print_verif('Monthly', 'PORTFOLIO', beta_port_m, mu_port_m, Rf_monthly, mu_mkt_m);

figure('Name', 'P18: NYSE SML', 'Color', 'w', 'Position', [100, 100, 900, 800]);

subplot(2,1,1);
x_grid = linspace(min([betas_d;0]), max([betas_d;1])*1.1, 10);
plot(x_grid, Rf_daily + x_grid * (mu_mkt_d - Rf_daily), 'k-', 'LineWidth', 2); hold on;
scatter(betas_d, mu_assets_d, 50, [0.5 0.5 0.5], 'filled'); 
scatter(betas_d(idx_check), mu_assets_d(idx_check), 100, 'b', 'filled');
scatter(beta_port_d, mu_port_d, 150, 'r', 'p', 'filled');
text(betas_d(idx_check), mu_assets_d(idx_check), names_check, 'Color', 'b', 'VerticalAlignment', 'bottom');
title('NYSE SML - Daily'); xlabel('Beta'); ylabel('Avg Return (%)');
legend('Theoretical SML', 'Assets', 'Chosen 2', 'Portfolio', 'Location', 'best'); grid on;

subplot(2,1,2);
x_grid_m = linspace(min([betas_m;0]), max([betas_m;1])*1.1, 10);
plot(x_grid_m, Rf_monthly + x_grid_m * (mu_mkt_m - Rf_monthly), 'k-', 'LineWidth', 2); hold on;
scatter(betas_m, mu_assets_m, 50, [0.5 0.5 0.5], 'filled');
scatter(betas_m(idx_check), mu_assets_m(idx_check), 100, 'b', 'filled');
scatter(beta_port_m, mu_port_m, 150, 'r', 'p', 'filled');
text(betas_m(idx_check), mu_assets_m(idx_check), names_check, 'Color', 'b', 'VerticalAlignment', 'bottom');
title('NYSE SML - Monthly'); xlabel('Beta'); ylabel('Avg Return (%)'); grid on;

%% --- SML VERIFICATION - NASDAQ ---
if ~exist('betas_d', 'var') || ~exist('beta_p_d', 'var')
    error('Errore: Esegui prima il Punto 17 per calcolare i Beta.');
end

Rf_annual = 2.0; 
Rf_daily = Rf_annual / 252;
Rf_monthly = Rf_annual / 12;
idx_check = [1, 2]; 
names_check_nas = sel_names_nas(idx_check);

idx_bench = find(contains(names_idx, 'Nasdaq', 'IgnoreCase', true), 1);
if isempty(idx_bench), idx_bench = 2; end
r_mkt_d_full = r_idx_d(:, idx_bench);
r_mkt_m_full = r_idx_m(:, idx_bench);

min_d = min(size(returns_nas_d(:, idx_best_nas), 1), size(r_mkt_d_full, 1));
data_d = [returns_nas_d(1:min_d, idx_best_nas), r_mkt_d_full(1:min_d)];
valid_d = ~any(isnan(data_d), 2);
r_asset_d = data_d(valid_d, 1:end-1);
r_mkt_d_clean = data_d(valid_d, end);

mu_assets_d = mean(r_asset_d)';
mu_mkt_d = mean(r_mkt_d_clean);
mu_port_d = mean(r_asset_d * w_nas_9_d);

min_m = min(size(returns_nas_m(:, idx_best_nas), 1), size(r_mkt_m_full, 1));
data_m = [returns_nas_m(1:min_m, idx_best_nas), r_mkt_m_full(1:min_m)];
valid_m = ~any(isnan(data_m), 2);
r_asset_m = data_m(valid_m, 1:end-1);
r_mkt_m_clean = data_m(valid_m, end);

mu_assets_m = mean(r_asset_m)';
mu_mkt_m = mean(r_mkt_m_clean);
mu_port_m = mean(r_asset_m * w_nas_9_m);

fprintf('\n===========================================================\n');
fprintf('POINT 19: SML VERIFICATION (NASDAQ)\n');
fprintf('===========================================================\n');

print_verif = @(freq, name, beta, mu_real, rf, mu_mkt) ...
    fprintf('%s | %-12s | Beta: %.3f | Real: %.4f%% | CAPM: %.4f%% | Alpha: %.4f%%\n', ...
    freq, name, beta, mu_real, (rf + beta*(mu_mkt - rf)), mu_real - (rf + beta*(mu_mkt - rf)));

fprintf('--- DAILY FREQUENCY ---\n');
for k = idx_check
    print_verif('Daily  ', names_check_nas{k==idx_check}, betas_d(k), mu_assets_d(k), Rf_daily, mu_mkt_d);
end
print_verif('Daily  ', 'PORTFOLIO', beta_p_d, mu_port_d, Rf_daily, mu_mkt_d);

fprintf('\n--- MONTHLY FREQUENCY ---\n');
for k = idx_check
    print_verif('Monthly', names_check_nas{k==idx_check}, betas_m(k), mu_assets_m(k), Rf_monthly, mu_mkt_m);
end
print_verif('Monthly', 'PORTFOLIO', beta_p_m, mu_port_m, Rf_monthly, mu_mkt_m);

figure('Name', 'P19: Nasdaq SML Verification', 'Color', 'w', 'Position', [150, 150, 900, 800]);

subplot(2,1,1);
x_grid = linspace(min([betas_d;0]), max([betas_d;1])*1.1, 10);
plot(x_grid, Rf_daily + x_grid * (mu_mkt_d - Rf_daily), 'k-', 'LineWidth', 2); hold on;
scatter(betas_d, mu_assets_d, 60, [0.46 0.67 0.18], 'filled');
scatter(betas_d(idx_check), mu_assets_d(idx_check), 100, 'b', 'filled');
scatter(beta_p_d, mu_port_d, 180, 'r', 'p', 'filled');
text(betas_d(idx_check), mu_assets_d(idx_check), names_check_nas, 'Color', 'b', 'VerticalAlignment', 'bottom', 'FontSize', 9);
title('Nasdaq Security Market Line (Daily)'); xlabel('Beta'); ylabel('Avg Return (%)');
legend('Theoretical SML', 'Nasdaq Stocks', 'Check Assets', 'Portfolio', 'Location', 'best'); grid on;

subplot(2,1,2);
x_grid_m = linspace(min([betas_m;0]), max([betas_m;1])*1.1, 10);
plot(x_grid_m, Rf_monthly + x_grid_m * (mu_mkt_m - Rf_monthly), 'k-', 'LineWidth', 2); hold on;
scatter(betas_m, mu_assets_m, 60, [0.92 0.69 0.12], 'filled');
scatter(betas_m(idx_check), mu_assets_m(idx_check), 100, 'b', 'filled');
scatter(beta_p_m, mu_port_m, 180, 'r', 'p', 'filled');
text(betas_m(idx_check), mu_assets_m(idx_check), names_check_nas, 'Color', 'b', 'VerticalAlignment', 'bottom', 'FontSize', 9);
title('Nasdaq Security Market Line (Monthly)'); xlabel('Beta'); ylabel('Avg Return (%)'); grid on;

%% --- BLACK-LITTERMAN MODEL - NYSE ---

Rf_annual = 0.02; 
tau = 0.05;       
n = length(idx_best_nys); 
freqs = {'Daily', 'Monthly'};
factors = [252, 12];
options_opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nys_d','var') || ~exist('sel_names_nys','var')
    error('Errore: Esegui prima il caricamento dati (Master Setup/P5).');
end

for k = 1:2
    label = freqs{k};
    factor = factors(k);
    
    fprintf('\n==================================================\n');
    fprintf('ANALISI BLACK-LITTERMAN: NYSE %s\n', upper(label));
    fprintf('==================================================\n');

    if k == 1
        r_sel = returns_nys_d(:, idx_best_nys);
        r_m = r_idx_d(:, 1);
    else
        r_sel = returns_nys_m(:, idx_best_nys);
        r_m = r_idx_m(:, 1); 
    end
 
    min_len = min(size(r_sel, 1), size(r_m, 1));
    data_sync = [r_sel(1:min_len, :), r_m(1:min_len)];
    valid = ~any(isnan(data_sync), 2);
    r_sel_c = data_sync(valid, 1:n);
    r_m_c = data_sync(valid, n+1);

    Sigma = cov(r_sel_c);
    Rf_period = Rf_annual / factor;
    
    delta = (mean(r_m_c) - Rf_period) / var(r_m_c);
    if delta < 0, delta = 2.5; end 
    
    w_eq = ones(n, 1) / n; 
    
    Pi = delta * Sigma * w_eq;
   
    P = zeros(4, n); 
    Q = zeros(4, 1);
    
    P(1, 1) = 1; 
    Q(1) = Pi(1) + 0.10/factor; 
    
    P(2, 2) = 1; 
    Q(2) = Pi(2) - 0.05/factor;
    
    P(3, 3) = 1; P(3, 4) = -1; 
    Q(3) = 0.02/factor;

    P(4, 5) = 1; P(4, 6) = -1; 
    Q(4) = 0.03/factor;

    Omega = diag(diag(P * (tau * Sigma) * P'));

    M_inv = inv(tau * Sigma);

    Sigma_BL = inv(M_inv + P' * inv(Omega) * P);

    mu_BL = Sigma_BL * (M_inv * Pi + P' * inv(Omega) * Q);

    Sigma_post = Sigma + Sigma_BL;
    
    Aeq = ones(1,n); beq = 1; 
    lb = zeros(n,1); ub = []; 
    H_mv = delta * Sigma;
    f_mv = -mean(r_sel_c)'; 
    w_mv = quadprog(H_mv, f_mv, [], [], Aeq, beq, lb, ub, [], options_opt);
    
    H_bl = delta * Sigma_post;
    f_bl = -mu_BL;
    w_bl = quadprog(H_bl, f_bl, [], [], Aeq, beq, lb, ub, [], options_opt);

    calc_stats = @(r, w) [mean(r*w); std(r*w); var(r*w); skewness(r*w); kurtosis(r*w); (mean(r*w)-Rf_period)/std(r*w)];
    
    stats_mv = calc_stats(r_sel_c, w_mv);
    stats_bl = calc_stats(r_sel_c, w_bl);
    
    T_stats = table(stats_mv, stats_bl, ...
        'RowNames', {'Mean Return', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'}, ...
        'VariableNames', {'Standard_MV_Hist', 'Black_Litterman'});
    
    disp('Confronto Statistiche Portafoglio:');
    disp(T_stats);
    
    figure('Name', ['P20: BL Allocation ' label]);
    b = bar([w_mv, w_bl]);
    title(['Asset Allocation (' label '): Historical MV vs Black-Litterman']);
    legend('Standard MV (Historical)', 'Black-Litterman (Views)');
    xticks(1:n); xticklabels(sel_names_nys); xtickangle(45); 
    ylabel('Weight'); grid on;
    ylim([0 0.6]); 
end

%% --- BLACK-LITTERMAN MODEL - NASDAQ ---
Rf_annual = 0.02; 
tau = 0.05;       
n = length(idx_best_nas);
freqs = {'Daily', 'Monthly'};
factors = [252, 12];
options_opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nas_d','var') || ~exist('sel_names_nas','var')
    error('Errore: Esegui prima il caricamento dati (Master Setup).');
end

idx_bench = find(contains(names_idx, 'Nasdaq', 'IgnoreCase', true), 1);
if isempty(idx_bench), idx_bench = 2; end
name_bench = names_idx{idx_bench};

for k = 1:2
    label = freqs{k};
    factor = factors(k);
    
    fprintf('\n==================================================\n');
    fprintf('ANALISI BLACK-LITTERMAN: NASDAQ %s (Ref: %s)\n', upper(label), name_bench);
    fprintf('==================================================\n');

    if k == 1
        r_sel = returns_nas_d(:, idx_best_nas);
        r_m = r_idx_d(:, idx_bench); 
    else
        r_sel = returns_nas_m(:, idx_best_nas);
        r_m = r_idx_m(:, idx_bench); 
    end
    
    min_len = min(size(r_sel, 1), size(r_m, 1));
    data_sync = [r_sel(1:min_len, :), r_m(1:min_len)];
    valid = ~any(isnan(data_sync), 2);
    r_sel_c = data_sync(valid, 1:n);
    r_m_c = data_sync(valid, n+1);

    Sigma = cov(r_sel_c);
    Rf_period = Rf_annual / factor;
    
    delta = (mean(r_m_c) - Rf_period) / var(r_m_c);
    if delta < 0, delta = 2.5; end 
 
    w_eq = ones(n, 1) / n; 
    Pi = delta * Sigma * w_eq;

    P = zeros(4, n); 
    Q = zeros(4, 1);

    P(1, 1) = 1; 
    Q(1) = Pi(1) + 0.10/factor; 

    P(2, 2) = 1; 
    Q(2) = Pi(2) - 0.05/factor;
    
    P(3, 3) = 1; P(3, 4) = -1; 
    Q(3) = 0.02/factor;
   
    P(4, 5) = 1; P(4, 6) = -1; 
    Q(4) = 0.03/factor;
    

    Omega = diag(diag(P * (tau * Sigma) * P'));

    M_inv = inv(tau * Sigma);
    Sigma_BL = inv(M_inv + P' * inv(Omega) * P);
    mu_BL = Sigma_BL * (M_inv * Pi + P' * inv(Omega) * Q);
    Sigma_post = Sigma + Sigma_BL;

    Aeq = ones(1,n); beq = 1; 
    lb = zeros(n,1); ub = []; 

    H_mv = delta * Sigma;
    f_mv = -mean(r_sel_c)';
    w_mv = quadprog(H_mv, f_mv, [], [], Aeq, beq, lb, ub, [], options_opt);

    H_bl = delta * Sigma_post;
    f_bl = -mu_BL;
    w_bl = quadprog(H_bl, f_bl, [], [], Aeq, beq, lb, ub, [], options_opt);

    calc_stats = @(r, w) [mean(r*w); std(r*w); var(r*w); skewness(r*w); kurtosis(r*w); (mean(r*w)-Rf_period)/std(r*w)];
    
    stats_mv = calc_stats(r_sel_c, w_mv);
    stats_bl = calc_stats(r_sel_c, w_bl);
    
    T_stats = table(stats_mv, stats_bl, ...
        'RowNames', {'Mean Return', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'}, ...
        'VariableNames', {'Standard_MV_Hist', 'Black_Litterman'});
    
    disp('Confronto Statistiche Portafoglio (Nasdaq):');
    disp(T_stats);

    figure('Name', ['P21: BL Allocation Nasdaq ' label]);
    bar([w_mv, w_bl]);
    title(['Nasdaq Allocation (' label '): Historical MV vs Black-Litterman']);
    legend('Standard MV (Historical)', 'Black-Litterman (Views)');
    xticks(1:n); xticklabels(sel_names_nas); xtickangle(45); 
    ylabel('Weight'); grid on;
    ylim([0 0.6]);
end

%% --- STANDARD BAYESIAN ASSET ALLOCATION - NYSE ---
Rf_annual = 0.02;     
n = 12;
freqs = {'Daily', 'Monthly'};
factors = [252, 12];
opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nys_d','var') || ~exist('idx_best_nys','var')
    error('Esegui prima il Master Setup.');
end

for k = 1:2
    label = freqs{k};
    factor = factors(k);
    
    fprintf('\n==================================================\n');
    fprintf('P22: BAYESIAN ANALYSIS - NYSE %s\n', upper(label));
    fprintf('==================================================\n');

    if k == 1
        r_sel = returns_nys_d(:, idx_best_nys);
    else
        r_sel = returns_nys_m(:, idx_best_nys);
    end

    r_sel(any(isnan(r_sel), 2), :) = [];
    [T_obs, ~] = size(r_sel); 

    mu_sample = mean(r_sel)';
    Sigma_sample = cov(r_sel);
    std_sample = std(r_sel)';

    mu_0 = mu_sample + 1 * std_sample; 

    Sigma_0 = 2 * Sigma_sample; 

    H_0 = inv(Sigma_0);          
    H_sample = T_obs * inv(Sigma_sample); 

    Sigma_pos_mean = inv(H_0 + H_sample);

    mu_pos = Sigma_pos_mean * (H_0 * mu_0 + H_sample * mu_sample);

    Sigma_pred = Sigma_sample + Sigma_pos_mean;

    T_bayes_params = table(sel_names_nys', round(mu_sample*100,2), round(mu_pos*100,2), ...
        round(std_sample*100,2), round(sqrt(diag(Sigma_pred))*100,2), ...
        'VariableNames', {'Asset', 'Sample_Mean_%', 'Bayesian_Mean_%', 'Sample_Std_%', 'Bayesian_Std_%'});
    disp('--- Bayesian Model Parameters (Predictive) ---');
    disp(T_bayes_params);

    lb = zeros(n,1);
    ub = [];
    Aeq = ones(1,n); beq = 1;

    target_mv = mean(mu_sample); 
    Amv = -mu_sample'; bmv = -target_mv;
    w_mv = quadprog(Sigma_sample, zeros(n,1), Amv, bmv, Aeq, beq, lb, ub, [], opt);
    if isempty(w_mv), w_mv = quadprog(Sigma_sample, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt); end

    target_bayes = mean(mu_pos); 
    Abayes = -mu_pos'; bbayes = -target_bayes;
    w_bayes = quadprog(Sigma_pred, zeros(n,1), Abayes, bbayes, Aeq, beq, lb, ub, [], opt);
    if isempty(w_bayes), w_bayes = quadprog(Sigma_pred, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt); end

    Rf_period = Rf_annual / factor;
    calc_stats = @(r, w) [mean(r*w); std(r*w); var(r*w); skewness(r*w); kurtosis(r*w); (mean(r*w)-Rf_period)/std(r*w)];
    
    stats_mv = calc_stats(r_sel, w_mv);
    stats_bayes = calc_stats(r_sel, w_bayes);
    
    T_stats = table(stats_mv, stats_bayes, ...
        'RowNames', {'Mean Return', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'}, ...
        'VariableNames', {'Standard_MV', 'Bayesian_Allocation'});
    
    disp('--- Portfolio Comparison ---');
    disp(T_stats);
    
    figure('Color', 'w', 'Name', ['P22: Bayesian NYSE - ' label], 'Position', [100 100 1100 500]);

    subplot(1, 2, 1);
    b = bar([w_mv, w_bayes]);
    b(1).FaceColor = [0.2 0.4 0.8]; 
    b(2).FaceColor = [0.2 0.7 0.4]; 
    legend('Standard MV', 'Bayesian');
    title(['Asset Allocation (' label ') - Long Only']);
    xticklabels(sel_names_nys); xtickangle(45); grid on; ylabel('Weight'); 
    ylim([0 0.3]);

    subplot(1, 2, 2);
    hold on;

    plot(1:n, mu_sample, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Sample Mean (History)');

    plot(1:n, mu_0, 'g--d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Prior Mean (Optimistic)');

    plot(1:n, mu_pos, 'r-x', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Posterior Mean (Result)');

    legend('Location', 'best');
    title(['Shrinkage Effect on Returns (' label ')']);
    ylabel('Expected Return');
    xticks(1:n); xticklabels(sel_names_nys); xtickangle(45); 
    grid on;

    if k == 1
       subtitle('Daily: Red line overlaps Blue line (Data dominates)');
    else
       subtitle('Monthly: Red line moves towards Green line (Prior pulls)');
    end
end

Rf_annual = 0.02;     
n = length(idx_best_nas); 
freqs = {'Daily', 'Monthly'};
factors = [252, 12];
opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nas_d','var') || ~exist('idx_best_nas','var')
    error('Errore: Dati Nasdaq mancanti.');
end

for k = 1:2
    label = freqs{k};
    factor = factors(k);
    
    if k == 1
        r_sel = returns_nas_d(:, idx_best_nas);
    else
        r_sel = returns_nas_m(:, idx_best_nas);
    end
    
    r_sel(any(isnan(r_sel), 2), :) = [];
    [T_obs, ~] = size(r_sel); 
    
    mu_sample = mean(r_sel)';
    Sigma_sample = cov(r_sel);
    std_sample = std(r_sel)';
    
    mu_0 = mu_sample + 1 * std_sample; 
    Sigma_0 = 2 * Sigma_sample; 
    
    H_0 = inv(Sigma_0);                    
    H_sample = T_obs * inv(Sigma_sample);  
    
    Sigma_pos_mean = inv(H_0 + H_sample);
    mu_pos = Sigma_pos_mean * (H_0 * mu_0 + H_sample * mu_sample);
    Sigma_pred = Sigma_sample + Sigma_pos_mean;
    
    T_bayes_params = table(sel_names_nas', round(mu_sample*100,2), round(mu_pos*100,2), ...
        round(std_sample*100,2), round(sqrt(diag(Sigma_pred))*100,2), ...
        'VariableNames', {'Asset', 'Sample_Mean_pct', 'Bayesian_Mean_pct', 'Sample_Std_pct', 'Bayesian_Std_pct'});
    disp(T_bayes_params);
    
    lb = zeros(n,1);
    ub = []; 
    Aeq = ones(1,n); beq = 1;
    
    target_mv = mean(mu_sample); 
    w_mv = quadprog(Sigma_sample, zeros(n,1), -mu_sample', -target_mv, Aeq, beq, lb, ub, [], opt);
    if isempty(w_mv), w_mv = quadprog(Sigma_sample, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt); end
    
    target_bayes = mean(mu_pos); 
    w_bayes = quadprog(Sigma_pred, zeros(n,1), -mu_pos', -target_bayes, Aeq, beq, lb, ub, [], opt);
    if isempty(w_bayes), w_bayes = quadprog(Sigma_pred, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt); end
    
    Rf_period = Rf_annual / factor;
    calc_stats = @(r, w) [mean(r*w); std(r*w); var(r*w); skewness(r*w); kurtosis(r*w); (mean(r*w)-Rf_period)/std(r*w)];
    
    T_stats = table(calc_stats(r_sel, w_mv), calc_stats(r_sel, w_bayes), ...
        'RowNames', {'Mean', 'Std_Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe_Ratio'}, ...
        'VariableNames', {'Standard_MV', 'Bayesian'});
    disp(T_stats);
    
    figure('Color', 'w', 'Name', ['P22: Bayesian NASDAQ - ' label], 'Position', [100 100 1100 450]);
    
    subplot(1, 2, 1);
    b = bar([w_mv, w_bayes]);
    b(1).FaceColor = [0.4660 0.6740 0.1880]; 
    b(2).FaceColor = [0.9290 0.6940 0.1250]; 
    legend('Standard MV', 'Bayesian');
    title(['Allocation - ' label]);
    xticklabels(sel_names_nas); xtickangle(45); grid on; ylabel('Weight'); 
    top_y = min(1, max([w_mv; w_bayes]) * 1.15);
    ylim([0 top_y]);
    
    subplot(1, 2, 2);
    hold on;
    plot(1:n, mu_sample, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Sample');
    plot(1:n, mu_0, 'g--d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Prior');
    plot(1:n, mu_pos, 'r-x', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Posterior');
    legend('Location', 'best');
    title(['Shrinkage Effect - ' label]);
    ylabel('Return');
    xticks(1:n); xticklabels(sel_names_nas); xtickangle(45); 
    grid on;
end

%% --- GMV STATISTICS (NYSE) ---
Rf_annual = 2.0;     
Rf_daily = Rf_annual / 252;
Rf_monthly = Rf_annual / 12;
n = length(idx_best_nys);
opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nys_d', 'var') || ~exist('idx_best_nys', 'var')
    error('Variabili mancanti. Esegui il Master Setup.');
end

r_d = returns_nys_d(:, idx_best_nys);
r_d(any(isnan(r_d), 2), :) = [];
Sigma_d = cov(r_d);

r_m = returns_nys_m(:, idx_best_nys);
r_m(any(isnan(r_m), 2), :) = [];
Sigma_m = cov(r_m);

lb = zeros(n, 1);
ub = [];          
Aeq = ones(1, n); 
beq = 1;
w_gmv_d = quadprog(Sigma_d, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
w_gmv_m = quadprog(Sigma_m, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);

calc_stats = @(r_mat, w, rf) deal(...
    mean(r_mat * w), ...           
    std(r_mat * w), ...              
    var(r_mat * w), ...               
    skewness(r_mat * w), ...          
    kurtosis(r_mat * w), ...          
    (mean(r_mat * w) - rf) / std(r_mat * w) ... 
);

[mu_d, std_d, var_d, sk_d, ku_d, sh_d] = calc_stats(r_d, w_gmv_d, Rf_daily);
col_d = [mu_d; std_d; var_d; sk_d; ku_d; sh_d];

[mu_m, std_m, var_m, sk_m, ku_m, sh_m] = calc_stats(r_m, w_gmv_m, Rf_monthly);
col_m = [mu_m; std_m; var_m; sk_m; ku_m; sh_m];

RowNames = {'Mean_Return', 'Std_Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe_Ratio'};
T_gmv = table(col_d, col_m, ...
    'VariableNames', {'NYSE_Daily_GMV', 'NYSE_Monthly_GMV'}, ...
    'RowNames', RowNames);

fprintf('\n==================================================\n');
fprintf('POINT 23: GLOBAL MINIMUM VARIANCE STATISTICS (NYSE)\n');
fprintf('==================================================\n');
disp(T_gmv);

figure('Name', 'P23: NYSE GMV Weights', 'Color', 'w', 'Position', [100, 100, 800, 600]);

subplot(2,1,1);
bar(w_gmv_d, 'FaceColor', [0.2 0.4 0.6]); 
title('NYSE Daily GMV Weights (Long-Only, No Cap)');
ylabel('Weight'); grid on; ylim([0 0.3]);
set(gca, 'XTick', 1:n, 'XTickLabel', sel_names_nys, 'XTickLabelRotation', 45, 'FontSize', 8, 'TickLabelInterpreter', 'none');

subplot(2,1,2);
bar(w_gmv_m, 'FaceColor', [0.6 0.2 0.2]); 
title('NYSE Monthly GMV Weights (Long-Only, No Cap)');
ylabel('Weight'); grid on; ylim([0 0.3]);
set(gca, 'XTick', 1:n, 'XTickLabel', sel_names_nys, 'XTickLabelRotation', 45, 'FontSize', 8, 'TickLabelInterpreter', 'none');

%% --- GMV STATISTICS (NASDAQ) ---

Rf_annual = 2.0; 
Rf_daily = Rf_annual / 252;
Rf_monthly = Rf_annual / 12;
n = length(idx_best_nas);
opt = optimoptions('quadprog', 'Display', 'off');

if ~exist('returns_nas_d', 'var') || ~exist('idx_best_nas', 'var')
    error('Variabili Nasdaq mancanti. Esegui il Master Setup.');
end

r_d = returns_nas_d(:, idx_best_nas);
r_d(any(isnan(r_d), 2), :) = []; 
Sigma_d = cov(r_d);

r_m = returns_nas_m(:, idx_best_nas);
r_m(any(isnan(r_m), 2), :) = [];
Sigma_m = cov(r_m);

lb = zeros(n, 1);
ub = [];   
Aeq = ones(1, n); 
beq = 1;

w_gmv_d = quadprog(Sigma_d, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);
w_gmv_m = quadprog(Sigma_m, zeros(n,1), [], [], Aeq, beq, lb, ub, [], opt);

calc_stats = @(r_mat, w, rf) deal(...
    mean(r_mat * w), ...             
    std(r_mat * w), ...              
    var(r_mat * w), ...               
    skewness(r_mat * w), ...         
    kurtosis(r_mat * w), ...         
    (mean(r_mat * w) - rf) / std(r_mat * w) ... 
);

[mu_d, std_d, var_d, sk_d, ku_d, sh_d] = calc_stats(r_d, w_gmv_d, Rf_daily);
col_d = [mu_d; std_d; var_d; sk_d; ku_d; sh_d];

[mu_m, std_m, var_m, sk_m, ku_m, sh_m] = calc_stats(r_m, w_gmv_m, Rf_monthly);
col_m = [mu_m; std_m; var_m; sk_m; ku_m; sh_m];

RowNames = {'Mean_Return', 'Std_Dev', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe_Ratio'};
T_gmv_nas = table(col_d, col_m, ...
    'VariableNames', {'Nasdaq_Daily_GMV', 'Nasdaq_Monthly_GMV'}, ...
    'RowNames', RowNames);

fprintf('\n==================================================\n');
fprintf('POINT 24: GLOBAL MINIMUM VARIANCE STATISTICS (NASDAQ)\n');
fprintf('==================================================\n');
disp(T_gmv_nas);
figure('Name', 'P24: Nasdaq GMV Weights', 'Color', 'w', 'Position', [150, 150, 800, 600]);

subplot(2,1,1);
bar(w_gmv_d, 'FaceColor', [0.4660 0.6740 0.1880]); 
title('Nasdaq Daily GMV Weights (Long-Only, No Cap)');
ylabel('Weight'); grid on; ylim([0 0.5]);
set(gca, 'XTick', 1:n, 'XTickLabel', sel_names_nas, 'XTickLabelRotation', 45, 'FontSize', 8, 'TickLabelInterpreter', 'none');

subplot(2,1,2);
bar(w_gmv_m, 'FaceColor', [0.9290 0.6940 0.1250]); 
title('Nasdaq Monthly GMV Weights (Long-Only, No Cap)');
ylabel('Weight'); grid on; ylim([0 0.5]);
set(gca, 'XTick', 1:n, 'XTickLabel', sel_names_nas, 'XTickLabelRotation', 45, 'FontSize', 8, 'TickLabelInterpreter', 'none');

%% --- MODEL AVERAGING & FINAL COMPARISON (Monthly) ---
Rf_annual = 2.0;     
Rf_monthly = 2.0 / 12; 
n = 12;
tau = 0.05; 
opt = optimoptions('quadprog', 'Display', 'off');
markets = {'NYSE', 'NASDAQ'};

if ~exist('returns_nys_m', 'var') || ~exist('returns_nas_m', 'var')
    error('Esegui prima il Master Setup.');
end

for k = 1:2
    curr_mkt = markets{k};
    fprintf('\n======================================================\n');
    fprintf('FINAL SYNTHESIS: %s PORTFOLIO COMBINATION\n', curr_mkt);
    fprintf('======================================================\n');

    if k == 1 
        r_sel = returns_nys_m(:, idx_best_nys);
        r_bench = r_idx_m(:, 1); 
        names_curr = sel_names_nys;
    else 
        r_sel = returns_nas_m(:, idx_best_nas);
        idx_bench = find(contains(names_idx, 'Nasdaq', 'IgnoreCase', true), 1);
        if isempty(idx_bench), idx_bench = 2; end
        r_bench = r_idx_m(:, idx_bench);
        names_curr = sel_names_nas;
    end
    
    min_len = min(size(r_sel,1), size(r_bench,1));
    data_sync = [r_sel(1:min_len, :), r_bench(1:min_len)];
    valid = ~any(isnan(data_sync), 2);
    r_sel = data_sync(valid, 1:n);
    r_bench = data_sync(valid, n+1);
    
    mu = mean(r_sel)'; 
    Sigma = cov(r_sel); 
    std_vec = std(r_sel)';
    T_obs = size(r_sel, 1);

    lb = zeros(n,1); 
    ub_vec = []; 
    Aeq = ones(1,n); beq = 1;

    tgt_mv = mean(mu);
    w_mv = quadprog(Sigma, zeros(n,1), -mu', -tgt_mv, Aeq, beq, lb, ub_vec, [], opt);
    if isempty(w_mv), w_mv = quadprog(Sigma, zeros(n,1), [], [], Aeq, beq, lb, ub_vec, [], opt); end
    

    w_gmv = quadprog(Sigma, zeros(n,1), [], [], Aeq, beq, lb, ub_vec, [], opt);

    w_eq = ones(n,1)/n;
    mu_mkt = mean(r_bench); var_mkt = var(r_bench);
    delta = (mu_mkt - Rf_monthly)/var_mkt; if delta<0, delta=2.5; end
    Pi = delta * Sigma * w_eq;

    P_mat = zeros(4,n); Q_vec = zeros(4,1);
    P_mat(1,1)=1; Q_vec(1)=Pi(1)+0.5*std_vec(1); 
    P_mat(2,2)=1; Q_vec(2)=Pi(2)-0.5*std_vec(2);
    if k==1, spread=2/12; else, spread=3/12; end
    P_mat(3,3)=1; P_mat(3,4)=-1; Q_vec(3)=spread;
    P_mat(4,5)=1; P_mat(4,6)=-1; Q_vec(4)=spread; 
    
    Omega = diag(diag(P_mat*(tau*Sigma)*P_mat'));
    M_inv = inv(tau*Sigma);
    Sigma_BL = inv(M_inv + P_mat'*inv(Omega)*P_mat);
    mu_BL = Sigma_BL*(M_inv*Pi + P_mat'*inv(Omega)*Q_vec);
    Sigma_Tot_BL = Sigma + Sigma_BL;
    
    tgt_bl = mean(mu_BL);
    w_bl = quadprog(Sigma_Tot_BL, zeros(n,1), -mu_BL', -tgt_bl, Aeq, beq, lb, ub_vec, [], opt);
    if isempty(w_bl), w_bl = quadprog(Sigma_Tot_BL, zeros(n,1), [], [], Aeq, beq, lb, ub_vec, [], opt); end

    mu_0 = mu + 1*std_vec; 
    Sigma_0 = 2*Sigma;     
    H_0 = inv(Sigma_0); H_s = T_obs*inv(Sigma);
    Sigma_pos = inv(H_0 + H_s);
    mu_pos = Sigma_pos*(H_0*mu_0 + H_s*mu);
    Sigma_pred = Sigma + Sigma_pos;

    tgt_bayes = mean(mu_pos);
    w_bayes = quadprog(Sigma_pred, zeros(n,1), -mu_pos', -tgt_bayes, Aeq, beq, lb, ub_vec, [], opt);
    if isempty(w_bayes), w_bayes = quadprog(Sigma_pred, zeros(n,1), [], [], Aeq, beq, lb, ub_vec, [], opt); end

    w_combo = 0.25*w_mv + 0.25*w_gmv + 0.25*w_bl + 0.25*w_bayes;

    calc_m = @(w) [mean(r_sel*w); std(r_sel*w); skewness(r_sel*w); kurtosis(r_sel*w); (mean(r_sel*w)-Rf_monthly)/std(r_sel*w)];
    res_mat = [calc_m(w_mv), calc_m(w_gmv), calc_m(w_bl), calc_m(w_bayes), calc_m(w_combo)];
    
    T_final = array2table(res_mat, 'VariableNames', {'Mean_Var','GMV','Black_Lit','Bayesian','COMBINED'}, ...
        'RowNames', {'Mean_Ret','Std_Dev','Skewness','Kurtosis','Sharpe'});
    disp(T_final);

    figure('Color','w', 'Name', ['Final Synthesis: ' curr_mkt], 'Position', [100, 100, 1000, 600]);

    subplot(2,1,1);
    bar([w_mv, w_gmv, w_bl, w_bayes, w_combo]);
    legend('MV','GMV','BL','Bayes','COMBO', 'Location','NorthOutside','Orientation','horizontal');
    title(['Asset Allocation Comparison (' curr_mkt ') - Long Only (No Cap)']); ylabel('Weight'); 
    xticks(1:n); xticklabels(names_curr); xtickangle(45); grid on; ylim([0 0.5]);

    subplot(2,1,2); hold on; grid on;
    means = res_mat(1,:); vols = res_mat(2,:); sharpes = res_mat(5,:);
    labels = {'MV','GMV','BL','Bayes','COMBO'};
    colors = lines(5);
    
    for i=1:5
        scatter(vols(i), means(i), 150, colors(i,:), 'filled', 'DisplayName', labels{i});
        text(vols(i), means(i)+0.05, sprintf('%s\nSR:%.2f', labels{i}, sharpes(i)), 'FontSize',8, 'HorizontalAlignment','center');
    end

    scatter(vols(5), means(5), 300, 'p', 'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','y', 'DisplayName', 'Optimal Combo');
    
    xlabel('Monthly Volatility (Std Dev)'); ylabel('Expected Monthly Return (%)');
    title('Efficiency Map: The Power of Diversification across Models'); 
    legend('Location','best');
end