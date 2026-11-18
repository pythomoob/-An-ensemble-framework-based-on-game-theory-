#!/usr/bin/env python
# coding: utf-8
"""
博弈论模型集成脚本
方法：Game Theory（博弈论子博弈均衡）
策略：测试集前 N_TRAIN 行用于训练权重，剩余行测试
"""

import numpy as np
import pandas as pd
import itertools
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

N_TRAIN = 2000


# ============================================================================
# 通用工具函数
# ============================================================================

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    mse = mean_squared_error(y_true, y_pred)
    return {
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100,
        'R2': r2_score(y_true, y_pred),
    }


def natural_sort_key(name):
    parts = name.rsplit('_Step_', 1)
    return (parts[0], int(parts[1]) if len(parts) > 1 else 0)


# ============================================================================
# Game Theory (博弈论子博弈均衡)
# ============================================================================

def find_optimal_difference_subset(data, n_models):
    base = data[0]
    differences = [data[i] - base for i in range(1, n_models + 1)]
    best_abs_sum = float('inf')
    best_subset_positions = []
    for r in range(1, n_models + 1):
        for subset_indices in itertools.combinations(range(n_models), r):
            abs_sum = abs(sum(differences[i] for i in subset_indices))
            if abs_sum < best_abs_sum:
                best_abs_sum = abs_sum
                best_subset_positions = [i + 1 for i in subset_indices]
                if abs_sum == 0:
                    return {'optimal_positions': best_subset_positions}
    return {'optimal_positions': best_subset_positions}


def game_theory_weights(train_df, model_cols):
    n_models = len(model_cols)
    zero_df = pd.DataFrame(0.0, index=train_df.index, columns=train_df.columns)
    for i in range(len(train_df)):
        result = find_optimal_difference_subset(train_df.iloc[i].tolist(), n_models)
        for j in result['optimal_positions']:
            zero_df.iloc[i, j] = 1
    totals = zero_df.sum().tolist()
    model_totals = totals[1:]
    total = sum(model_totals)
    return np.array([t / total for t in model_totals])


# ============================================================================
# 主流程
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, 'houston gru调参')
    output_path = os.path.join(base_dir, 'houston_game_theory_1h调参2.xlsx')

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.xlsx')])
    print('Method: Game Theory')
    print(f'N_TRAIN: {N_TRAIN}')
    print(f'Files: {len(test_files)}')
    print('=' * 70)

    all_predictions = []
    all_metrics = []
    all_weights = []

    for tf_idx, test_file in enumerate(test_files):
        site_name = os.path.splitext(test_file)[0]
        file_path = os.path.join(test_dir, test_file)
        xls = pd.ExcelFile(file_path)
        sheets = xls.sheet_names

        for sheet in sheets:
            site_step = f'{site_name}_{sheet}'
            df = pd.read_excel(file_path, sheet_name=sheet)
            model_cols = df.columns[1:]
            model_list = list(model_cols)

            train_chunk = df.iloc[:N_TRAIN]
            test_chunk = df.iloc[N_TRAIN:]

            y_test = test_chunk.iloc[:, 0].values
            X_test = test_chunk.iloc[:, 1:].values

            # 构建预测 DataFrame
            pred_df = pd.DataFrame({'Site_Step': site_step, 'True_Value': y_test})
            for idx, col in enumerate(model_cols):
                pred_df[col] = X_test[:, idx]

            # 博弈论集成
            weights = game_theory_weights(train_chunk, model_list)
            ensemble_pred = (X_test * weights).sum(axis=1)
            pred_df['Ensemble_Pred'] = ensemble_pred

            # 评估指标
            m = compute_metrics(y_test, ensemble_pred)
            m['Model'] = 'Ensemble'
            m['Site_Step'] = site_step
            all_metrics.append(m)

            # 权重
            wdf = pd.DataFrame({'Model': model_list, 'Weight': weights})
            wdf['Site_Step'] = site_step
            all_weights.append(wdf)

            # 子模型指标
            for idx, col in enumerate(model_cols):
                m = compute_metrics(y_test, X_test[:, idx])
                m['Model'] = col
                m['Site_Step'] = site_step
                all_metrics.append(m)

            all_predictions.append(pred_df)

        print(f'[{tf_idx + 1}/{len(test_files)}] {site_name} done', flush=True)

    # ========================================================================
    # 保存
    # ========================================================================
    print(f'\nSaving to: {output_path}', flush=True)

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    predictions_all['_sort'] = predictions_all['Site_Step'].map(natural_sort_key)
    predictions_all = predictions_all.sort_values('_sort', kind='stable').drop(columns=['_sort'])

    metrics_all = pd.DataFrame(all_metrics)
    metrics_all = metrics_all[['Site_Step', 'Model', 'RMSE', 'MAE', 'MAPE', 'R2']]
    metrics_all['_sort'] = metrics_all['Site_Step'].map(natural_sort_key)
    metrics_all = metrics_all.sort_values(['_sort', 'Model'], kind='stable').drop(columns=['_sort'])

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Predictions 行数超限时分两个 sheet
        n_pred = len(predictions_all)
        if n_pred > 1000000:
            mid = n_pred // 2
            predictions_all.iloc[:mid].to_excel(writer, sheet_name='Predictions_1', index=False)
            predictions_all.iloc[mid:].to_excel(writer, sheet_name='Predictions_2', index=False)
        else:
            predictions_all.to_excel(writer, sheet_name='Predictions', index=False)
        metrics_all.to_excel(writer, sheet_name='Metrics', index=False)

        # 博弈论权重
        wa = pd.concat(all_weights, ignore_index=True)
        wp = wa.pivot(index='Model', columns='Site_Step', values='Weight')
        sorted_cols = sorted(wp.columns, key=natural_sort_key)
        wp = wp[sorted_cols]
        wp.to_excel(writer, sheet_name='Weights_Ensemble')

    # ========================================================================
    # 标黄
    # ========================================================================
    wb = load_workbook(output_path)
    ws = wb['Metrics']
    yellow = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
    df_m = metrics_all.reset_index(drop=True)
    for group in df_m['Site_Step'].unique():
        mask = df_m['Site_Step'] == group
        ws.cell(row=df_m[mask]['RMSE'].idxmin() + 2, column=3).fill = yellow
        ws.cell(row=df_m[mask]['MAE'].idxmin() + 2, column=4).fill = yellow
        ws.cell(row=df_m[mask]['MAPE'].idxmin() + 2, column=5).fill = yellow
        ws.cell(row=df_m[mask]['R2'].idxmax() + 2, column=6).fill = yellow
    wb.save(output_path)

    # ========================================================================
    # 全局指标汇总
    # ========================================================================
    y_all = predictions_all['True_Value'].values
    g_results = []
    for col in predictions_all.columns:
        if col in ('Site_Step', 'True_Value', '_sort'):
            continue
        m = compute_metrics(y_all, predictions_all[col].values)
        m['Model'] = col
        g_results.append(m)

    rdf = pd.DataFrame(g_results).sort_values('RMSE', ignore_index=True)

    print(f'\nGlobal Metrics ({metrics_all["Site_Step"].nunique()} groups, {len(predictions_all)} samples):')
    print('=' * 75)
    for i, (_, r) in enumerate(rdf.iterrows()):
        tag = ' [集成]' if r['Model'] == 'Ensemble_Pred' else ''
        print(f'{i + 1:2d}. {r["Model"]:22s} RMSE={r["RMSE"]:.4f}  MAE={r["MAE"]:.4f}  '
              f'MAPE={r["MAPE"]:.2f}%  R2={r["R2"]:.4f}{tag}')
    # 保存全局指标到新分表
    g_df = rdf[['Model', 'RMSE', 'MAE', 'MAPE', 'R2']]
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        g_df.to_excel(writer, sheet_name='Global_Metrics', index=False)
    print('\nDone!')


if __name__ == '__main__':
    main()
