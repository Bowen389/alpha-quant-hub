# -*- coding: utf-8 -*-
"""
大类资产 ETF 动量轮动策略 (加权线性回归斜率 x R^2 + 绝对动量避险)
直接兼容国金证券 PTrade 测试端 / 实盘端（日频回测 & 仿真）
由本地 TDX 量化框架自动验证并导出
"""
import numpy as np


def initialize(context):
    # 风险资产标的池（高流动性、低相关性大类资产）
    g.etf_pool = [
        "513100.SS",  # 纳指ETF
        "510300.SS",  # 沪深300ETF
        "159915.SZ",  # 创业板ETF
        "588000.SS",  # 科创50ETF
        "513180.SS",  # 恒生科技ETF
        "510880.SS",  # 红利ETF
        "518880.SS",  # 黄金ETF
        "511260.SS",  # 十年国债ETF
    ]
    g.safe_etf = "511360.SS"  # 避险资产：短融ETF（全部风险资产动量<=0时切换）

    # 策略核心参数
    g.lookback = 20          # 动量回归窗口（交易日）
    g.top_k = 1                # 持仓数量
    g.rebal_freq = 5      # 调仓周期（交易日）
    g.max_score_cap = 6.0    # 过滤极端过热暴涨标的上限
    g.day_count = 0

    # 设置基准与滑点/佣金（ETF免印花税）
    set_benchmark("510300.SS")
    set_commission(commission_ratio=0.0001, min_commission=5.0, type="ETF")
    set_fixed_slippage(fixedslippage=0.0004)


def calc_r2_momentum_score(prices):
    """计算加权对数价格线性回归年化斜率 * R^2"""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 10 or np.any(prices <= 0) or np.any(~np.isfinite(prices)):
        return np.nan
    y = np.log(prices)
    n = len(y)
    x = np.arange(n, dtype=float)
    w = np.linspace(1.0, 2.0, n)
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    y_bar = (w * y).sum() / w_sum
    dx = x - x_bar
    dy = y - y_bar
    sxx = (w * dx * dx).sum()
    sxy = (w * dx * dy).sum()
    syy = (w * dy * dy).sum()
    if sxx <= 1e-14 or syy <= 1e-14:
        return 0.0
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy)
    ann_slope = np.exp(slope * 250.0) - 1.0
    return float(ann_slope * r2)


def handle_data(context, data):
    # 按调仓周期执行（在日频回测中对应当日开盘/盘中执行，使用的是截至昨日收盘的历史数据，无未来函数）
    if g.day_count % g.rebal_freq != 0:
        g.day_count += 1
        return
    g.day_count += 1

    # 获取截至上一交易日收盘的前复权历史收盘价
    hist = get_history(g.lookback, "1d", "close", security_list=g.etf_pool, fq="pre", include=False)
    scores = {}
    for sec in g.etf_pool:
        try:
            series = hist[sec].dropna().values
            if len(series) >= g.lookback:
                sc = calc_r2_momentum_score(series[-g.lookback:])
                if np.isfinite(sc) and 0.0 < sc <= g.max_score_cap:
                    scores[sec] = sc
        except Exception:
            continue

    # 选出得分最高的 Top K；若无正动量标的则切入避险资产
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        target_list = [sec for sec, _ in ranked[: g.top_k]]
    else:
        target_list = [g.safe_etf]

    # 先卖出不在目标持仓的标的
    current_positions = list(context.portfolio.positions.keys())
    for sec in current_positions:
        if sec not in target_list and context.portfolio.positions[sec].amount > 0:
            order_target_value(sec, 0)

    # 再按等权买入目标标的（留 1% 现金缓冲防手续费不足）
    total_value = context.portfolio.portfolio_value * 0.99
    target_val = total_value / len(target_list)
    for sec in target_list:
        order_target_value(sec, target_val)
