import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import networkx as nx
import scipy as sp
import cvxpy
import sklearn
import fast_tmfg
import os
import time
from itertools import combinations
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties



def make_bar_graph(d: dict, x: str, y: str, title: str, sims: int, highlight_dict: dict = None):
    # Prepare x and y values (sorted)
    sorted_items = sorted(d.items(), key=lambda item: item[1])
    portfolios = [k for k, v in sorted_items]
    scores = [v for k, v in sorted_items]

    labels = ['-'.join(k) for k in portfolios]
    labels = []
    for portfolio in sorted_items:
        labels.append(portfolio[0])

    # Determine bar colors
    colors = []
    widths = []
    for key in portfolios:
        if highlight_dict and highlight_dict.get(key, 0) > sims//2:
            colors.append('crimson')  # Highlighted bar
            widths.append(3)
        else:
            colors.append('#de8585')  # Default bar
            widths.append(1)
    # Plot bar chart
    plt.figure(figsize=(14, 6))
    plt.bar(labels, scores, color=colors, linewidth=widths, edgecolor='black')
    plt.xticks(rotation=90, fontsize=13, fontname="STIXGeneral")
    plt.xlabel(x, fontsize=16, fontname="STIXGeneral")
    plt.ylabel(y, fontsize=16, fontname="STIXGeneral")
    plt.title(title, fontsize=16)
    # legend_elements = [
    # Patch(facecolor='tomato', edgecolor='black', label='Winning Portfolio'),
    # Patch(facecolor='skyblue', edgecolor='black', label='Losing Portfolio'),
    # ]
    # plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("poster_bar_graph.png", dpi=300, bbox_inches="tight")
    plt.show()

# Displays a single centrality bar graph for dual graphs based on S&P 500 wins
def dual_single_bar_graph_wins(data: dict, y_label: str, title: str, wins: dict, sims: int):
    # Identify winning portfolios
    highlighted_portfolios = {k for k, v in wins.items() if v > sims // 2}

    # Sort keys by value
    sorted_keys = sorted(data.keys(), key=lambda k: data[k])

    # Prepare labels and values
    portfolios = ['-'.join(k) for k in sorted_keys]
    values = [data[k] for k in sorted_keys]

    x = np.arange(len(portfolios))
    width = 1

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        if key in highlighted_portfolios:
            color = 'crimson'
            linewidth = 2
        else:
            color = '#de8585'
            linewidth = 1

        plt.bar(x[i], values[i], width, linewidth=linewidth, color=color, edgecolor='black')



    plt.xticks(x, portfolios, rotation=90, fontsize=13, fontname="STIXGeneral")
    plt.xlabel('Portfolio', fontsize=16, fontname='STIXGeneral')
    plt.ylabel(y_label, fontsize=16, fontname='STIXGeneral')
    plt.title(title, fontsize=16, fontname="STIXGeneral")

    # legend_elements = [
    #     Patch(facecolor='tomato', linewidth=2, edgecolor='black', label=f'Winning Portfolio'),
    #     Patch(facecolor='skyblue', edgecolor='black', label=f'Losing Portfolio'),
    # ]
    # plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("poster_graph1.png", dpi=300, bbox_inches="tight")
    plt.show()



# Displays a single centrality bar graph for dual graphs based on expected returns
def dual_single_bar_graph_returns(data: dict, y_label: str, title: str, returns: dict):
    # Identify winning portfolios
    highlighted_portfolios = {k for k, v in returns.items() if v > 0}

    # Sort keys by value
    sorted_keys = sorted(data.keys(), key=lambda k: data[k])

    # Prepare labels and values
    portfolios = ['-'.join(k) for k in sorted_keys]
    values = [data[k] for k in sorted_keys]

    x = np.arange(len(portfolios))
    width = 1

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        if key in highlighted_portfolios:
            color = 'tomato'
            linewidth = 2
        else:
            color = 'skyblue'
            linewidth = 1

        plt.bar(x[i], values[i], width, linewidth=linewidth, color=color, edgecolor='black')

    plt.xticks(x, portfolios, rotation=90, fontsize=8)
    plt.xlabel('Portfolio')
    plt.ylabel(y_label)
    plt.title(title)

    legend_elements = [
        Patch(facecolor='tomato', linewidth=2, edgecolor='black', label=f'Portfolio with a Positive Expected Return'),
        Patch(facecolor='skyblue', edgecolor='black', label=f'Portfolio with a Nonpositive Expected Return'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# Displays a single centrality bar graph for dual graphs based on S&P 500 wins
def dual_edge_weights_single_bar_graph(data: dict, y_label: str, title: str, portfolio_success: dict, sims=None):
    # Identify winning portfolios OR portfolios with positive average returns
    if sims is not None:
        highlighted_portfolios = {k for k, v in portfolio_success.items() if v > sims // 2}
    else:
        highlighted_portfolios = {k for k, v in portfolio_success.items() if v > 0}

    # Sort portflios by value
    sorted_keys = sorted(data.keys(), key=lambda k: data[k])

    # Prepare labels and values
    portfolios = ['-'.join(k) for k in sorted_keys]
    values = [data[k] for k in sorted_keys]

    x = np.arange(len(portfolios))
    width = 1

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        if key in highlighted_portfolios:
            color = 'tomato'
            linewidth = 2
        else:
            color = 'skyblue'
            linewidth = 1

        plt.bar(x[i], values[i], width, linewidth=linewidth, color=color, edgecolor='black')

    plt.xticks(x, portfolios, rotation=90, fontsize=8)
    plt.xlabel('Portfolio')
    plt.ylabel(y_label)
    plt.title(title)

    legend_elements = [
        Patch(facecolor='tomato', linewidth=2, edgecolor='black', label=f'Winning Portfolio'),
        Patch(facecolor='skyblue', edgecolor='black', label=f'Losing Portfolio'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# Displays a single centrality bar graph for TMFGs
def tmfg_single_bar_graph(data: dict, y_label: str):

    # Get stocks
    stocks = data.keys()

    # Sort keys by value
    sorted_keys = sorted(stocks, key=lambda k: data[k])

    # Prepare labels and values
    values = [data[k] for k in sorted_keys]

    x = np.arange(len(sorted_keys))
    width = 1

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        plt.bar(x[i], values[i], width, linewidth=1, color='skyblue', edgecolor='black')

    plt.xticks(x, stocks, rotation=90, fontsize=8)
    plt.xlabel('Stocks')
    plt.ylabel(y_label)
    plt.title(f'{y_label} Per Stock')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# Compares 2 centralities
def compare_bar_graphs(dict1: dict, dict2: dict, label1: str, label2: str, y_label: str, title: str, wins: dict, sims: int):

    # Identify winning portfolios
    highlighted_portfolios = {k for k, v in wins.items() if v > sims // 2}

    # Sort keys by sum of dict1[key] + dict2[key]
    sorted_keys = sorted(dict1.keys(), key=lambda k: dict1[k] + dict2[k])

    # Rebuild ordered lists
    portfolios = ['-'.join(k) for k in sorted_keys]
    values1 = [dict1[k] for k in sorted_keys]
    values2 = [dict2[k] for k in sorted_keys]

    x = np.arange(len(portfolios))
    width = 0.4

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        if key in highlighted_portfolios:
            color1 = 'orange'
            color2 = 'tomato'
            wid = 2
        else:
            color1 = 'cadetblue'
            color2 = 'royalblue'
            wid = 1

        plt.bar(x[i] - width / 2, values1[i], width, linewidth=wid, color=color1, edgecolor='black')
        plt.bar(x[i] + width / 2, values2[i], width, linewidth=wid, color=color2, edgecolor='black')

    plt.xticks(x, portfolios, rotation=90, fontsize=8)
    plt.xlabel('Portfolio (3-stock Combination)')
    plt.ylabel(y_label)
    plt.title(title)
    legend_elements = [
        Patch(facecolor='orange', linewidth=2, edgecolor='black', label=f'{label1} (Winning Portfolios)'),
        Patch(facecolor='cadetblue', edgecolor='black', label=f'{label1} (Losing Portfolios)'),
        Patch(facecolor='tomato', linewidth=2, edgecolor='black', label=f'{label2} (Winning Portfolios)'),
        Patch(facecolor='royalblue', edgecolor='black', label=f'{label2} (Losing Portfolios)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# Compares n centralities
def compare_n_bar_graphs(dict_dict: dict, y_label: str, title: str, wins: dict, sims: int):

    plt.figure(figsize=(16, 6))
    counter = 0

    for d in dict_dict:
        counter += 1
        # Make Highlighted portfolios
        highlighted_portfolios = {k for k, v in wins.items() if v > sims//2}

        # Ensure same portfolio order
        keys = list(d.keys())
        portfolios = ['-'.join(k) for k in keys]
        
        vals = [d[k] for k in keys]

        x = np.arange(len(portfolios))
        width = 0.4

    for i, key in enumerate(keys):
        color1 = 'orange' if key in highlighted_portfolios else 'cadetblue'
        color2 = 'tomato' if key in highlighted_portfolios else 'royalblue'
        plt.bar(x[i] - width/len(dict_dict), vals[i], width, color=color1, edgecolor='black')
        plt.bar(x[i] + width/len(dict_dict), values2[i], width, color=color2, edgecolor='black')

    plt.xticks(x, portfolios, rotation=90, fontsize=8)
    plt.xlabel('Portfolio (3-stock Combination)')
    plt.ylabel(y_label)
    plt.title(title)
    legend_elements = [
    Patch(facecolor='orange', edgecolor='black', label=f'{label1} (Winning Portfolios)'),
    Patch(facecolor='cadetblue', edgecolor='black', label=f'{label1} (Losing Portfolios)'),
    Patch(facecolor='tomato', edgecolor='black', label=f'{label2} (Winning Portfolios)'),
    Patch(facecolor='royalblue', edgecolor='black', label=f'{label2} (Losing Portfolios)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def visualize_stock_data(stock: str, date1: str, date2: str, date3: str, date4: str):
    df = pd.read_csv(f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    df.columns = df.columns.str.strip() 

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

    # Filter date range
    mask = (df['Date'] >= date1) & (df['Date'] <= date4)
    df_filtered = df.loc[mask].sort_values(by='Date')  # Sort so plot is left-to-right in time

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_filtered['Date'], df_filtered['Close'], linestyle='-', color='#4B1F65')
    plt.axvspan(date1, date2, color='#9DA2FF', alpha=0.5)
    plt.axvspan(date3, date4, color='#FF9999', alpha=0.5)
    legend_elements = [
    Patch(facecolor='#9DA2FF', edgecolor='black', label='Data Collection Range'),
    Patch(facecolor='#FF9999', edgecolor='black', label='Portfolio Selling Range'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{stock} Closing Prices ({date1} to {date4})')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks()
    plt.show()

def hot_stocks_in_dual_portfolios(hot_stocks: list[str], y_label: str, SPX_wins=None, avg_returns=None):
    if SPX_wins is not None:
        portfolios = SPX_wins
    elif avg_returns is not None:
        portfolios = avg_returns

    # Count number of hot stocks in each portfolio
    hot_stock_counts = {
        portfolio: sum(1 for stock in hot_stocks if stock in portfolio)
        for portfolio in portfolios
    }

    # Define color mapping for 0, 1, 2, or 3 hot stocks
    color_map = {
        0: '#d0e1f9',  # light blue
        1: '#f4a582',  # orange
        2: '#ca0020',  # red
        3: '#67001f',  # dark red
    }

    # Sort keys by value
    sorted_keys = sorted(portfolios, key=lambda k: portfolios[k])
    values = [portfolios[k] for k in sorted_keys]

    x = np.arange(len(portfolios))
    width = 1

    plt.figure(figsize=(16, 6))

    for i, key in enumerate(sorted_keys):
        count = hot_stock_counts.get(key, 0)
        color = color_map.get(count, '#d0e1f9')  # default to light blue if unexpected
        linewidth = 1 if count == 0 else 2

        plt.bar(x[i], values[i], width, linewidth=linewidth, color=color, edgecolor='black')

    plt.xticks(x, sorted_keys, rotation=90, fontsize=8)
    plt.xlabel('Portfolio')
    plt.ylabel(y_label)
    plt.title(f'{y_label} Per Portfolio')

    # Legend
    legend_elements = [
        Patch(facecolor=color_map[0], edgecolor='black', label='0 Hot Stocks'),
        Patch(facecolor=color_map[1], edgecolor='black', label='1 Hot Stock'),
        Patch(facecolor=color_map[2], edgecolor='black', label='2 Hot Stocks'),
        Patch(facecolor=color_map[3], edgecolor='black', label='3 Hot Stocks'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def visualize_multiple_stock_data(stocks: list[str], y_axis: str, date1: str, date2: str, date3: str, date4: str):
    plt.figure(figsize=(12, 6))

    for stock in stocks:
        df = pd.read_csv(f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        
        # Filter date range
        mask = (df['Date'] >= date1) & (df['Date'] <= date4)
        df_filtered = df.loc[mask].sort_values(by='Date')
        df_filtered['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        plt.plot(df_filtered['Date'], df_filtered[y_axis], label=stock)

    # Add shaded regions
    plt.axvspan(pd.to_datetime(date1), pd.to_datetime(date2), color='#9DA2FF', alpha=0.5)
    plt.axvspan(pd.to_datetime(date3), pd.to_datetime(date4), color='#FF9999', alpha=0.5)

    # Add legend and labels
    legend_elements = [
        Patch(facecolor='#9DA2FF', edgecolor='black', label='Data Collection Range'),
        Patch(facecolor='#FF9999', edgecolor='black', label='Portfolio Selling Range'),
    ]
    plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], loc='upper left')
    plt.xlabel('Date')
    plt.ylabel(f'Daily {y_axis}')
    plt.title(f'Daily {y_axis} ({date1} to {date4})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_portfolios_to_SPX(portfolios: dict, highlighted_stocks: list[tuple], date1: str, date2: str, date3: str, date4: str):
    '''
    Invests $100 into SPX and all portfolios, tracking all performances from date1 to date4.
    '''
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    date3 = pd.to_datetime(date3)
    date4 = pd.to_datetime(date4)

    plt.figure(figsize=(12, 6))

    # Process portfolio
    for stocks, allocation in portfolios.items():
        merged = pd.DataFrame()

        for stock, weight in zip(stocks, allocation):
            df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv',
                parse_dates=['Date'], date_format='%m/%d/%Y'
            )
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')

            mask = (df['Date'] >= date1) & (df['Date'] <= date4)
            df = df.loc[mask].sort_values(by='Date')

            # Normalize values
            df_weighted = (df['Close'] / df['Close'].iloc[0]) * weight
            df_weighted = df_weighted.to_frame(name=stock)

            df_weighted.index = df['Date']

            # Join into merged
            merged = merged.join(df_weighted, how='outer') if not merged.empty else df_weighted

        if merged.empty:
            continue

        portfolio_value = merged.sum(axis=1) * 100
        label = f"{stocks[0]}-{stocks[1]}-{stocks[2]}"
        color = 'blue'
        alpha = 0.5
        zorder = 10
        if type(highlighted_stocks[0]) is not tuple:
            for s in highlighted_stocks:
                if s in stocks:
                    color = 'red'
                    alpha = 0.75
                    zorder = 11
                    break
        elif type(highlighted_stocks[0]) is tuple:
            if stocks in highlighted_stocks:
                color = 'red'
                alpha = 0.75
                zorder = 11
                break

        plt.plot(portfolio_value.index, portfolio_value, alpha=alpha, zorder=zorder, color=color, label=label)

    # Plot SPX
    spx_df = pd.read_csv(
        'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
        parse_dates=['Date'], date_format='%m/%d/%Y'
    )
    spx_df.columns = spx_df.columns.str.strip()
    spx_df['Date'] = pd.to_datetime(spx_df['Date'], format='%m/%d/%y')
    spx_df = spx_df.sort_values('Date')
    mask = (spx_df['Date'] >= date1) & (spx_df['Date'] <= date4)
    spx_df_filtered = spx_df.loc[mask].set_index('Date')
    spx_normalized = spx_df_filtered['Close'] / spx_df_filtered['Close'].iloc[0] * 100
    plt.plot(spx_normalized.index, spx_normalized, label='S&P 500', zorder=12, linewidth=3, color='black')
    plt.axhline(y=100, color='black', linestyle='--', label='$100')

    # Shaded regions
    plt.axvspan(date1, date2, color='#9DA2FF', alpha=0.4)
    plt.axvspan(date3, date4, color='#FF9999', alpha=0.4)

    legend_elements = [Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='S&P 500'), 
                       Line2D([0], [0], color='red', linestyle='-', label='Portfolios With Hot Stocks'), 
                       Line2D([0], [0], color='blue', linestyle='-', label='Portfolios Without Hot Stocks'),
                       Patch(facecolor='#9DA2FF', edgecolor='black', label='Data Collection Period'),
                       Patch(facecolor='#FF9999', edgecolor='black', label='Portfolio Selling Period')]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Starting at $100)')
    plt.title(f'Portfolio Growth ({date1.date()} to {date4.date()})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def graph_dict(d: dict, x_label: str, y_label: str):
    # Sort by keys (optional)
    d = dict(sorted(d.items()))

    x_axis = list(d.keys())
    y_axis = list(d.values())

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(x_axis, y_axis)

    # X-axis tick labels: the keys
    plt.xticks(ticks=x_axis, labels=x_axis)

    # Axis labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()


def heatmap(A, labels, cmap="viridis",
                mask_diagonal=True, tri=None,        # tri: None | "upper" | "lower"
                quantile=None,                       # e.g. (0.02, 0.98) for robust vmin/vmax
                gamma=None,                          # e.g. 0.7 for PowerNorm (more contrast at lows)
                vmin=None, vmax=None,
                rotate=90, save=None, show=True):
    A = np.asarray(A)
    n = A.shape[0]
    assert A.ndim == 2 and A.shape[0] == A.shape[1] == len(labels)

    # Colormap (allow list of colors)
    if isinstance(cmap, (list, tuple)):
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("#d9d9d9")   # masked cells (diag/hidden triangle) in light gray

    # Build mask: diagonal and/or one triangle
    mask = np.zeros_like(A, dtype=bool)
    if mask_diagonal:
        np.fill_diagonal(mask, True)
    if tri == "upper":
        mask |= np.tril(np.ones_like(A, dtype=bool), -1)
    elif tri == "lower":
        mask |= np.triu(np.ones_like(A, dtype=bool),  1)

    M = np.ma.array(A, mask=mask)
    off = A[~mask]
    if off.size == 0:
        raise ValueError("Nothing left to plot after masking.")

    # Scale from off-diagonals only
    if quantile is not None:
        qlo, qhi = quantile
        vmin = np.quantile(off, qlo) if vmin is None else vmin
        vmax = np.quantile(off, qhi) if vmax is None else vmax
    else:
        vmin = off.min() if vmin is None else vmin
        vmax = off.max() if vmax is None else vmax

    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if gamma else \
           mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal", origin="upper")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n), labels=labels)
    ax.set_yticks(np.arange(n), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="left", rotation_mode="anchor", fontsize=9.5)
    plt.setp(ax.get_yticklabels(), fontsize=9.5)
    for s in ax.spines.values(): s.set_visible(False)

    fig.tight_layout()
    if save: fig.savefig(save, dpi=300, bbox_inches="tight")
    # if show: plt.show()
    return ax


# visualize_stock_data('SPX', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('SPX', '2022-03-29', '2022-04-22', '2022-04-25', '2022-05-20')
# visualize_stock_data('SPX', '2024-10-11', '2024-11-19', '2024-11-20', '2024-12-31')
# visualize_stock_data('KMX', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('HAS', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('HSY', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('WYNN', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('ADBE', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('KR', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('COST', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('SJM', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_stock_data('CVS', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')

# visualize_multiple_stock_data(['GOOG'], 'Close', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')
# visualize_multiple_stock_data(['TGT'], 'Close', '2022-03-29', '2022-04-22', '2022-04-25', '2022-05-20')
# visualize_multiple_stock_data(['MCHP'], 'Close', '2024-10-11', '2024-11-19', '2024-11-20', '2024-12-31')
# visualize_multiple_stock_data(['LLY'], 'Close', '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')




# BUY: 2017  |  SELL: 2019 - 2020
# wins_dict = {('AAPL', 'HSY', 'NFLX'): 100, ('AAPL', 'MRK', 'NFLX'): 100, ('DIS', 'KMX', 'NFLX'): 15, ('AAPL', 'ALL', 'KMX'): 100, ('HAS', 'KMX', 'NFLX'): 28, ('AAPL', 'KMX', 'T'): 0, ('ADBE', 'KHC', 'NFLX'): 95, ('AAPL', 'ADBE', 'EA'): 100, ('EA', 'GOOG', 'NFLX'): 100, ('AAPL', 'EA', 'GOOG'): 100, ('COST', 'GOOG', 'NFLX'): 100, ('AAPL', 'GOOG', 'HD'): 100, ('HD', 'KR', 'NFLX'): 97, ('AAPL', 'HD', 'KR'): 100, ('CVS', 'KR', 'NFLX'): 0, ('AAPL', 'KR', 'MNST'): 94, ('MNST', 'NFLX', 'NKE'): 83, ('AAPL', 'MNST', 'NKE'): 100, ('LULU', 'NFLX', 'NKE'): 98, ('AAPL', 'NKE', 'V'): 100, ('JPM', 'NFLX', 'V'): 100, ('AAPL', 'JPM', 'V'): 100, ('FDX', 'JPM', 'NFLX'): 0, ('AAPL', 'BA', 'JPM'): 97, ('DIS', 'MA', 'NFLX'): 44, ('DIS', 'KMX', 'MA'): 41, ('AAPL', 'BBY', 'MA'): 100, ('BBY', 'KMX', 'MA'): 100, ('AAPL', 'BBY', 'KMI'): 100, ('BBY', 'KMI', 'KMX'): 26, ('ADBE', 'HAS', 'NFLX'): 100, ('ADBE', 'HAS', 'KMX'): 90, ('LULU', 'NFLX', 'V'): 100, ('LULU', 'NKE', 'V'): 100, ('BA', 'FDX', 'NFLX'): 18, ('BA', 'FDX', 'JPM'): 19, ('LLY', 'MA', 'NFLX'): 100, ('AAPL', 'LLY', 'MA'): 100, ('JNJ', 'LLY', 'NFLX'): 59, ('AAPL', 'LLY', 'MRK'): 100, ('EA', 'KHC', 'NFLX'): 0, ('ADBE', 'EA', 'KHC'): 77, ('KO', 'MNST', 'NFLX'): 9, ('CVS', 'KR', 'MNST'): 0, ('COST', 'HD', 'NFLX'): 100, ('COST', 'GOOG', 'HD'): 100, ('BA', 'HSY', 'NFLX'): 28, ('AAPL', 'BA', 'HSY'): 100, ('JNJ', 'MRK', 'NFLX'): 12, ('JNJ', 'LLY', 'MRK'): 24, ('AAPL', 'ALL', 'KMI'): 98, ('ALL', 'KMI', 'KMX'): 10, ('CVS', 'KO', 'NFLX'): 0, ('CVS', 'KO', 'MNST'): 0, ('AAPL', 'ADBE', 'T'): 55, ('ADBE', 'KMX', 'T'): 10}

# dual_TMFG_betweenness = {('AAPL', 'HSY', 'NFLX'): 0.211, ('AAPL', 'MRK', 'NFLX'): 0.213, ('DIS', 'KMX', 'NFLX'): 0.132, ('AAPL', 'ALL', 'KMX'): 0.106, ('HAS', 'KMX', 'NFLX'): 0.134, ('AAPL', 'KMX', 'T'): 0.115, ('ADBE', 'KHC', 'NFLX'): 0.106, ('AAPL', 'ADBE', 'EA'): 0.191, ('EA', 'GOOG', 'NFLX'): 0.099, ('AAPL', 'EA', 'GOOG'): 0.223, ('COST', 'GOOG', 'NFLX'): 0.055, ('AAPL', 'GOOG', 'HD'): 0.226, ('HD', 'KR', 'NFLX'): 0.074, ('AAPL', 'HD', 'KR'): 0.243, ('CVS', 'KR', 'NFLX'): 0.042, ('AAPL', 'KR', 'MNST'): 0.241, ('MNST', 'NFLX', 'NKE'): 0.07, ('AAPL', 'MNST', 'NKE'): 0.243, ('LULU', 'NFLX', 'NKE'): 0.026, ('AAPL', 'NKE', 'V'): 0.216, ('JPM', 'NFLX', 'V'): 0.056, ('AAPL', 'JPM', 'V'): 0.208, ('FDX', 'JPM', 'NFLX'): 0.034, ('AAPL', 'BA', 'JPM'): 0.18, ('DIS', 'MA', 'NFLX'): 0.108, ('DIS', 'KMX', 'MA'): 0.027, ('AAPL', 'BBY', 'MA'): 0.103, ('BBY', 'KMX', 'MA'): 0.036, ('AAPL', 'BBY', 'KMI'): 0.07, ('BBY', 'KMI', 'KMX'): 0.026, ('ADBE', 'HAS', 'NFLX'): 0.104, ('ADBE', 'HAS', 'KMX'): 0.037, ('LULU', 'NFLX', 'V'): 0.025, ('LULU', 'NKE', 'V'): 0.013, ('BA', 'FDX', 'NFLX'): 0.042, ('BA', 'FDX', 'JPM'): 0.008, ('LLY', 'MA', 'NFLX'): 0.115, ('AAPL', 'LLY', 'MA'): 0.151, ('JNJ', 'LLY', 'NFLX'): 0.06, ('AAPL', 'LLY', 'MRK'): 0.147, ('EA', 'KHC', 'NFLX'): 0.082, ('ADBE', 'EA', 'KHC'): 0.03, ('KO', 'MNST', 'NFLX'): 0.032, ('CVS', 'KR', 'MNST'): 0.037, ('COST', 'HD', 'NFLX'): 0.045, ('COST', 'GOOG', 'HD'): 0.011, ('BA', 'HSY', 'NFLX'): 0.054, ('AAPL', 'BA', 'HSY'): 0.156, ('JNJ', 'MRK', 'NFLX'): 0.059, ('JNJ', 'LLY', 'MRK'): 0.003, ('AAPL', 'ALL', 'KMI'): 0.061, ('ALL', 'KMI', 'KMX'): 0.027, ('CVS', 'KO', 'NFLX'): 0.02, ('CVS', 'KO', 'MNST'): 0.006, ('AAPL', 'ADBE', 'T'): 0.151, ('ADBE', 'KMX', 'T'): 0.043}
# make_bar_graph(dual_TMFG_betweenness, 'Portfolio (3-stock Combination)', 'Betweenness Centrality', 'Dual TMFG Betweenness Centrality Per Portfolio', wins_dict)

# dual_TMFG_closeness = {('AAPL', 'HSY', 'NFLX'): 0.168, ('AAPL', 'MRK', 'NFLX'): 0.168, ('DIS', 'KMX', 'NFLX'): 0.163, ('AAPL', 'ALL', 'KMX'): 0.165, ('HAS', 'KMX', 'NFLX'): 0.165, ('AAPL', 'KMX', 'T'): 0.17, ('ADBE', 'KHC', 'NFLX'): 0.166, ('AAPL', 'ADBE', 'EA'): 0.183, ('EA', 'GOOG', 'NFLX'): 0.175, ('AAPL', 'EA', 'GOOG'): 0.185, ('COST', 'GOOG', 'NFLX'): 0.164, ('AAPL', 'GOOG', 'HD'): 0.185, ('HD', 'KR', 'NFLX'): 0.167, ('AAPL', 'HD', 'KR'): 0.185, ('CVS', 'KR', 'NFLX'): 0.157, ('AAPL', 'KR', 'MNST'): 0.183, ('MNST', 'NFLX', 'NKE'): 0.159, ('AAPL', 'MNST', 'NKE'): 0.18, ('LULU', 'NFLX', 'NKE'): 0.149, ('AAPL', 'NKE', 'V'): 0.177, ('JPM', 'NFLX', 'V'): 0.155, ('AAPL', 'JPM', 'V'): 0.174, ('FDX', 'JPM', 'NFLX'): 0.152, ('AAPL', 'BA', 'JPM'): 0.171, ('DIS', 'MA', 'NFLX'): 0.161, ('DIS', 'KMX', 'MA'): 0.154, ('AAPL', 'BBY', 'MA'): 0.162, ('BBY', 'KMX', 'MA'): 0.157, ('AAPL', 'BBY', 'KMI'): 0.159, ('BBY', 'KMI', 'KMX'): 0.152, ('ADBE', 'HAS', 'NFLX'): 0.166, ('ADBE', 'HAS', 'KMX'): 0.164, ('LULU', 'NFLX', 'V'): 0.147, ('LULU', 'NKE', 'V'): 0.155, ('BA', 'FDX', 'NFLX'): 0.152, ('BA', 'FDX', 'JPM'): 0.15, ('LLY', 'MA', 'NFLX'): 0.161, ('AAPL', 'LLY', 'MA'): 0.168, ('JNJ', 'LLY', 'NFLX'): 0.159, ('AAPL', 'LLY', 'MRK'): 0.167, ('EA', 'KHC', 'NFLX'): 0.169, ('ADBE', 'EA', 'KHC'): 0.17, ('KO', 'MNST', 'NFLX'): 0.146, ('CVS', 'KR', 'MNST'): 0.16, ('COST', 'HD', 'NFLX'): 0.162, ('COST', 'GOOG', 'HD'): 0.165, ('BA', 'HSY', 'NFLX'): 0.159, ('AAPL', 'BA', 'HSY'): 0.169, ('JNJ', 'MRK', 'NFLX'): 0.158, ('JNJ', 'LLY', 'MRK'): 0.151, ('AAPL', 'ALL', 'KMI'): 0.159, ('ALL', 'KMI', 'KMX'): 0.153, ('CVS', 'KO', 'NFLX'): 0.147, ('CVS', 'KO', 'MNST'): 0.143, ('AAPL', 'ADBE', 'T'): 0.178, ('ADBE', 'KMX', 'T'): 0.168}
# make_bar_graph(dual_TMFG_closeness, 'Portfolio (3-stock Combination)', 'Closeness Centrality', 'Dual TMFG Closeness Centrality Per Portfolio', wins_dict)

# compare_bar_graphs(dual_TMFG_betweenness, dual_TMFG_closeness, 'Betweenness Centrality', 'Closeness Centrality', 'Centrality Score', 'Betweenness vs Closeness Centrality of Dual TMFG Portfolios', wins_dict)





# BUY: 2023  |  SELL: 2024 - 2025

# wins_dict = {('AAPL', 'FDX', 'NKE'): 0, ('AAPL', 'FDX', 'KHC'): 0, ('AAPL', 'BA', 'KMI'): 20, ('BA', 'FDX', 'HAS'): 0, ('AAPL', 'BA', 'V'): 17, ('BA', 'FDX', 'MNST'): 3, ('AAPL', 'ADBE', 'DIS'): 28, ('ADBE', 'DIS', 'FDX'): 35, ('AAPL', 'DIS', 'GOOG'): 76, ('DIS', 'FDX', 'GOOG'): 76, ('AAPL', 'GOOG', 'LLY'): 100, ('FDX', 'GOOG', 'HD'): 88, ('AAPL', 'HD', 'KMX'): 1, ('FDX', 'HD', 'KMX'): 0, ('AAPL', 'KMX', 'LULU'): 58, ('FDX', 'KMX', 'LULU'): 32, ('AAPL', 'LULU', 'MA'): 32, ('FDX', 'LULU', 'MA'): 36, ('AAPL', 'COST', 'MA'): 100, ('FDX', 'MA', 'NFLX'): 97, ('AAPL', 'NFLX', 'NKE'): 0, ('FDX', 'NFLX', 'NKE'): 0, ('BBY', 'FDX', 'HAS'): 0, ('BA', 'BBY', 'HAS'): 0, ('AAPL', 'ADBE', 'V'): 76, ('ADBE', 'BA', 'V'): 18, ('AAPL', 'BBY', 'JPM'): 94, ('ALL', 'BBY', 'FDX'): 1, ('AAPL', 'BBY', 'KMI'): 40, ('BA', 'BBY', 'KMI'): 0, ('AAPL', 'EA', 'HD'): 0, ('EA', 'GOOG', 'HD'): 2, ('AAPL', 'COST', 'NFLX'): 100, ('COST', 'MA', 'NFLX'): 100, ('ADBE', 'FDX', 'MNST'): 16, ('ADBE', 'BA', 'KO'): 0, ('FDX', 'JPM', 'T'): 0, ('ALL', 'BBY', 'JPM'): 7, ('AAPL', 'JPM', 'MRK'): 0, ('CVS', 'FDX', 'JPM'): 0, ('AAPL', 'CVS', 'KR'): 0, ('CVS', 'FDX', 'KHC'): 0, ('AAPL', 'EA', 'LLY'): 100, ('EA', 'GOOG', 'LLY'): 100, ('ALL', 'FDX', 'T'): 0, ('ALL', 'JPM', 'T'): 0, ('AAPL', 'KHC', 'KR'): 0, ('CVS', 'HSY', 'KHC'): 0, ('BA', 'KO', 'MNST'): 0, ('ADBE', 'KO', 'MNST'): 0, ('AAPL', 'CVS', 'JNJ'): 0, ('CVS', 'JPM', 'MRK'): 0, ('CVS', 'HSY', 'KR'): 0, ('HSY', 'KHC', 'KR'): 0, ('AAPL', 'JNJ', 'MRK'): 0, ('CVS', 'JNJ', 'MRK'): 0}

# dual_TMFG_betweenness = {('AAPL', 'FDX', 'NKE'): 0.209, ('AAPL', 'FDX', 'KHC'): 0.225, ('AAPL', 'BA', 'KMI'): 0.147, ('BA', 'FDX', 'HAS'): 0.122, ('AAPL', 'BA', 'V'): 0.145, ('BA', 'FDX', 'MNST'): 0.129, ('AAPL', 'ADBE', 'DIS'): 0.138, ('ADBE', 'DIS', 'FDX'): 0.164, ('AAPL', 'DIS', 'GOOG'): 0.105, ('DIS', 'FDX', 'GOOG'): 0.186, ('AAPL', 'GOOG', 'LLY'): 0.054, ('FDX', 'GOOG', 'HD'): 0.193, ('AAPL', 'HD', 'KMX'): 0.064, ('FDX', 'HD', 'KMX'): 0.173, ('AAPL', 'KMX', 'LULU'): 0.068, ('FDX', 'KMX', 'LULU'): 0.156, ('AAPL', 'LULU', 'MA'): 0.063, ('FDX', 'LULU', 'MA'): 0.159, ('AAPL', 'COST', 'MA'): 0.041, ('FDX', 'MA', 'NFLX'): 0.157, ('AAPL', 'NFLX', 'NKE'): 0.06, ('FDX', 'NFLX', 'NKE'): 0.141, ('BBY', 'FDX', 'HAS'): 0.108, ('BA', 'BBY', 'HAS'): 0.026, ('AAPL', 'ADBE', 'V'): 0.125, ('ADBE', 'BA', 'V'): 0.03, ('AAPL', 'BBY', 'JPM'): 0.181, ('ALL', 'BBY', 'FDX'): 0.111, ('AAPL', 'BBY', 'KMI'): 0.159, ('BA', 'BBY', 'KMI'): 0.028, ('AAPL', 'EA', 'HD'): 0.037, ('EA', 'GOOG', 'HD'): 0.027, ('AAPL', 'COST', 'NFLX'): 0.048, ('COST', 'MA', 'NFLX'): 0.004, ('ADBE', 'FDX', 'MNST'): 0.132, ('ADBE', 'BA', 'KO'): 0.019, ('FDX', 'JPM', 'T'): 0.104, ('ALL', 'BBY', 'JPM'): 0.043, ('AAPL', 'JPM', 'MRK'): 0.153, ('CVS', 'FDX', 'JPM'): 0.185, ('AAPL', 'CVS', 'KR'): 0.061, ('CVS', 'FDX', 'KHC'): 0.191, ('AAPL', 'EA', 'LLY'): 0.026, ('EA', 'GOOG', 'LLY'): 0.006, ('ALL', 'FDX', 'T'): 0.079, ('ALL', 'JPM', 'T'): 0.019, ('AAPL', 'KHC', 'KR'): 0.069, ('CVS', 'HSY', 'KHC'): 0.028, ('BA', 'KO', 'MNST'): 0.016, ('ADBE', 'KO', 'MNST'): 0.021, ('AAPL', 'CVS', 'JNJ'): 0.06, ('CVS', 'JPM', 'MRK'): 0.086, ('CVS', 'HSY', 'KR'): 0.007, ('HSY', 'KHC', 'KR'): 0.005, ('AAPL', 'JNJ', 'MRK'): 0.062, ('CVS', 'JNJ', 'MRK'): 0.007}
# make_bar_graph(dual_TMFG_betweenness, 'Portfolio (3-stock Combination)', 'Betweenness Centrality', 'Dual TMFG Betweenness Centrality Per Portfolio', wins_dict)

# dual_TMFG_closeness = {('AAPL', 'FDX', 'NKE'): 0.174, ('AAPL', 'FDX', 'KHC'): 0.179, ('AAPL', 'BA', 'KMI'): 0.181, ('BA', 'FDX', 'HAS'): 0.18, ('AAPL', 'BA', 'V'): 0.179, ('BA', 'FDX', 'MNST'): 0.179, ('AAPL', 'ADBE', 'DIS'): 0.177, ('ADBE', 'DIS', 'FDX'): 0.182, ('AAPL', 'DIS', 'GOOG'): 0.171, ('DIS', 'FDX', 'GOOG'): 0.18, ('AAPL', 'GOOG', 'LLY'): 0.155, ('FDX', 'GOOG', 'HD'): 0.177, ('AAPL', 'HD', 'KMX'): 0.157, ('FDX', 'HD', 'KMX'): 0.175, ('AAPL', 'KMX', 'LULU'): 0.155, ('FDX', 'KMX', 'LULU'): 0.173, ('AAPL', 'LULU', 'MA'): 0.155, ('FDX', 'LULU', 'MA'): 0.172, ('AAPL', 'COST', 'MA'): 0.154, ('FDX', 'MA', 'NFLX'): 0.172, ('AAPL', 'NFLX', 'NKE'): 0.164, ('FDX', 'NFLX', 'NKE'): 0.172, ('BBY', 'FDX', 'HAS'): 0.18, ('BA', 'BBY', 'HAS'): 0.173, ('AAPL', 'ADBE', 'V'): 0.177, ('ADBE', 'BA', 'V'): 0.168, ('AAPL', 'BBY', 'JPM'): 0.184, ('ALL', 'BBY', 'FDX'): 0.18, ('AAPL', 'BBY', 'KMI'): 0.183, ('BA', 'BBY', 'KMI'): 0.173, ('AAPL', 'EA', 'HD'): 0.152, ('EA', 'GOOG', 'HD'): 0.156, ('AAPL', 'COST', 'NFLX'): 0.155, ('COST', 'MA', 'NFLX'): 0.152, ('ADBE', 'FDX', 'MNST'): 0.178, ('ADBE', 'BA', 'KO'): 0.158, ('FDX', 'JPM', 'T'): 0.176, ('ALL', 'BBY', 'JPM'): 0.177, ('AAPL', 'JPM', 'MRK'): 0.18, ('CVS', 'FDX', 'JPM'): 0.183, ('AAPL', 'CVS', 'KR'): 0.159, ('CVS', 'FDX', 'KHC'): 0.18, ('AAPL', 'EA', 'LLY'): 0.15, ('EA', 'GOOG', 'LLY'): 0.148, ('ALL', 'FDX', 'T'): 0.174, ('ALL', 'JPM', 'T'): 0.17, ('AAPL', 'KHC', 'KR'): 0.162, ('CVS', 'HSY', 'KHC'): 0.158, ('BA', 'KO', 'MNST'): 0.157, ('ADBE', 'KO', 'MNST'): 0.16, ('AAPL', 'CVS', 'JNJ'): 0.162, ('CVS', 'JPM', 'MRK'): 0.176, ('CVS', 'HSY', 'KR'): 0.144, ('HSY', 'KHC', 'KR'): 0.147, ('AAPL', 'JNJ', 'MRK'): 0.165, ('CVS', 'JNJ', 'MRK'): 0.154}
# make_bar_graph(dual_TMFG_closeness, 'Portfolio (3-stock Combination)', 'Closeness Centrality', 'Dual TMFG Closeness Centrality Per Portfolio', wins_dict)

# compare_bar_graphs(dual_TMFG_betweenness, dual_TMFG_closeness, 'Betweenness Centrality', 'Closeness Centrality', 'Centrality Score', 'Betweenness vs Closeness Centrality of Dual TMFG Portfolios', wins_dict)