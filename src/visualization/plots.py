"""
Portfolio visualization module.

This module creates compelling visualizations that transform complex portfolio
analytics into intuitive visual insights. Each visualization is designed to
answer specific questions investors ask about their portfolios.

The philosophy here is that a good visualization should tell a story at a glance
while rewarding deeper examination with additional insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure default plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


class PortfolioVisualizer:
    """
    Create professional visualizations for portfolio analysis.
    
    This class provides a comprehensive suite of visualization tools that help
    investors understand portfolio performance, risk, and characteristics through
    carefully designed charts and plots.
    
    Each method creates a specific type of visualization designed to answer
    common investment questions like "How risky is my portfolio?" or 
    "When did my strategy underperform?"
    """
    
    def __init__(self, style: str = 'professional'):
        """
        Initialize the visualizer with a specific style.
        
        Args:
            style: Visual style preset ('professional', 'academic', 'presentation')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """
        Configure the visual style based on the chosen preset.
        
        Different contexts require different visual approaches. A chart for
        an academic paper needs different styling than one for a client presentation.
        """
        if self.style == 'professional':
            # Clean, business-appropriate styling
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            self.bg_color = 'white'
            self.grid_alpha = 0.3
            
        elif self.style == 'academic':
            # More austere styling suitable for journals
            self.colors = ['#000000', '#666666', '#999999', '#cccccc']
            self.bg_color = 'white'
            self.grid_alpha = 0.5
            
        elif self.style == 'presentation':
            # High contrast for projection
            self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            self.bg_color = '#f8f9fa'
            self.grid_alpha = 0.2
    
    def plot_efficient_frontier(
        self,
        returns: List[float],
        volatilities: List[float],
        sharpe_ratios: List[float],
        optimal_portfolio: Optional[Dict] = None,
        current_portfolio: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create an efficient frontier visualization.
        
        The efficient frontier is perhaps the most iconic visualization in finance.
        It shows the trade-off between risk and return, helping investors understand
        that higher returns require accepting higher risk. The curve represents
        optimal portfolios - you can't get better returns without taking more risk.
        
        Args:
            returns: Expected returns for each portfolio
            volatilities: Risk (standard deviation) for each portfolio
            sharpe_ratios: Risk-adjusted returns for each portfolio
            optimal_portfolio: Dict with 'return', 'volatility' for optimal point
            current_portfolio: Dict with 'return', 'volatility' for current point
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create the main scatter plot
        # Color points by Sharpe ratio to show risk-adjusted performance
        scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, 
                           cmap='viridis', s=50, alpha=0.6, edgecolors='none')
        
        # Add a colorbar to show Sharpe ratio scale
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Find and plot the efficient frontier curve
        # Sort points by volatility to create a smooth curve
        sorted_indices = np.argsort(volatilities)
        sorted_vols = [volatilities[i] for i in sorted_indices]
        sorted_returns = [returns[i] for i in sorted_indices]
        
        # Plot the upper edge (efficient frontier)
        # We'll use a simple approach: for each volatility level, take the maximum return
        unique_vols = sorted(list(set(sorted_vols)))
        frontier_returns = []
        for vol in unique_vols:
            vol_returns = [r for r, v in zip(returns, volatilities) if abs(v - vol) < 0.001]
            if vol_returns:
                frontier_returns.append(max(vol_returns))
            else:
                frontier_returns.append(np.nan)
        
        # Plot the frontier line
        ax.plot(unique_vols, frontier_returns, 'r-', linewidth=2, 
                label='Efficient Frontier', alpha=0.8)
        
        # Mark special portfolios if provided
        if optimal_portfolio:
            ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                      color='red', s=200, marker='*', edgecolors='black', 
                      linewidth=2, label='Optimal Portfolio', zorder=5)
            
            # Add annotation for optimal portfolio
            ax.annotate('Optimal\nPortfolio', 
                       xy=(optimal_portfolio['volatility'], optimal_portfolio['return']),
                       xytext=(optimal_portfolio['volatility'] + 0.02, 
                              optimal_portfolio['return'] - 0.02),
                       fontsize=10, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if current_portfolio:
            ax.scatter(current_portfolio['volatility'], current_portfolio['return'], 
                      color='blue', s=200, marker='o', edgecolors='black', 
                      linewidth=2, label='Current Portfolio', zorder=5)
        
        # Styling and labels
        ax.set_xlabel('Annual Volatility (Risk)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Expected Annual Return', fontsize=14, fontweight='bold')
        ax.set_title('Efficient Frontier: Risk-Return Trade-off', fontsize=16, fontweight='bold', pad=20)
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Add grid for easier reading
        ax.grid(True, alpha=self.grid_alpha, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add informative text box
        textstr = ('The Efficient Frontier represents optimal portfolios that offer\n'
                  'the highest expected return for each level of risk. Portfolios\n'
                  'below the frontier are suboptimal; those above are unattainable.')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_portfolio_performance(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        events: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive performance visualization.
        
        This chart tells the story of a portfolio's journey over time, showing
        not just the end result but the path taken to get there. The inclusion
        of drawdowns and rolling metrics reveals the lived experience of holding
        the portfolio through various market conditions.
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_values: Optional benchmark for comparison
            events: Dict of {date: event_name} for marking significant events
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Create a figure with multiple subplots for comprehensive analysis
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main performance chart
        ax1 = fig.add_subplot(gs[0])
        
        # Calculate cumulative returns
        portfolio_returns = portfolio_values.pct_change().fillna(0)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Plot portfolio performance
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                 linewidth=2.5, label='Portfolio', color=self.colors[0])
        
        # Plot benchmark if provided
        if benchmark_values is not None:
            benchmark_returns = benchmark_values.pct_change().fillna(0)
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    linewidth=2, label='Benchmark', color=self.colors[1], alpha=0.7)
        
        # Mark significant events if provided
        if events:
            for date, event in events.items():
                if pd.to_datetime(date) in cumulative_returns.index:
                    ax1.axvline(x=pd.to_datetime(date), color='red', 
                              linestyle='--', alpha=0.5)
                    ax1.text(pd.to_datetime(date), ax1.get_ylim()[1] * 0.95, 
                            event, rotation=90, verticalalignment='top', 
                            fontsize=9, alpha=0.7)
        
        ax1.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=self.grid_alpha)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{(y-1)*100:.0f}%'))
        
        # Drawdown chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Calculate drawdowns
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        # Fill the drawdown area
        ax2.fill_between(drawdowns.index, 0, drawdowns.values * 100, 
                        color='red', alpha=0.3)
        ax2.plot(drawdowns.index, drawdowns.values * 100, 
                color='darkred', linewidth=1)
        
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_ylim(top=0)
        ax2.grid(True, alpha=self.grid_alpha)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        
        # Rolling volatility
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252) * 100
        ax3.plot(rolling_vol.index, rolling_vol.values, 
                color=self.colors[2], linewidth=1.5)
        ax3.fill_between(rolling_vol.index, 0, rolling_vol.values, 
                        color=self.colors[2], alpha=0.2)
        
        ax3.set_ylabel('Rolling Volatility %', fontsize=12)
        ax3.grid(True, alpha=self.grid_alpha)
        
        # Rolling Sharpe ratio
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        
        rolling_sharpe = (portfolio_returns.rolling(window=252).mean() * 252) / \
                        (portfolio_returns.rolling(window=252).std() * np.sqrt(252))
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 
                color=self.colors[3], linewidth=1.5)
        ax4.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax4.axhline(y=1, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
        
        ax4.set_ylabel('Rolling Sharpe', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.grid(True, alpha=self.grid_alpha)
        
        # Hide x-axis labels for all but bottom plot
        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([])
            
        plt.suptitle('Comprehensive Portfolio Analysis Dashboard', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_asset_allocation(
        self,
        weights: Dict[str, float],
        title: str = "Portfolio Allocation",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create an intuitive visualization of portfolio weights.
        
        This visualization helps investors understand their portfolio composition
        at a glance. We use both a pie chart and a bar chart because different
        people process information differently - some prefer parts of a whole
        (pie) while others prefer direct comparison (bars).
        
        Args:
            weights: Dictionary mapping asset names to weights
            title: Chart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Filter out zero weights for cleaner visualization
        weights = {k: v for k, v in weights.items() if v > 0.001}
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = ax1.pie(weights.values(), labels=weights.keys(), 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors, textprops={'fontsize': 12})
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
            
        ax1.set_title(f'{title} - Pie Chart', fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart
        assets = list(weights.keys())
        values = list(weights.values())
        
        bars = ax2.bar(assets, values, color=colors)
        ax2.set_ylim(0, max(values) * 1.2)
        ax2.set_ylabel('Weight', fontsize=12)
        ax2.set_title(f'{title} - Bar Chart', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Rotate x-axis labels if needed
        if len(assets) > 5:
            ax2.set_xticklabels(assets, rotation=45, ha='right')
        
        ax2.grid(True, axis='y', alpha=self.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_performance_dashboard(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create an interactive dashboard using Plotly.
        
        Interactive visualizations allow users to explore data dynamically,
        zooming into specific periods, hovering for exact values, and toggling
        different metrics on and off. This creates a more engaging and insightful
        analysis experience.
        
        Args:
            portfolio_data: DataFrame with columns for different metrics over time
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Plotly Figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Value', 'Drawdown Analysis', 'Rolling Metrics'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value plot
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data['value'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1.5, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Drawdown plot
        if 'drawdown' in portfolio_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Rolling volatility
        if 'rolling_volatility' in portfolio_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data['rolling_volatility'] * 100,
                    mode='lines',
                    name='Volatility (90d)',
                    line=dict(color='orange', width=1.5),
                    hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Portfolio Performance Dashboard',
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Volatility %", row=3, col=1)
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            row=3, col=1
        )
        
        return fig
    
    def plot_risk_return_scatter(
        self,
        assets_data: Dict[str, Dict[str, float]],
        highlight_assets: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a risk-return scatter plot for multiple assets.
        
        This visualization helps investors compare different assets or strategies
        on the two dimensions that matter most: risk and return. The visual
        clustering of assets reveals natural groupings and outliers.
        
        Args:
            assets_data: Dict mapping asset names to dicts with 'return' and 'volatility'
            highlight_assets: Optional list of assets to highlight
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract data
        assets = list(assets_data.keys())
        returns = [assets_data[asset]['return'] for asset in assets]
        volatilities = [assets_data[asset]['volatility'] for asset in assets]
        
        # Calculate Sharpe ratios (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratios = [(r - risk_free_rate) / v if v > 0 else 0 
                        for r, v in zip(returns, volatilities)]
        
        # Create scatter plot
        scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, 
                           cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black')
        
        # Add asset labels
        for i, asset in enumerate(assets):
            if highlight_assets and asset in highlight_assets:
                fontweight = 'bold'
                fontsize = 12
            else:
                fontweight = 'normal'
                fontsize = 10
                
            ax.annotate(asset, (volatilities[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=fontsize, fontweight=fontweight)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Add diagonal lines for constant Sharpe ratios
        max_vol = max(volatilities) * 1.1
        for sharpe in [0.5, 1.0, 1.5]:
            x = np.linspace(0, max_vol, 100)
            y = risk_free_rate + sharpe * x
            ax.plot(x, y, '--', alpha=0.3, color='gray')
            ax.text(max_vol * 0.95, risk_free_rate + sharpe * max_vol * 0.95, 
                   f'SR={sharpe}', fontsize=9, alpha=0.5, ha='right')
        
        # Styling
        ax.set_xlabel('Annual Volatility (Risk)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Return', fontsize=14, fontweight='bold')
        ax.set_title('Risk-Return Profile of Assets', fontsize=16, fontweight='bold', pad=20)
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Set axis limits with some padding
        ax.set_xlim(0, max(volatilities) * 1.15)
        ax.set_ylim(min(returns) * 0.9 if min(returns) < 0 else 0, max(returns) * 1.1)
        
        ax.grid(True, alpha=self.grid_alpha)
        
        # Add quadrant labels
        mid_vol = max(volatilities) * 0.5
        mid_ret = (max(returns) + min(returns)) * 0.5
        
        ax.text(mid_vol * 0.5, mid_ret * 1.5, 'Low Risk\nHigh Return', 
               fontsize=11, alpha=0.3, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.1))
        
        ax.text(mid_vol * 1.5, mid_ret * 1.5, 'High Risk\nHigh Return', 
               fontsize=11, alpha=0.3, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig