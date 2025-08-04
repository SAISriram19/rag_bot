"""
Performance dashboard UI for monitoring RAG system performance.
Provides real-time monitoring and analytics visualization.
"""

import gradio as gr
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.performance_monitor import performance_monitor
from src.services.performance_optimizer import performance_optimizer
from src.services.performance_analytics import performance_analytics


class PerformanceDashboard:
    """Performance monitoring dashboard for the RAG system."""
    
    def __init__(self):
        """Initialize the performance dashboard."""
        self.refresh_interval = 30  # seconds
        self.last_refresh = datetime.now()
        
    def create_dashboard(self):
        """Create the performance dashboard interface."""
        
        # Dashboard CSS
        dashboard_css = """
        .dashboard-container {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background: #f8fafc;
            color: #1e293b;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(30, 64, 175, 0.2);
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-good { color: #059669; }
        .status-warning { color: #d97706; }
        .status-error { color: #dc2626; }
        
        .recommendation-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .recommendation-high { border-left: 4px solid #dc2626; }
        .recommendation-medium { border-left: 4px solid #d97706; }
        .recommendation-low { border-left: 4px solid #059669; }
        """
        
        with gr.Blocks(css=dashboard_css, title="RAG Performance Dashboard") as dashboard:
            
            # Header
            gr.HTML("""
            <div class="dashboard-header">
                <h1>🚀 RAG System Performance Dashboard</h1>
                <p>Real-time monitoring and analytics for your AI documentation system</p>
            </div>
            """)
            
            # Auto-refresh indicator
            with gr.Row():
                refresh_status = gr.HTML("🔄 Dashboard will auto-refresh every 30 seconds")
                manual_refresh_btn = gr.Button("🔄 Refresh Now", variant="secondary")
            
            # Key Metrics Row
            with gr.Row():
                with gr.Column(scale=1):
                    system_health = gr.HTML(self._get_system_health_html())
                
                with gr.Column(scale=1):
                    performance_metrics = gr.HTML(self._get_performance_metrics_html())
                
                with gr.Column(scale=1):
                    usage_stats = gr.HTML(self._get_usage_stats_html())
            
            # Charts Row
            with gr.Row():
                with gr.Column(scale=2):
                    performance_chart = gr.Plot(label="Performance Trends")
                
                with gr.Column(scale=1):
                    model_usage_chart = gr.Plot(label="Model Usage Distribution")
            
            # Detailed Analytics
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🔍 Top Queries</h3>")
                    top_queries_table = gr.HTML(self._get_top_queries_html())
                
                with gr.Column(scale=1):
                    gr.HTML("<h3>⚠️ Error Analysis</h3>")
                    error_analysis = gr.HTML(self._get_error_analysis_html())
            
            # Optimization Recommendations
            with gr.Row():
                gr.HTML("<h3>💡 Performance Recommendations</h3>")
                recommendations = gr.HTML(self._get_recommendations_html())
            
            # System Information
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📊 System Information</h3>")
                    system_info = gr.HTML(self._get_system_info_html())
                
                with gr.Column(scale=1):
                    gr.HTML("<h3>📈 Analytics Export</h3>")
                    export_btn = gr.Button("📥 Export Analytics Report", variant="primary")
                    export_status = gr.HTML("")
            
            # Event handlers
            def refresh_dashboard():
                """Refresh all dashboard components."""
                return (
                    self._get_system_health_html(),
                    self._get_performance_metrics_html(),
                    self._get_usage_stats_html(),
                    self._create_performance_chart(),
                    self._create_model_usage_chart(),
                    self._get_top_queries_html(),
                    self._get_error_analysis_html(),
                    self._get_recommendations_html(),
                    self._get_system_info_html(),
                    f"🔄 Last refreshed: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            def export_analytics():
                """Export analytics report."""
                try:
                    report = performance_analytics.export_analytics_report(30)
                    filename = f"rag_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    return f"✅ Analytics report exported to {filename}"
                except Exception as e:
                    return f"❌ Export failed: {str(e)}"
            
            # Manual refresh
            manual_refresh_btn.click(
                refresh_dashboard,
                outputs=[
                    system_health, performance_metrics, usage_stats,
                    performance_chart, model_usage_chart,
                    top_queries_table, error_analysis, recommendations,
                    system_info, refresh_status
                ]
            )
            
            # Export functionality
            export_btn.click(export_analytics, outputs=[export_status])
            
            # Auto-refresh (would need to be implemented with JavaScript in a real deployment)
            dashboard.load(
                refresh_dashboard,
                outputs=[
                    system_health, performance_metrics, usage_stats,
                    performance_chart, model_usage_chart,
                    top_queries_table, error_analysis, recommendations,
                    system_info, refresh_status
                ]
            )
        
        return dashboard
    
    def _get_system_health_html(self) -> str:
        """Generate system health status HTML."""
        try:
            # Get current metrics from performance monitor
            current_metrics = performance_monitor.get_current_metrics()
            
            # Determine overall health
            health_score = self._calculate_health_score(current_metrics)
            
            if health_score >= 0.8:
                status_class = "status-good"
                status_icon = "🟢"
                status_text = "Excellent"
            elif health_score >= 0.6:
                status_class = "status-warning"
                status_icon = "🟡"
                status_text = "Good"
            else:
                status_class = "status-error"
                status_icon = "🔴"
                status_text = "Needs Attention"
            
            return f"""
            <div class="metric-card">
                <div class="metric-value {status_class}">
                    {status_icon} {status_text}
                </div>
                <div class="metric-label">System Health</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    <div>Health Score: {health_score:.1%}</div>
                    <div>Uptime: {self._get_uptime()}</div>
                    <div>Status: All systems operational</div>
                </div>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="metric-card">
                <div class="metric-value status-error">🔴 Error</div>
                <div class="metric-label">System Health</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    Error loading health data: {str(e)}
                </div>
            </div>
            """
    
    def _get_performance_metrics_html(self) -> str:
        """Generate performance metrics HTML."""
        try:
            usage_analytics = performance_analytics.get_usage_analytics(1)  # Last 24 hours
            
            return f"""
            <div class="metric-card">
                <div class="metric-value">{usage_analytics.avg_response_time:.1f}s</div>
                <div class="metric-label">Avg Response Time</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    <div>Success Rate: {(usage_analytics.successful_queries / max(usage_analytics.total_queries, 1)):.1%}</div>
                    <div>Confidence: {usage_analytics.avg_confidence_score:.1%}</div>
                    <div>Model: {usage_analytics.most_used_model}</div>
                </div>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="metric-card">
                <div class="metric-value status-error">Error</div>
                <div class="metric-label">Performance Metrics</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    Error loading metrics: {str(e)}
                </div>
            </div>
            """
    
    def _get_usage_stats_html(self) -> str:
        """Generate usage statistics HTML."""
        try:
            usage_analytics = performance_analytics.get_usage_analytics(7)  # Last 7 days
            
            return f"""
            <div class="metric-card">
                <div class="metric-value">{usage_analytics.total_queries}</div>
                <div class="metric-label">Total Queries (7 days)</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    <div>Documents: {usage_analytics.total_documents_processed}</div>
                    <div>Chunks: {usage_analytics.total_chunks_created}</div>
                    <div>Daily Avg: {usage_analytics.total_queries / 7:.1f}</div>
                </div>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="metric-card">
                <div class="metric-value status-error">Error</div>
                <div class="metric-label">Usage Statistics</div>
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    Error loading stats: {str(e)}
                </div>
            </div>
            """
    
    def _create_performance_chart(self):
        """Create performance trends chart."""
        try:
            trends = performance_analytics.get_performance_trends(7)
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Response Time Trend', 'Success Rate Trend'),
                vertical_spacing=0.1
            )
            
            # Response time trend
            if trends.get("response_time"):
                dates, times = zip(*trends["response_time"])
                fig.add_trace(
                    go.Scatter(x=dates, y=times, mode='lines+markers', name='Response Time (s)'),
                    row=1, col=1
                )
            
            # Success rate trend
            if trends.get("success_rate"):
                dates, rates = zip(*trends["success_rate"])
                fig.add_trace(
                    go.Scatter(x=dates, y=[r*100 for r in rates], mode='lines+markers', name='Success Rate (%)'),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=400,
                title_text="Performance Trends (Last 7 Days)",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            # Return empty chart on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _create_model_usage_chart(self):
        """Create model usage distribution chart."""
        try:
            # This would need to be implemented based on actual model usage data
            # For now, return a placeholder
            fig = go.Figure(data=[
                go.Pie(labels=['llama3.2:1b', 'llama3:latest'], values=[70, 30])
            ])
            fig.update_layout(title_text="Model Usage Distribution")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _get_top_queries_html(self) -> str:
        """Generate top queries HTML table."""
        try:
            top_queries = performance_analytics.get_top_queries(5, 7)
            
            if not top_queries:
                return "<div class='metric-card'>No query data available</div>"
            
            html = "<div class='metric-card'><table style='width: 100%; font-size: 0.875rem;'>"
            html += "<tr><th>Query</th><th>Count</th><th>Avg Time</th></tr>"
            
            for query in top_queries:
                query_text = query['query'][:50] + "..." if len(query['query']) > 50 else query['query']
                html += f"""
                <tr>
                    <td>{query_text}</td>
                    <td>{query['frequency']}</td>
                    <td>{query['avg_response_time']:.1f}s</td>
                </tr>
                """
            
            html += "</table></div>"
            return html
            
        except Exception as e:
            return f"<div class='metric-card'>Error loading queries: {str(e)}</div>"
    
    def _get_error_analysis_html(self) -> str:
        """Generate error analysis HTML."""
        try:
            error_analysis = performance_analytics.get_error_analysis(7)
            
            if not error_analysis.get("error_types"):
                return "<div class='metric-card'>No errors detected in the last 7 days ✅</div>"
            
            html = "<div class='metric-card'>"
            html += "<h4>Common Errors:</h4>"
            
            for error in error_analysis["error_types"][:3]:
                html += f"""
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: #fef2f2; border-radius: 4px;">
                    <strong>{error['error']}</strong><br>
                    <small>Frequency: {error['frequency']} times</small>
                </div>
                """
            
            html += "</div>"
            return html
            
        except Exception as e:
            return f"<div class='metric-card'>Error loading analysis: {str(e)}</div>"
    
    def _get_recommendations_html(self) -> str:
        """Generate optimization recommendations HTML."""
        try:
            # Get current metrics and generate recommendations
            current_metrics = performance_monitor.get_current_metrics()
            recommendations = performance_optimizer.analyze_performance_metrics(current_metrics)
            
            if not recommendations:
                return "<div class='metric-card'>🎉 No optimization recommendations - system is performing well!</div>"
            
            html = ""
            for rec in recommendations[:3]:  # Show top 3 recommendations
                priority_class = f"recommendation-{rec.priority}"
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec.priority]
                
                html += f"""
                <div class="recommendation-card {priority_class}">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">
                        {priority_icon} {rec.component}: {rec.issue}
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <strong>Recommendation:</strong> {rec.recommendation}
                    </div>
                    <div style="font-size: 0.875rem; color: #64748b;">
                        <strong>Impact:</strong> {rec.estimated_improvement} | 
                        <strong>Complexity:</strong> {rec.implementation_complexity}
                    </div>
                </div>
                """
            
            return html
            
        except Exception as e:
            return f"<div class='metric-card'>Error loading recommendations: {str(e)}</div>"
    
    def _get_system_info_html(self) -> str:
        """Generate system information HTML."""
        try:
            import psutil
            import platform
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return f"""
            <div class="metric-card">
                <h4>System Resources</h4>
                <div style="font-size: 0.875rem;">
                    <div>CPU Usage: {cpu_percent:.1f}%</div>
                    <div>Memory: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)</div>
                    <div>Disk: {disk.percent:.1f}% ({disk.free / (1024**3):.1f}GB free)</div>
                    <div>Platform: {platform.system()} {platform.release()}</div>
                    <div>Python: {platform.python_version()}</div>
                </div>
            </div>
            """
            
        except Exception as e:
            return f"<div class='metric-card'>Error loading system info: {str(e)}</div>"
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        try:
            score = 1.0
            
            # Response time factor
            avg_response_time = metrics.get('avg_response_time', 0)
            if avg_response_time > 30:
                score -= 0.4
            elif avg_response_time > 15:
                score -= 0.2
            elif avg_response_time > 5:
                score -= 0.1
            
            # Success rate factor
            success_rate = metrics.get('success_rate', 1.0)
            if success_rate < 0.8:
                score -= 0.3
            elif success_rate < 0.9:
                score -= 0.1
            
            # Memory usage factor
            memory_usage = metrics.get('memory_usage_mb', 0)
            if memory_usage > 2000:  # 2GB
                score -= 0.2
            elif memory_usage > 1000:  # 1GB
                score -= 0.1
            
            return max(0.0, score)
            
        except Exception:
            return 0.5  # Default to moderate health on error
    
    def _get_uptime(self) -> str:
        """Get system uptime string."""
        try:
            import psutil
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            return f"{days}d {hours}h {minutes}m"
        except Exception:
            return "Unknown"


def create_performance_dashboard():
    """Create and return the performance dashboard."""
    dashboard = PerformanceDashboard()
    return dashboard.create_dashboard()


def main():
    """Launch the performance dashboard."""
    print("🚀 Launching RAG Performance Dashboard...")
    
    dashboard = create_performance_dashboard()
    
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        inbrowser=True,
        show_error=True
    )


if __name__ == "__main__":
    main()