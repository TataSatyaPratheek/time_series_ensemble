"""
Command Line Interface for Time Series Ensemble Forecasting.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Confirm

from src.crew import quick_forecast, list_available_workflows
from src.api.endpoints import app
from src.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)
console = Console()

# Create Typer app
cli_app = typer.Typer(
    name="ts-ensemble",
    help="Time Series Forecasting Ensemble CLI",
    add_completion=False
)

@cli_app.command()
def forecast(
    data_file: str = typer.Argument(..., help="Path to CSV data file"),
    workflow: str = typer.Option("standard", help="Workflow type to use"),
    horizon: int = typer.Option(30, help="Forecast horizon"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    verbose: bool = typer.Option(False, help="Verbose output")
):
    """Generate time series forecast from CSV data."""
    
    if verbose:
        configure_logging("DEBUG")
    else:
        configure_logging("INFO")
    
    console.print(Panel.fit("🔮 Time Series Ensemble Forecasting", style="bold blue"))
    
    try:
        # Load data
        console.print(f"📊 Loading data from: {data_file}")
        data_path = Path(data_file)
        
        if not data_path.exists():
            console.print(f"❌ Error: Data file not found: {data_file}", style="bold red")
            raise typer.Exit(1)
        
        # Read CSV
        df = pd.read_csv(data_path)
        
        if df.empty:
            console.print("❌ Error: Data file is empty", style="bold red")
            raise typer.Exit(1)
        
        # Convert to series (assume first column is values)
        if len(df.columns) >= 2:
            # Try to parse datetime column
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                series = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0])
            except:
                series = pd.Series(df.iloc[:, 1].values)
        else:
            series = pd.Series(df.iloc[:, 0].values)
        
        console.print(f"✅ Loaded {len(series)} data points")
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating forecast...", total=None)
            
            # Run forecast
            results = asyncio.run(quick_forecast(
                series, 
                workflow_type=workflow,
                forecast_horizon=horizon
            ))
            
            progress.update(task, description="✅ Forecast completed!")
        
        # Display results
        if results.get('status') == 'completed':
            forecast_data = results.get('ensemble_forecast', [])
            
            # Create results table
            table = Table(title="Forecast Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Status", "✅ Success")
            table.add_row("Workflow", workflow)
            table.add_row("Horizon", str(horizon))
            table.add_row("Points Generated", str(len(forecast_data)))
            table.add_row("Execution Time", f"{results.get('execution_time', 0):.2f}s")
            
            console.print(table)
            
            # Save output if requested
            if output:
                output_path = Path(output)
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'forecast': forecast_data,
                    'period': range(1, len(forecast_data) + 1)
                })
                
                # Add confidence intervals if available
                ci = results.get('confidence_intervals', [])
                if ci:
                    forecast_df['lower_bound'] = [c[0] for c in ci]
                    forecast_df['upper_bound'] = [c[1] for c in ci]
                
                forecast_df.to_csv(output_path, index=False)
                console.print(f"💾 Results saved to: {output_path}")
        
        else:
            console.print("❌ Forecast failed", style="bold red")
            if 'error' in results:
                console.print(f"Error: {results['error']}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        raise typer.Exit(1)

@cli_app.command()
def workflows():
    """List available forecasting workflows."""
    console.print(Panel.fit("📋 Available Workflows", style="bold green"))
    
    try:
        available_workflows = list_available_workflows()
        
        table = Table()
        table.add_column("Workflow", style="cyan")
        table.add_column("Description", style="white")
        
        workflow_descriptions = {
            'standard': 'Balanced analysis with all components',
            'fast': 'Quick analysis for rapid results',
            'comprehensive': 'Deep analysis with extended timeouts'
        }
        
        for workflow in available_workflows:
            description = workflow_descriptions.get(workflow, 'Standard workflow')
            table.add_row(workflow, description)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        raise typer.Exit(1)

@cli_app.command()
def setup():
    """Setup Ollama models for time series forecasting."""
    from src.llm.model_manager import model_manager
    
    console.print(Panel.fit("🤖 Setting up Ollama Models", style="bold blue"))
    
    try:
        # Check Ollama availability
        if not model_manager.check_ollama_available():
            console.print("❌ Ollama not found. Please install Ollama first:", style="bold red")
            console.print("Visit: https://ollama.ai/download")
            raise typer.Exit(1)
        
        console.print("✅ Ollama is available")
        
        # Get recommended models
        recommended = model_manager.get_recommended_models()
        console.print(f"📋 Recommended models: {', '.join(recommended)}")
        
        # Download models
        for model in recommended[:2]:  # Download top 2 recommended
            if Confirm.ask(f"Download {model}?", default=True):
                console.print(f"📥 Downloading {model}...")
                success, message = model_manager.download_model(model)
                if success:
                    console.print(f"✅ {message}", style="bold green")
                else:
                    console.print(f"❌ {message}", style="bold red")
        
    except Exception as e:
        console.print(f"❌ Setup failed: {str(e)}", style="bold red")
        raise typer.Exit(1)

@cli_app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """Start the API server."""
    console.print(Panel.fit("🚀 Starting API Server", style="bold blue"))
    
    try:
        import uvicorn
        
        console.print(f"🌐 Server starting at: http://{host}:{port}")
        console.print(f"📖 API Docs: http://{host}:{port}/docs")
        console.print(f"🏥 Health Check: http://{host}:{port}/health")
        
        uvicorn.run(
            "src.api.endpoints:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except Exception as e:
        console.print(f"❌ Server failed to start: {str(e)}", style="bold red")
        raise typer.Exit(1)

@cli_app.command()
def version():
    """Show version information."""
    console.print(Panel.fit("ℹ️  Version Information", style="bold cyan"))
    
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="magenta")
    
    table.add_row("ts-ensemble", "0.1.0")
    table.add_row("Python", "3.10+")
    
    console.print(table)

def main():
    """Main CLI entry point."""
    try:
        cli_app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="bold yellow")
    except Exception as e:
        console.print(f"❌ Unexpected error: {str(e)}", style="bold red")
        raise typer.Exit(1)

if __name__ == "__main__":
    main()
