"""
Interactive Text-based User Interface for monitoring and controlling PDF processing.

This module provides a rich-based TUI that displays real-time progress, allows
operators to pause/resume processing, and shows detailed statistics.
"""

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Rich library not installed. Please run: pip install rich>=13.7")
    raise

from utilities.manifest import ProcessingManifest

logger = logging.getLogger(__name__)


class ProcessingTUI:
    """
    Interactive TUI for monitoring and controlling PDF processing pipeline.
    
    Features:
    - Real-time progress display
    - Pause/resume controls
    - Processing statistics
    - Error monitoring
    - ETA calculations
    """

    def __init__(self, manifest: ProcessingManifest, total_files: int = 0):
        self.manifest = manifest
        self.total_files = total_files
        self.console = Console()

        # Control events
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start in running state
        self.shutdown_event = asyncio.Event()

        # State tracking
        self.start_time = time.time()
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_time = 0.0

        # Statistics
        self.last_stats = {}
        self.processing_rates = []  # For ETA calculation

        # Display refresh interval
        self.refresh_interval = 0.5

    def get_current_stats(self) -> dict[str, Any]:
        """Get current processing statistics from manifest."""
        try:
            return self.manifest.get_summary()
        except Exception as e:
            logger.error(f"Error getting stats from manifest: {e}")
            return {
                "total_files": self.total_files,
                "classified": 0,
                "extracted": 0,
                "failed": 0,
                "pending_extraction": 0,
                "pending_classification": 0
            }

    def calculate_eta(self, stats: dict[str, Any]) -> str | None:
        """Calculate estimated time to completion."""
        try:
            current_time = time.time()
            elapsed_time = current_time - self.start_time - self.total_pause_time

            if elapsed_time < 30:  # Need some time to calculate meaningful rate
                return "Calculating..."

            # Count completed files (extracted files represent fully completed processing)
            completed_files = stats["extracted"]
            total_files = stats["total_files"]

            if completed_files == 0:
                return "Calculating..."

            # Calculate rate based on fully completed files
            rate = completed_files / elapsed_time
            remaining_files = total_files - completed_files

            if rate > 0:
                eta_seconds = remaining_files / rate
                eta = timedelta(seconds=int(eta_seconds))
                return str(eta)
            return "Unknown"

        except Exception as e:
            logger.debug(f"Error calculating ETA: {e}")
            return "Unknown"

    def create_status_panel(self, stats: dict[str, Any]) -> Panel:
        """Create the main status panel."""
        # Calculate progress percentages
        total_files = max(stats["total_files"], 1)
        classified_pct = (stats["classified"] / total_files) * 100
        extracted_pct = (stats["extracted"] / total_files) * 100

        # Create main table
        table = Table.grid(padding=1)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="magenta")

        # Status indicators
        status_color = "green" if not self.is_paused else "yellow"
        status_text = "🟢 RUNNING" if not self.is_paused else "🟡 PAUSED"

        table.add_row("Status:", f"[{status_color}]{status_text}[/{status_color}]")
        table.add_row("", "")

        # Progress rows
        table.add_row("Total Files:", f"{stats['total_files']:,}")
        table.add_row("Classified:", f"{stats['classified']:,} / {total_files:,} ({classified_pct:.1f}%)")
        table.add_row("Extracted:", f"{stats['extracted']:,} / {total_files:,} ({extracted_pct:.1f}%)")
        table.add_row("Failed:", f"[red]{stats['failed']:,}[/red]")
        table.add_row("", "")

        # Queue information
        table.add_row("Pending Classification:", f"{stats['pending_classification']:,}")
        table.add_row("Pending Extraction:", f"{stats['pending_extraction']:,}")
        table.add_row("", "")

        # Timing information
        current_time = time.time()
        elapsed_time = current_time - self.start_time - self.total_pause_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        eta = self.calculate_eta(stats)

        table.add_row("Elapsed Time:", elapsed_str)
        table.add_row("Estimated ETA:", eta or "Unknown")

        # Create panel with controls
        title = "[bold blue]📊 Processing Status[/bold blue]"
        subtitle = "[dim]Press [b]p[/b]=pause, [b]r[/b]=resume, [b]q[/b]=quit gracefully[/dim]"

        return Panel(
            table,
            title=title,
            subtitle=subtitle,
            border_style="blue",
            box=box.ROUNDED
        )

    def create_progress_bars(self, stats: dict[str, Any]) -> Panel:
        """Create progress bars panel."""
        total_files = max(stats["total_files"], 1)

        # Classification progress
        classify_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            expand=True,
        )

        classify_task = classify_progress.add_task(
            "Classification",
            total=total_files,
            completed=stats["classified"]
        )

        # Extraction progress
        extract_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            expand=True,
        )

        extract_task = extract_progress.add_task(
            "Extraction",
            total=total_files,
            completed=stats["extracted"]
        )

        # Combine progress bars
        progress_table = Table.grid()
        progress_table.add_row(classify_progress)
        progress_table.add_row(extract_progress)

        return Panel(
            progress_table,
            title="[bold green]📈 Progress Details[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )

    def create_layout(self, stats: dict[str, Any]):
        """Create the complete TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
        )

        layout["header"].update(
            Panel(
                "[bold cyan]📄 Invoice PDF Processing Pipeline[/bold cyan] - SQLite Manifest Resume Mode",
                style="cyan on blue",
                box=box.HEAVY
            )
        )

        layout["main"].split_row(
            Layout(self.create_status_panel(stats), name="status"),
            Layout(self.create_progress_bars(stats), name="progress")
        )

        return layout

    def toggle_pause(self):
        """Toggle pause/resume state."""
        current_time = time.time()

        if self.is_paused:
            # Resume
            self.is_paused = False
            self.pause_event.set()
            if self.pause_start_time:
                self.total_pause_time += current_time - self.pause_start_time
                self.pause_start_time = None
            logger.info("🔄 Processing resumed by operator")
        else:
            # Pause
            self.is_paused = True
            self.pause_event.clear()
            self.pause_start_time = current_time
            logger.info("⏸️  Processing paused by operator")

    def request_shutdown(self):
        """Request graceful shutdown."""
        self.shutdown_event.set()
        logger.info("🛑 Graceful shutdown requested by operator")

    async def handle_keyboard_input(self):
        """Handle keyboard input in a separate task."""
        import sys
        import termios
        import tty

        def getch():
            """Get a single character from stdin."""
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.cbreak(fd)
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
            except:
                return None

        def get_char_non_blocking():
            """Get character without blocking."""
            import select
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return getch()
            return None

        while not self.shutdown_event.is_set():
            try:
                char = await asyncio.to_thread(get_char_non_blocking)
                if char:
                    char = char.lower()
                    if (char == "p" and not self.is_paused) or (char == "r" and self.is_paused):
                        self.toggle_pause()
                    elif char == "q":
                        self.request_shutdown()
                        break

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.debug(f"Error in keyboard input handler: {e}")
                await asyncio.sleep(0.1)

    async def run(self):
        """Run the TUI display and input handling."""
        logger.info("🖥️  Starting interactive TUI...")

        # Start keyboard input handler
        input_task = asyncio.create_task(self.handle_keyboard_input())

        try:
            with Live(
                self.create_layout(self.get_current_stats()),
                console=self.console,
                refresh_per_second=2,  # 2 FPS for smooth updates
                screen=True
            ) as live:

                while not self.shutdown_event.is_set():
                    try:
                        # Get current stats
                        stats = self.get_current_stats()

                        # Update display
                        live.update(self.create_layout(stats))

                        # Check if processing is complete
                        if (stats["classified"] >= stats["total_files"] and
                            stats["extracted"] >= stats["total_files"] -
                            stats.get("irrelevant_count", 0)):
                            logger.info("✅ Processing complete!")
                            break

                        await asyncio.sleep(self.refresh_interval)

                    except Exception as e:
                        logger.error(f"Error in TUI display loop: {e}")
                        await asyncio.sleep(1)  # Longer delay on error

        except KeyboardInterrupt:
            logger.info("🛑 TUI interrupted by user")
        finally:
            # Clean up
            input_task.cancel()
            try:
                await input_task
            except asyncio.CancelledError:
                pass

            # Ensure we're not paused when exiting
            if self.is_paused:
                self.pause_event.set()

            logger.info("🖥️  TUI shut down")


async def run_tui_monitor(manifest: ProcessingManifest, total_files: int = 0, disable_tui: bool = False):
    """
    Run TUI monitoring for the processing pipeline.
    
    Args:
        manifest: The processing manifest to monitor
        total_files: Total number of files being processed
        disable_tui: If True, skip TUI launch for headless mode
        
    Returns:
        Tuple of (pause_event, shutdown_event) for use by processing pipeline
    """
    if disable_tui:
        logger.info("🖥️  TUI disabled for headless mode")
        # Return dummy events that are always set
        pause_event = asyncio.Event()
        pause_event.set()
        shutdown_event = asyncio.Event()  # Never set, so no shutdown
        return pause_event, shutdown_event

    try:
        tui = ProcessingTUI(manifest, total_files)

        # Start TUI in background
        tui_task = asyncio.create_task(tui.run())

        logger.info("🖥️  TUI started - use 'p' to pause, 'r' to resume, 'q' to quit")

        # Return events for pipeline coordination
        return tui.pause_event, tui.shutdown_event, tui_task

    except Exception as e:
        logger.error(f"Failed to start TUI: {e}")
        logger.info("🖥️  Continuing without TUI...")

        # Return dummy events
        pause_event = asyncio.Event()
        pause_event.set()
        shutdown_event = asyncio.Event()
        return pause_event, shutdown_event, None
