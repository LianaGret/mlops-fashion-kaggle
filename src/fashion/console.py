from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "highlight": "bold magenta",
        "muted": "dim white",
    }
)

console = Console(theme=custom_theme)


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def print_header(title: str, subtitle: str | None = None) -> None:
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[muted]{subtitle}[/muted]"
    console.print(Panel(content, border_style="cyan", padding=(0, 2)))


def print_info(message: str) -> None:
    console.print(f"[info]>[/info] {message}")


def print_success(message: str) -> None:
    console.print(f"[success]>[/success] {message}")


def print_warning(message: str) -> None:
    console.print(f"[warning]![/warning] {message}")


def print_error(message: str) -> None:
    console.print(f"[error]x[/error] {message}")


def print_key_value(key: str, value: str, indent: int = 0) -> None:
    prefix = "  " * indent
    console.print(f"{prefix}[muted]{key}:[/muted] {value}")


def create_table(title: str, columns: list[str]) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    return table
