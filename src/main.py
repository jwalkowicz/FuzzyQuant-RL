import typer
from src.core.logger import logger
from src.data.indicators import get_spy_technical_data

app = typer.Typer(help="fuzzyQuant-RL")


@app.command(name="rum")
def run():
    logger.info("Starting the application..")
    data = get_spy_technical_data()

if __name__ == "__main__":
    app()
