import typer

from hello_clipseg.util.segment import app as segment_app


app = typer.Typer()

app.add_typer(segment_app, name="segment")
