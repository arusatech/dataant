from shiny import Inputs, Outputs, Session, App, ui, render, reactive
from shinywidgets import output_widget, render_plotly  # Use output_widget for Plotly
import plotly.express as px
import pandas as pd
import signal
import sys
from faicons import icon_svg

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print('\n !!!You pressed Ctrl+C, Exiting ...... ')
    sys.exit(0)

# Sample DataFrame
df = pd.DataFrame({
    'x_column': [1, 2, 3, 4, 5],
    'y_column': [10, 20, 30, 40, 50],
    'size_column': [8, 12, 16, 20, 24]
})

def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render_plotly
    @reactive.event(input.selectField)  # React to changes in the dropdown
    def plot_metric():
        try:
            selected_field = input.selectField()
            plot_df = df

            # Debug prints
            print("Debug info:")
            print(f"Selected field: {selected_field}")
            print(f"DataFrame shape: {plot_df.shape}")
            print(f"DataFrame columns: {plot_df.columns.tolist()}")
            print(f"First few rows:\n{plot_df.head()}")

            # Create a Plotly histogram
            p = px.histogram(plot_df, x=selected_field)
            p.update_layout(
                height=400,  # Increased height for better visibility
                xaxis_title=selected_field,
                yaxis_title="Count",
                title=f"Histogram of {selected_field}"
            )
            return p
        except Exception as e:
            print(f"Error in plot_metric: {e}")
            return None

def plot_shiny():
    plot_tab = ui.nav_panel(
        "Plot",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("PLOT_HEADER"),
                    output_widget("plot_metric"),  # Use output_widget for Plotly plots
                    ui.input_select(
                        "selectField",
                        "Select Field",
                        choices=['x_column', 'y_column', 'size_column'],
                        selected='x_column'  # Default selected value
                    ),
                ),
                fills=True,
                width="300px",
                height="1000px",
                limitsize=False
            ),
        ),
    )

    app_ui = ui.page_navbar(
        plot_tab,
        id="tabs",
        title=[
            icon_svg("microchip", width='25px', height='25px'),
            ' ',
            icon_svg("d", width='20px', height='20px'),
            icon_svg("a", width='20px', height='20px'),
            icon_svg("t", width='20px', height='20px'),
            icon_svg("a", width='20px', height='20px'),
            icon_svg("plus", width='20px', height='20px'),
            icon_svg("a", width='20px', height='20px'),
            icon_svg("n", width='20px', height='20px'),
            icon_svg("t", width='20px', height='20px'),
            " (ai) "
        ],
        footer=ui.card_footer("Yakub Mohammad <yakub@arusatech.com> | AR USA LLC")
    )
    app = App(app_ui, server)
    app.run(port=5173)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    plot_shiny()