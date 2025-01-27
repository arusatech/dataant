import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
from dataant.model import ModelTrainer
import numpy as np
from shinywidgets import render_plotly
from threading import Timer
from shiny import Inputs, Outputs, Session, App, ui, render, reactive, req
import socket
import webbrowser
from jsonpath_nz import log, jprint
from faicons import icon_svg
import plotnine as plt
from plotnine import (
    ggplot, aes, geom_line, geom_point, theme_minimal,
    labs, theme, element_rect, element_line, element_text,
    scale_x_discrete, geom_abline, geom_path  # Add geom_path here
)
from .ui_plots import (
    plot_roc_curve, plot_precision_recall_curve, plot_score_distribution,
    plot_production_score_distribution, plot_api_response
)

# UI Definition
def find_free_port(start_port=50000, end_port=60000):
    """Find a free port between start_port and end_port"""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                s.listen(1)
                return port
        except OSError:
            continue
    raise IOError(f"No free ports available in range {start_port}-{end_port}")

def open_browser(port):
    webbrowser.open(f'http://localhost:{port}')
    
def start_DataAnt(appDict, df):
    '''Create UI'''
    log.info(f"Starting DataAnt with appDict: {appDict} - df: {len(df)}")
    model_trainer = ModelTrainer()
       
    
    plot_tab = ui.nav_panel(
        "Plot ",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header(f"{appDict['info']}"),
                    ui.output_plot("plot_metric"),
                    ui.input_select( 
                        "selectField",
                        "Select Field",
                        choices=[str(f) for f in appDict['fields']],
                    ),
                ),
                fills=True,
                width="300px",
                height="1000px"
            ),
        ),
    )
    data_tab = ui.nav_panel(
        "Data ",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header(f"{appDict['info']}"),
                    ui.output_data_frame("summary_statistics"),  # Added DataGrid output here
                ),
                fills=True,
                width="250px"
            )
        )
    )
    training_tab = ui.nav_panel(
        "Training ",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Model Metrics"),
                    ui.output_plot("training_metric"),
                    ui.input_select(
                        "metric",
                        "Metric",
                        choices=["ROC Curve", "Precision-Recall"],
                    ),
                ),
                ui.card(
                    ui.card_header("Training Scores"),
                    ui.output_plot("score_dist"),
                ),
                fills=True,
                width="250px"
            ),
        ),
    )
    monitoring_tab = ui.nav_panel(
        "Model Monitoring",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Model Training Time History"),
                    ui.output_plot("api_response"),
                ),
                ui.card(
                    ui.card_header("Production Scores"),
                    ui.output_plot("prod_score_dist"),
                ),
                fills=True,
                width="250px"
            ),
        ),
    )
    annotation_tab = ui.nav_panel(
        "Data Annotation",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Input Fields"),
                    ui.card(ui.output_ui("review_card")),
                    ui.card_footer(
                        ui.layout_column_wrap(
                            1 / 2,
                            ui.input_action_button(
                                "reset_form",
                                "Reset",
                                class_="btn btn-secondary"
                            ),
                            ui.input_action_button(
                                "submit_review",
                                "Predict",
                                class_="btn btn-primary"
                            ),
                            style="margin-bottom:0",
                            width="100%"
                        ),
                    ),
                ),
                ui.card(
                    ui.card_header("Prediction Results"), 
                    ui.output_data_frame("results")
                ),
                fills=True,
                width="400px"  # Increased width for better readability
            )
        ),
    )
    
    app_ui = ui.page_navbar(
        # ui.panel_title("Data Ant"),
        plot_tab,
        data_tab,
        training_tab,
        monitoring_tab,
        annotation_tab,
        sidebar=ui.sidebar(
                ui.h3("Filters"),
                ui.input_slider(
                    "amount_range",
                    f"{appDict['slider_name']} Range",
                    min=appDict['slider_min'],
                    max=appDict['slider_max'],
                    value=[appDict['slider_max']/2, appDict['slider_max']],
                    drag_range=True
                ),
                ui.output_text("count"),
                ui.output_text("range_value"),
                ui.output_text("validation_message"),
                ui.input_numeric(
                    "min_input",
                    "Minimum Amount:",
                    value=appDict['slider_max']/2
                ),
                ui.input_numeric(
                    "max_input",
                    "Maximum Amount:",
                    value=appDict['slider_max']
                ),
                 ui.input_action_button(
                    "submit_range",  # ID for the submit button
                    "Apply Range",   # Button text
                    class_="btn-target"  # Bootstrap styling
                ),
                width=300,
                bg="#f8f8f8"
            ),
        id="tabs",
        title=[
            icon_svg("microchip", width='25px', height='25px'),
            ' ',
            icon_svg("d",width='20px', height='20px'),
            icon_svg("a",width='20px', height='20px'),
            icon_svg("t",width='20px', height='20px'),
            icon_svg("a",width='20px', height='20px'),
            icon_svg("plus",width='20px', height='20px'),
            icon_svg("a",width='20px', height='20px'),
            icon_svg("n",width='20px', height='20px'),
            icon_svg("t",width='20px', height='20px'),
            " (ai) "
        ],
        footer=ui.card_footer("Yakub Mohammad <yakub@arusatech.com> | AR USA LLC")
    )

    
    

    def server(input: Inputs, output: Outputs, session: Session):
        validation = reactive.Value("")
        
        @reactive.Calc
        def get_filtered_df():
            min_val, max_val = input.amount_range()
            return df[
                (df[f'{appDict["slider_name"]}'] >= min_val) & 
                (df[f'{appDict["slider_name"]}'] <= max_val)
            ]

        @reactive.Calc
        def get_trained_model():
            """Get or train the model with current filtered data"""
            try:
                filtered_df = get_filtered_df()
                
                # Get numeric feature columns
                feature_columns = [
                    f for f in appDict['fields'] 
                    if f != appDict['target'] and filtered_df[f].dtype != 'object'
                ]
                
                # Drop rows with missing values
                filtered_df = filtered_df.dropna(subset=feature_columns + [appDict['target']])
                
                X = filtered_df[feature_columns]
                y = filtered_df[appDict['target']]
                
                # Train model (will reuse if already trained with same data)
                metrics = model_trainer.train_model(X, y)
                return metrics
                
            except Exception as e:
                log.error(f"Error in get_trained_model: {str(e)}")
                raise

        @reactive.Effect
        @reactive.event(input.submit_range)
        def _():
            try:
                min_val = float(input.min_input())
                max_val = float(input.max_input())
                
                if min_val > max_val:
                    validation.set("Minimum amount cannot be greater than maximum amount")
                elif min_val < appDict['slider_min'] or max_val > appDict['slider_max']:
                    validation.set(f"Values must be between {appDict['slider_min']} and {appDict['slider_max']}")
                else:
                    validation.set("")
                    ui.update_slider("amount_range", value=[min_val, max_val])
            except (ValueError, TypeError):
                validation.set("Please enter valid numbers")

        @reactive.Effect
        @reactive.event(input.amount_range)
        def _():
            min_val, max_val = input.amount_range()
            ui.update_numeric("min_input", value=min_val)
            ui.update_numeric("max_input", value=max_val)

        @output
        @render.text
        def validation_message():
            return validation.get()

        @output
        @render.text
        @reactive.event(input.amount_range)
        def count():
            min_val, max_val = input.amount_range()
            filtered_df = df[
                (df[f'{appDict["slider_name"]}'] >= min_val) & 
                (df[f'{appDict["slider_name"]}'] <= max_val)
            ]
            return f"Count: {len(filtered_df)} / {len(df)}"

        @output
        @render.plot
        @reactive.event(input.selectField, input.amount_range)
        def plot_metric():
            """Create plot with selected field against target key using plotnine"""
            try:
                index = appDict['target']
                # Get filtered data
                filtered_df = get_filtered_df()
                
                # Get selected field
                selected_field = input.selectField()
                
                #For a given selected field, filter all the invalid fields except target key and drop the row that has NaN or None or empty values
                
                # Filter all invalid fields except target key and selected field
                invalid_fields = [
                    f for f in appDict['fields'] 
                    if f != selected_field and f != appDict['target']
                ]
                filtered_df = filtered_df.drop(columns=invalid_fields)
                filtered_df = filtered_df.dropna()
                # Additional cleaning for empty strings or whitespace
                if filtered_df[selected_field].dtype == object:  # For string columns
                    filtered_df = filtered_df[
                        filtered_df[selected_field].str.strip().astype(bool)
                    ]
                
                log.info(f"Cleaned data shape: {filtered_df.shape}")
                log.info(f"Remaining columns: {filtered_df.columns.tolist()}")
                
                
                # Reset index and prepare data for plotting
                plot_df = filtered_df.reset_index()
                
                # Ensure column names are strings
                plot_df.columns = plot_df.columns.astype(str)
                
               # Get value counts for the selected field
                value_counts = plot_df[selected_field].value_counts()
                
                # Create labels with counts
                x_labels = [f"{val} ({count})" for val, count in value_counts.items()]
                
                # Convert selected field to string type if it's not already
                plot_df[selected_field] = plot_df[selected_field].astype(str)
                
                # Sort the data to ensure consistent ordering
                plot_df = plot_df.sort_values(by=selected_field)
                 # Check if the field is numeric and has more than 100 unique values
                if (pd.api.types.is_numeric_dtype(plot_df[selected_field]) and 
                    len(plot_df[selected_field].unique()) > 100):
                    
                    # Create bins using round numbers
                    min_val = plot_df[selected_field].min()
                    max_val = plot_df[selected_field].max()
                    
                    # Create bin edges with step of 1
                    bin_edges = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
                    
                    # Create bin labels
                    bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" 
                                for i in range(len(bin_edges)-1)]
                    
                    # Add binned column
                    plot_df['binned'] = pd.cut(plot_df[selected_field], 
                                             bins=bin_edges, 
                                             labels=bin_labels, 
                                             include_lowest=True)
                    
                    # Get value counts for binned data
                    value_counts = plot_df['binned'].value_counts().sort_index()
                    
                    # Create labels with counts
                    x_labels = [f"{val} ({count})" for val, count in value_counts.items()]
                    
                    # Use binned column for plotting
                    plot_field = 'binned'
                else:
                    # Use original field if not numeric or less than 100 unique values
                    value_counts = plot_df[selected_field].value_counts().sort_index()
                    x_labels = [f"{val} ({count})" for val, count in value_counts.items()]
                    plot_field = selected_field

                plot = (
                    ggplot(data=plot_df) +
                    aes(x=str(plot_field), y=str(index)) +
                    geom_line(color='blue', alpha=0.7) +
                    geom_point(color='blue', alpha=0.5, size=2) +
                    theme_minimal() +
                    labs(
                        title=f"{selected_field}",
                        x=str(selected_field),
                        y=str(index)
                    ) +
                    theme(
                        figure_size=(10, 6),
                        plot_background=element_rect(fill='white'),
                        panel_grid_major=element_line(color='lightgray'),
                        panel_grid_minor=element_line(color='lightgray'),
                        axis_text_x=element_text(angle=45, hjust=1)
                    ) +
                    scale_x_discrete(limits=value_counts.index.tolist(), labels=x_labels)
                )
                
                return plot
                
            except Exception as e:
                log.error(f"Error creating plot: {str(e)}")
                # return plt.figure()  #
                return ggplot()  # Return empty plot on error

        @output
        @render.plot
        @reactive.event(input.metric)
        def training_metric():
            """Create ROC or Precision-Recall curve using plotnine"""
            try:
                metrics = get_trained_model()
                if input.metric() == "ROC Curve":
                    log.info("ROC Curve selected")
                    return plot_roc_curve(metrics)
                else:
                    log.info("Precision-Recall Curve selected")
                    return plot_precision_recall_curve(metrics)
                    
            except Exception as e:
                log.error(f"Error creating metric plot: {str(e)}")
                return ggplot()  # Return empty plot on error
                    
        @output
        @render.plot
        @reactive.event(input.amount_range)
        def score_dist():
            """Display the distribution of model scores for the filtered dataset"""
            try:
                metrics = get_trained_model()
                # filtered_df = get_filtered_df()
                # log.info(f"Creating score distribution plot for {len(filtered_df)} entries")
                
                # # Get numeric feature columns
                # feature_columns = [
                #     f for f in appDict['fields'] 
                #     if f != appDict['target'] and filtered_df[f].dtype != 'object'
                # ]
                
                # # Drop rows with missing values
                # filtered_df = filtered_df.dropna(subset=feature_columns + [appDict['target']])
                
                # X = filtered_df[feature_columns]
                # y = filtered_df[appDict['target']]
                
                # # Initialize and train model
                # model_trainer = ModelTrainer()
                # metrics = model_trainer.train_model(X, y)
                
                return plot_score_distribution(metrics)
                    
            except Exception as e:
                log.error(f"Error creating score distribution plot: {str(e)}")
                return ggplot()
        
        @output
        @render.text
        @reactive.event(input.amount_range)
        def range_value():
            min_val, max_val = input.amount_range()
            return f"Selected Range: {min_val:,.2f} - {max_val:,.2f}"
        
        @output
        @render.text
        @reactive.event(input.amount_range)
        def record_count():
            filtered_df = get_filtered_df()
            total = len(df)
            filtered = len(filtered_df)
            return f"{filtered:,} / {total:,}"

        
        @output
        @render.data_frame
        def summary_statistics():
            cols = [f for f in appDict['fields']]
            return render.DataGrid(get_filtered_df()[cols], filters=True,height="800px", width="100%")
        
        @output
        @render.text
        @reactive.event(input.amount_range)
        def mean_value():
            filtered_df = get_filtered_df()
            mean = filtered_df[f'{appDict["slider_name"]}'].mean()
            return f"${mean:,.2f}"

        @output
        @render.text
        @reactive.event(input.amount_range)
        def std_dev():
            filtered_df = get_filtered_df()
            std = filtered_df[f'{appDict["slider_name"]}'].std()
            return f"${std:,.2f}"
        
        @output
        @render.plot
        def prod_score_dist():
            """Display the distribution of model scores in production"""
            try:
                metrics = get_trained_model()  # Reuse trained model
                filtered_df = get_filtered_df()
                
                # Get numeric feature columns
                feature_columns = [
                    f for f in appDict['fields'] 
                    if f != appDict['target'] and filtered_df[f].dtype != 'object'
                ]
                
                # Drop rows with missing values
                filtered_df = filtered_df.dropna(subset=feature_columns + [appDict['target']])
                
                X = filtered_df[feature_columns]
                prod_scores = model_trainer.predict(X)
                
                return plot_production_score_distribution(prod_scores)
               
                
            except Exception as e:
                log.error(f"Error creating production score distribution plot: {str(e)}")
                return ggplot()
        
        @output
        @render.plot
        def api_response():
            """Display API response time trend"""
            try:
                # Get training history from model trainer
                history = model_trainer.training_history
                if not history:
                    log.info("No training history available yet")
                    return ggplot()  # Return empty plot if no history
               
                return plot_api_response(history)
                
            except Exception as e:
                log.error(f"Error creating API response plot: {str(e)}")
                return ggplot()

        @output
        @render.ui
        def review_card():
            """Create dynamic input fields for annotation"""
            try:
                # Get feature columns (excluding target field)
                feature_columns = [
                    f for f in appDict['fields'] 
                    if f != appDict['target']
                ]
                
                # Create input fields based on data types
                input_fields = []
                filtered_df = get_filtered_df()
                
                for col in feature_columns:
                    if filtered_df[col].dtype in ['int64', 'float64']:
                        # Numeric input for numeric columns
                        mean_val = filtered_df[col].mean()
                        input_fields.append(
                            ui.input_numeric(
                                f"input_{col}",
                                label=col,
                                value=round(mean_val, 2)
                            )
                        )
                    elif filtered_df[col].dtype == 'object':
                        # Dropdown for categorical columns
                        unique_vals = filtered_df[col].unique().tolist()
                        input_fields.append(
                            ui.input_select(
                                f"input_{col}",
                                label=col,
                                choices=unique_vals,
                                selected=unique_vals[0] if unique_vals else None
                            )
                        )
                    
                return ui.div(
                    ui.h4("Enter values for prediction:"),
                    *input_fields,
                    class_="well"
                )
                
            except Exception as e:
                log.error(f"Error creating review card: {str(e)}")
                return ui.p("Error creating input form")

        @output
        @render.data_frame
        @reactive.event(input.submit_review)
        def results():
            """Generate prediction results from input values"""
            try:
                # Get feature columns
                feature_columns = [
                    f for f in appDict['fields'] 
                    if f != appDict['target']
                ]
                
                # Collect input values into a dictionary
                input_values = {}
                for col in feature_columns:
                    input_val = input[f"input_{col}"]()
                    if input_val is not None:
                        input_values[col] = input_val
                
                if not input_values:
                    return pd.DataFrame()
                
                # Create a single-row DataFrame from input
                input_df = pd.DataFrame([input_values])
                
                # Get numeric feature columns for model
                numeric_features = [
                    f for f in feature_columns 
                    if input_df[f].dtype != 'object'
                ]
                
                # Ensure model is trained
                metrics = get_trained_model()
                
                # Get prediction
                prediction_score = model_trainer.predict(input_df[numeric_features])
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Field': list(input_values.keys()) + ['Prediction Score'],
                    'Value': list(input_values.values()) +  [f"{prediction_score[0]:.3f}"]
                })
                return results_df
                
            except Exception as e:
                log.error(f"Error generating results: {str(e)}")
                return pd.DataFrame({'Error': [str(e)]})

    app = App(app_ui, server)
    port = find_free_port()
    Timer(1.5, lambda: open_browser(port)).start()
    app.run(port=port)

                                    
