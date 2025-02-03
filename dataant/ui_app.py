import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import socket
import psutil
import time
import webbrowser
import pandas as pd
import plotly.graph_objects as go
from shiny import Inputs, Outputs, Session, App, ui, render, reactive, req
from shinywidgets import render_plotly, output_widget
from dataant.model import ModelTrainer
from threading import Timer
from faicons import icon_svg
from .ui_plot import (
    plot_score_distribution, plot_production_score_distribution, plot_api_response, plot_default_metric,
    plot_model_metrics
)
from .model import ModelTrainer
from datetime import datetime
from jsonpath_nz import log

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        for proc in psutil.process_iter():
            try:
                # Get connections for each process
                connections = proc.net_connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        log.info(f"Killing process {proc.pid} using port {port}")
                        proc.terminate()
                        time.sleep(0.5)  # Give process time to terminate
                        if proc.is_running():
                            proc.kill()  # Force kill if still running
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        log.error(f"Error killing process on port {port}: {str(e)}")
        log.traceback(e)
    return False

def find_free_port(start_port=50000, end_port=60000, retry_count=5):
    """Find a free port between start_port and end_port with retries"""
    try:
        for attempt in range(retry_count):
            for port in range(start_port, end_port + 1):
                try:
                    # Try to bind to the port
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        s.bind(('127.0.0.1', port))
                        s.listen(1)
                        log.info(f"Found free port: {port}")
                        return port
                except OSError:
                    # Try to kill process if port is in use
                    kill_process_on_port(port)
                    continue
            time.sleep(1)  # Wait before retry
            log.info(f"Retrying port search, attempt {attempt + 1}/{retry_count}")
    except Exception as e:
        log.error(f"Error in find_free_port: {str(e)}")
        log.traceback(e)
        raise IOError(f"No free ports available in range {start_port}-{end_port} after {retry_count} attempts")


def open_browser(port):
    try:
        webbrowser.open(f'http://localhost:{port}')
    except Exception as e:
        log.error(f"Error in open_browser: {str(e)}")
        log.traceback(e)
    
def start_DataAnt(appDict, df):
    '''Create UI'''
    log.info(f"Starting DataAnt with appDict: {appDict} - df: {len(df)}")
    plot_tab = ui.nav_panel(
        "Plot ",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header(f"{appDict['info']}"),
                    output_widget("plot_metric"),
                    # ui.output_plot("plot_metric"),
                    ui.input_select( 
                        "selectField1",
                        "Select Field 1",
                        choices=[str(f) for f in appDict['fields']],
                    ),
                    ui.input_select( 
                        "selectField2",
                        "Select Field 2",
                        choices=[str(f) for f in appDict['fields']],
                    ),
                ),
                fills=True,
                width="300px",
                height="1000px",
                limitsize=False
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
                width="250px",
                limitsize=False
            )
        )
    )
    training_tab = ui.nav_panel(
        "Training ",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Model Metrics"),
                    output_widget("training_metric"),
                    # ui.output_plot("training_metric"),
                    ui.input_select(
                        "model",
                        "Model",
                        choices=["logistic", "svm", "random_forest", "xgboost"],
                        selected="logistic"
                    ),
                    ui.input_select(
                        "metric",
                        "Metric",
                        choices=["roc", "pr"],
                        selected="roc"
                    ),
                ),
                ui.card(
                    ui.card_header("Training Scores"),
                    output_widget("score_dist"),
                    # ui.output_plot("score_dist"),
                ),
                fills=True,
                width="250px",
                limitsize=False
            ),
        ),
    )
    monitoring_tab = ui.nav_panel(
        "Model Monitoring",
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Model Training Time History"),
                    output_widget("api_response"),
                    # ui.output_plot("api_response"),
                ),
                ui.card(
                    ui.card_header("Production Scores"),
                    output_widget("prod_score_dist"),
                    # ui.output_plot("prod_score_dist"),
                ),
                fills=True,
                width="250px",
                limitsize=False
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
                width="400px" , # Increased width for better readability
                limitsize=False
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
                bg="#f8f8f8",
                limitsize=False
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
            " (ai) \u2122 "
        ],
        footer=ui.card_footer("Design and Architect: Mr. Yakub Mohammad  \u00A9 AR USA LLC Team  | arusa@arusatech.com")
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
                return(X, y)
            except Exception as e:
                log.error(f"Error in get_trained_model: {str(e)}")
                log.traceback(e)
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
                
                X, y = get_trained_model()
                
                # Validate data
                if X is None or y is None:
                    log.error("X or y is None - check get_trained_model()")
                    return pd.DataFrame({'Error': ['Model data not available']})
                    
                model_trainer = ModelTrainer(
                    random_state=42, 
                    model_name=input.model(), 
                    test_size=0.2, 
                    X=X, 
                    y=y
                )
                
                # Train model if not already trained
                model_trainer.train_model()
                
                # Get prediction
                prediction_data = model_trainer.predict(input_df[numeric_features])
                
                # Format results
                results = []
                # Add input values
                for field, value in input_values.items():
                    results.append({
                        'Field': field,
                        'Value': value
                    })
                
                # Add prediction scores for each class
                for i, class_label in enumerate(prediction_data['classes']):
                    score = prediction_data['scores'][0][i]
                    results.append({
                        'Field': f'Prediction Score (Class {class_label})',
                        'Value': f'{score:.3f}'
                    })
                
                # Add metadata
                results.append({
                    'Field': 'Model',
                    'Value': prediction_data['metadata']['model_name']
                })
                results.append({
                    'Field': 'Prediction Time',
                    'Value': datetime.fromtimestamp(
                        prediction_data['metadata']['prediction_time']
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })
                # jprint(results)
                return pd.DataFrame(results)
                
            except Exception as e:
                log.error(f"Error generating results: {str(e)}")
                log.traceback(e)
                return pd.DataFrame({'Error': [str(e)]})


        @output
        @render_plotly
        @reactive.event(input.selectField1, input.selectField2, input.amount_range)
        def plot_metric():
            try:
                selected_field1 = input.selectField1()
                selected_field2 = input.selectField2()
                plot_df = get_filtered_df()
                index = appDict['target']
                return(plot_default_metric(selected_field1, selected_field2, plot_df, index))
            except Exception as e:
                log.error(f"Error in plot_metric: {str(e)}")
                return go.Figure()
            
        @output
        @render_plotly
        @reactive.event(input.metric)
        def training_metric():
            try:
                X, y = get_trained_model()
                
                # Validate data before creating model
                if X is None or y is None:
                    log.error("X or y is None - check get_trained_model()")
                    return go.Figure()
                    
                if len(X) == 0 or len(y) == 0:
                    log.error("Empty X or y data - check get_trained_model()")
                    return go.Figure()
                    
                log.info(f"Training with X shape: {X.shape}, y shape: {y.shape}")
                
                model_trainer = ModelTrainer(
                    random_state=42, 
                    model_name=input.model(), 
                    test_size=0.2, 
                    X=X, 
                    y=y
                )
                model_data = model_trainer.train_model()
                return plot_model_metrics(model_data, metric_type=input.metric().lower())
                    
            except Exception as e:
                log.error(f"Error creating metric plot: {str(e)}")
                log.traceback(e)  # Add traceback for more detail
                return go.Figure()
                    
        @output
        @render_plotly
        @reactive.event(input.amount_range)
        def score_dist():
            """Display the distribution of model scores for the filtered dataset"""
            try:
                X, y = get_trained_model()
                
                # Validate data
                if X is None or y is None:
                    log.error("X or y is None - check get_trained_model()")
                    return go.Figure()
                    
                model_trainer = ModelTrainer(
                    random_state=42, 
                    model_name=input.model(), 
                    test_size=0.2, 
                    X=X, 
                    y=y
                )
                
                model_data = model_trainer.train_model()
                return plot_score_distribution(model_data)
                    
            except Exception as e:
                log.error(f"Error creating score distribution plot: {str(e)}")
                return go.Figure()
        
        @output
        @render_plotly
        def prod_score_dist():
            """Display the distribution of model scores in production"""
            try:
                X, y = get_trained_model()
                
                # Validate data
                if X is None or y is None:
                    log.error("X or y is None - check get_trained_model()")
                    return go.Figure()
                
                log.info(f"Data shapes - X: {X.shape}, y: {y.shape}")
                
                model_trainer = ModelTrainer(
                    random_state=42, 
                    model_name=input.model(), 
                    test_size=0.2, 
                    X=X, 
                    y=y
                )
                
                # Get training data predictions
                model_data = model_trainer.train_model()
                
                # Get production data (using last 20% of filtered data as mock production data)
                prod_size = int(len(X) * 0.2)
                X_prod = X.tail(prod_size)
                
                # Get production predictions
                prod_predictions = model_trainer.predict(X_prod)
                
                # Log the data structures
                log.info("Training data keys: " + str(model_data.keys()))
                log.info("Production data keys: " + str(prod_predictions.keys()))
                
                combined_data = {
                    'training_data': {
                        'y_scores_test': model_data['y_scores_test'],
                        'y_test': model_data['y_test']
                    },
                    'production_data': {
                        'y_scores': prod_predictions['scores']  # Changed from 'scores' to match predict output
                    },
                    'metadata': {
                        'training_samples': len(X),
                        'production_samples': len(X_prod),
                        'model_name': input.model(),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                
                return plot_production_score_distribution(combined_data)
               
            except Exception as e:
                log.error(f"Error creating production score distribution plot: {str(e)}")
                log.traceback(e)
                return go.Figure()
            
        @output
        @render_plotly
        def api_response():
            """Display API response time trend"""
            try:
                X, y = get_trained_model()
                
                # Validate data
                if X is None or y is None:
                    log.error("X or y is None - check get_trained_model()")
                    return go.Figure()
                    
                model_trainer = ModelTrainer(
                    random_state=42, 
                    model_name=input.model(), 
                    test_size=0.2, 
                    X=X, 
                    y=y
                )
                
                model_data = model_trainer.train_model()
                # jprint(model_data)
                # Get training history from model trainer
                if not model_data['training_history']:
                    log.info("No training history available yet")
                    return go.Figure()  # Return empty plot if no history
               
                return plot_api_response(model_data)
                
            except Exception as e:
                log.error(f"Error creating API response plot: {str(e)}")
                return go.Figure()

        

    try:
        # Try to find a free port multiple times
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                port = find_free_port()
                log.info(f"Starting DataAnt on port {port}")
                app = App(app_ui, server)
                Timer(1.5, lambda: open_browser(port)).start()
                app.run(port=port)
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                log.error(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
    except Exception as e:
        log.error(f"Error starting DataAnt: {str(e)}")
        log.traceback(e)
        raise

                                    
