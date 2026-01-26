# DataAnt - AI-Powered Data Analysis & Machine Learning Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-Managed-orange.svg)](https://python-poetry.org/)

**DataAnt** is an intelligent, interactive data analysis and machine learning platform that combines natural language processing, automated model training, and real-time visualization. Built with Python, Shiny, and Plotly, it provides a comprehensive solution for data scientists and analysts to explore, analyze, and build predictive models through an intuitive web interface.

## 🎯 Overview

DataAnt transforms natural language prompts into actionable data analysis workflows. It features:

- **Natural Language Interface**: Parse analysis requests using spaCy (with Gemini AI fallback)
- **Interactive Web Dashboard**: Real-time data exploration with Shiny for Python
- **Automated ML Pipeline**: Train, evaluate, and monitor machine learning models
- **Advanced Visualizations**: Interactive 3D plots, ROC curves, and performance metrics
- **Secure Configuration**: Encrypted credential management

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐
│   main.py       │  Entry point & CLI interface
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ prompt_parser   │  NLP parsing (spaCy + Gemini fallback)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    engine.py    │  Core business logic & data processing
└────────┬────────┘
         │
         ├──────────► ui_app.py (Shiny UI)
         │
         ├──────────► model.py (ML training)
         │
         └──────────► ui_plot.py (Visualizations)
```

### Key Components

1. **Prompt Parser** (`prompt_parser.py`)
   - Extracts structured data from natural language
   - Uses spaCy for local parsing (no API required)
   - Falls back to Gemini AI for complex queries
   - Supports actions: `analysis`, `run`, `list`, `show`, `filter`, etc.

2. **Analysis Engine** (`engine.py`)
   - Orchestrates data loading and preprocessing
   - Handles field selection, exclusion, and target identification
   - Manages data cleaning and transformation
   - Routes actions to appropriate handlers

3. **Model Trainer** (`model.py`)
   - Singleton pattern for model management
   - Supports multiple algorithms:
     - **Classification**: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM
     - **Regression**: Linear, Ridge, Lasso, Random Forest
     - **Clustering**: KMeans, DBSCAN, Agglomerative
   - Implements model caching and performance tracking
   - Handles stratified train/test splits

4. **Web Interface** (`ui_app.py`)
   - Shiny-based reactive dashboard
   - Real-time data filtering with sliders
   - Model training and evaluation interface
   - Interactive data annotation forms

5. **Visualization** (`ui_plot.py`)
   - 3D scatter plots with selectable axes
   - ROC and Precision-Recall curves
   - Score distributions
   - Performance metrics visualization

## 📦 Installation

### Prerequisites

- Python 3.10 - 3.13
- Poetry (recommended) or pip
- macOS/Linux/Windows

### Step 1: Clone Repository

```bash
git clone https://github.com/arusatech/dataant.git
cd dataant
```

### Step 2: Install Dependencies

**Using Poetry (Recommended):**
```bash
poetry install
poetry shell
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Step 3: Install spaCy Language Model

For local prompt parsing (optional but recommended):
```bash
python -m spacy download en_core_web_sm
```

**Note**: If spaCy model is not installed, the system will fall back to rule-based parsing. For better accuracy, install the model.

### Step 4: Configure Application

Create `config.json` in the root directory:

```json
{
    "db_file": "path/to/your/data.csv",
    "api_key": "your-google-api-key-optional"
}
```

**Setting up configuration via CLI:**
```bash
# Set database file path
python main.py -p "set db_file /path/to/your/data.csv"

# Set Google API key (optional, for Gemini AI fallback)
python main.py -p "set api_key YOUR_GOOGLE_API_KEY"
```

**Security Note**: API keys are automatically encrypted using PBKDF2 with SHA256 and user-specific keys.

## 🚀 Quick Start

### Basic Usage

1. **Start Analysis:**
```bash
python main.py -p "analyze db_file"
```

2. **Access Web Interface:**
   - The application will start a Shiny server (typically on port 50000)
   - Open your browser to the displayed URL (e.g., `http://127.0.0.1:50000`)

3. **Interactive Features:**
   - Use dropdowns to select fields for 3D visualization
   - Adjust sliders to filter data
   - Select model type and metrics
   - View ROC curves, score distributions, and predictions

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -p, --prompt PROMPT    Provide analysis prompt directly
  -f, --file FILE        Provide prompt from a file
  -d, --debug            Enable debug logging
  -t, --template         Generate a prompt template file
  -h, --help             Show help message
```

### Example Prompts

**Basic Analysis:**
```bash
python main.py -p "analyze db_file"
```

**Field-Specific Analysis:**
```bash
python main.py -p "analyze heart disease using age, sex, and cholesterol"
```

**With Target Selection:**
```bash
python main.py -p "analyze data with target as disease_status"
```

**Exclude Fields:**
```bash
python main.py -p "analyze data excluding id and timestamp"
```

## 📊 Features

### 1. Natural Language Processing
- **Local Parsing**: Uses spaCy for prompt understanding (no API required)
- **AI Fallback**: Optional Gemini AI integration for complex queries
- **Action Recognition**: Automatically detects analysis, filtering, and update operations
- **Field Extraction**: Identifies fields, targets, and exclusions from natural language

### 2. Data Analysis & Visualization
- **3D Scatter Plots**: Interactive visualization with selectable axes (Field 1, Field 2, Target)
- **Real-time Filtering**: Dynamic range sliders for numeric fields
- **Data Cleaning**: Automatic handling of missing values and outliers
- **Type Detection**: Smart conversion of mostly-numeric categorical columns

### 3. Machine Learning
- **Multiple Algorithms**: 15+ models for classification, regression, and clustering
- **Automated Training**: One-click model training with optimal hyperparameters
- **Model Caching**: Intelligent caching to avoid redundant training
- **Stratified Splits**: Ensures balanced train/test distributions

### 4. Model Evaluation
- **ROC Curves**: True Positive Rate vs False Positive Rate visualization
- **Precision-Recall Curves**: For imbalanced datasets
- **Score Distributions**: Training vs test performance comparison
- **Performance Metrics**: Real-time tracking of training time and accuracy

### 5. Data Annotation
- **Interactive Forms**: Input features for real-time predictions
- **Multi-class Support**: Handles binary and multi-class classification
- **Probability Scores**: Displays prediction confidence for each class

### 6. Security
- **Encrypted Credentials**: API keys encrypted with user-specific keys
- **Secure Storage**: Credentials stored in config.json with encryption
- **No Plaintext Secrets**: All sensitive data is encrypted at rest

## 📁 Project Structure

```
dataant/
├── dataant/                    # Main package
│   ├── __init__.py
│   ├── engine.py              # Core analysis engine
│   ├── model.py               # ML model implementations
│   ├── ui_app.py              # Shiny web interface
│   ├── ui_plot.py             # Plotly visualizations
│   ├── prompt_parser.py       # NLP prompt parsing
│   ├── util.py                # Utility functions
│   └── db.py                  # Database operations
├── tests/                      # Test files
│   ├── multiclass_roc.py
│   └── plot_shiny.py
├── main.py                     # Application entry point
├── config.json                 # Configuration file (create this)
├── pyproject.toml              # Poetry dependencies
├── requirements.txt            # pip dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

## ⚙️ Configuration

### Config File Structure

`config.json`:
```json
{
    "db_file": "/path/to/data.csv",
    "api_key": "encrypted_google_api_key_optional"
}
```

### Environment Variables

Currently, configuration is managed through `config.json`. Future versions may support environment variables.

### Database Support

DataAnt supports:
- **CSV files** (primary)
- **PostgreSQL** (via SQLAlchemy)
- **MySQL** (via SQLAlchemy)
- **DynamoDB** (via boto3)
- **Other SQL databases** (via SQLAlchemy connection strings)

For database connections, configure connection strings in `config.json` or use the `db_file` field.

## 🔧 Advanced Usage

### Custom Model Training

The system automatically selects the target column, but you can specify it:
```bash
python main.py -p "analyze data with target as disease_status"
```

### Field Selection

Include specific fields:
```bash
python main.py -p "analyze using age, sex, cholesterol, and blood_pressure"
```

Exclude fields:
```bash
python main.py -p "analyze data excluding id, timestamp, and notes"
```

### Model Selection

In the web interface:
- Select model type: Logistic Regression, Random Forest, XGBoost, etc.
- Choose metric: ROC, Precision-Recall
- Adjust filters and see real-time updates

## 🐛 Troubleshooting

### Common Issues

**1. spaCy Model Not Found**
```
WARNING: spaCy model 'en_core_web_sm' not found
```
**Solution**: Install the model:
```bash
python -m spacy download en_core_web_sm
```

**2. XGBoost OpenMP Error (macOS)**
```
XGBoostError: XGBoost Library could not be loaded
```
**Solution**: Install libomp:
```bash
brew install libomp
```

**3. Single Class Error**
```
ValueError: This solver needs samples of at least 2 classes
```
**Solution**: Adjust data filters to include samples from multiple classes. The UI will show a helpful message.

**4. Port Already in Use**
```
Address already in use
```
**Solution**: The application automatically finds the next available port. Check the logs for the actual port number.

### Debug Mode

Enable detailed logging:
```bash
python main.py -d -p "your prompt"
```

## 🧪 Testing

Run test scripts:
```bash
# Test ROC curve plotting
python tests/multiclass_roc.py

# Test Shiny plotting
python tests/plot_shiny.py
```

## 📈 Performance Considerations

- **Model Caching**: Models are cached based on data shape and features
- **Stratified Splits**: Ensures balanced class distribution
- **Efficient Data Processing**: Uses pandas vectorized operations
- **Lazy Loading**: Data loaded only when needed

## 🔐 Security Best Practices

1. **Never commit `config.json`** with unencrypted API keys
2. **Use environment-specific configs** for different deployments
3. **Rotate API keys** regularly
4. **Review access logs** for suspicious activity

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests
4. **Commit with clear messages**: `git commit -m 'Add amazing feature'`
5. **Push to your branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/dataant.git
cd dataant

# Install development dependencies
poetry install

# Run tests
pytest tests/

# Format code
black dataant/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team & Acknowledgments

- **Design & Architecture**: Mr. Yakub Mohammad
- **Organization**: AR USA LLC Team
- **Contact**: arusa@arusatech.com

## 📞 Support

- **Email**: arusa@arusatech.com
- **Issues**: [GitHub Issues](https://github.com/arusatech/dataant/issues)
- **Documentation**: See inline code documentation

## 🗺️ Roadmap

- [ ] Support for time series analysis
- [ ] Enhanced database connectors
- [ ] Model export/import functionality
- [ ] Batch prediction API
- [ ] Multi-user support with authentication
- [ ] Docker containerization
- [ ] Kubernetes deployment guides

## 📚 Additional Resources

- [Shiny for Python Documentation](https://shiny.posit.co/py/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [spaCy Documentation](https://spacy.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**Made with ❤️ by the AR USA LLC Team**
