import os
import re
import sys
from pathlib import Path
from jsonpath_nz import jprint, log
from typing import Dict, Any, Optional
import pandas as pd
from dataant.util import gemini_client_process, extract_json, TextCipher
from dataant.ui_app import start_DataAnt


def convert_mostly_numeric(series: pd.Series, threshold: float = 0.75) -> pd.Series:
    """Convert series to numeric if majority of values are numeric"""
    def remove_outliers_zscore(series: pd.Series, threshold_outliers: float = 3) -> pd.Series:
        """Remove outliers using Z-score method"""
        z_scores = (series - series.mean()) / series.std()
        return series[abs(z_scores) < threshold_outliers]
    try:
        if series.dtype == 'object':
            # Pattern for numeric values (including negatives and decimals)
            pattern = r'^-?\d*\.?\d+$'
            
            # Get unique values and their counts
            unique_values = series.unique()
            numeric_values = [v for v in unique_values if pd.notna(v) and 
                            bool(re.match(pattern, str(v).strip().replace(',','')))]
            
            # Calculate percentage of numeric unique values
            numeric_percentage = len(numeric_values) / len(unique_values)
            
            if numeric_percentage >= threshold:
                # Convert numeric strings to float
                def is_numeric(x):
                    if pd.isna(x):
                        return False
                    return bool(re.match(pattern, str(x).strip().replace(',','')))
                
                numeric_mask = series.apply(is_numeric)
                numeric_series = pd.to_numeric(series[numeric_mask].str.replace(',',''), 
                                            errors='coerce')
                
                # Calculate mean of numeric values truncate to 0 decimal places
                mean_value = numeric_series.mean().round(0)
                
                # Replace non-numeric values with mean
                series = pd.to_numeric(series.astype(str).str.strip().str.replace(',',''), 
                                     errors='coerce').fillna(mean_value)
                
                series = remove_outliers_zscore(series)
                
                log.info(f"Converted mostly numeric column: {numeric_percentage:.2%} numeric values")
                log.info(f"Replaced non-numeric values with mean: {mean_value:.0f}")
                
                # Determine if we need float or int
                if series.dropna().apply(float.is_integer).all():
                    return series.astype('Int64')
                return series.astype('float64')
                
        return series
        
    except Exception as e:
        log.error(f"Error converting mostly numeric column: {str(e)}")
        return series
    
        
def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing empty columns and applying get_dummies"""
    try:
        # Copy DataFrame to avoid modifying original
        df_clean = df.copy()
        
        # Calculate percentage of missing values in each column
        missing_percentages = (df_clean.isna().sum() / len(df_clean)) * 100
        
        # Remove columns with more than 50% missing values
        columns_to_drop = missing_percentages[missing_percentages > 50].index
        df_clean = df_clean.drop(columns=columns_to_drop)
        # log.info(f"Dropped columns with >50% missing values: {columns_to_drop.tolist()} -- {len(columns_to_drop.tolist())}")
        
        # Fill remaining missing values
        # For numeric columns: fill with mean
        numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean().round(2))
            # log.info(f"Filled missing values for numeric columns: {col} --  value: {df_clean[col].mean().round(2)}")
        
        # For categorical columns: fill with mode
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = convert_mostly_numeric(df_clean[col])
        
        log.info(f"Original shape: {df.shape}")
        log.info(f"Cleaned shape: {df_clean.shape}")
        log.info(f"Remaining missing values: {df_clean.isna().sum().sum()}")
        
        return df_clean
        
    except Exception as e:
        log.error(f"Error cleaning and transforming data: {str(e)}")
        return df
    
class ActionHandler:
    def __init__(self, config_dict: Dict[str, Any], prompt_dict: Dict[str, Any], df: pd.DataFrame):
        """
        Initialize action handler
        
        Args:
            config_dict: Configuration dictionary with URLs and credentials
            prompt_dict: Test data dictionary with action parameters
            logger: Logger object for logging
        """
        self.config = config_dict
        self.prompt_dict = prompt_dict
        self.df = df
        
    def process_action(self) -> Dict[str, Any]:
        """
        Process action based on test data
        
        Returns:
            Dict containing result/status
        """
        try:
            action = self.prompt_dict.get('action', '').lower()
            # Validate action
            if not action:
                raise ValueError("Action is required")
                
            # Process based on action type
            if action in ['run', 'analysis', 'analyze']:
                return self._handle_analysis()
            elif action in ['update', 'add_field', 'add_filter', 'update_filter', 'update_field']:
                return self._handle_update()
            elif action in ['list', 'show', 'filter']:
                return self._handle_list()
            elif action in ['raise']:
                return self._handle_raise()
            else:
                raise ValueError(f"Invalid action: {action}")
                
        except Exception as e:
            log.traceback(e)
            raise

    def _handle_analysis(self) -> Dict[str, Any]:
        """Handle Analysis action"""
        def clean_numeric_string(series):
            # Remove currency symbols, commas, spaces, and newlines
            cleaned = (series
                .astype(str)
                .str.replace(r'[$€£¥,\s\n\r]', '', regex=True)  # Remove currency symbols, commas, whitespace
                .str.replace(r'^-$', '0', regex=True)           # Replace lone dashes with 0
                .str.strip()                                    # Remove leading/trailing whitespace
            )
            return pd.to_numeric(cleaned, errors='coerce')
        
        df = self.df.copy()
        log.info(f"Analysis action started .. to analyise {len(df)} entites")
        retDict = {
            'info': '',
            'fields': self.prompt_dict.get('fields', []),
            'target': self.prompt_dict.get('target', None),
            'primary': self.prompt_dict.get('primary', None)
        }
        
        # Get unique fields to process
        prompt_fields = set(self.prompt_dict.get('fields', []))
        valid_fields = []
        unique_values = []
        uniqueFields = []#
        
        target_key = self.prompt_dict.get('target', None)
        if not target_key:
            log.info(f"target key not found... taking last field as target/target key")
            target_key = prompt_fields[-1]
        # Process each field
        for field in prompt_fields:
            # log.info(f"Processing field: {field}")
            # Skip if field doesn't exist in DataFrame
            if field not in df.columns:
                log.warning(f"Field '{field}' not found in DataFrame")
                continue
            try:
                # Convert to series if needed
                series = df[field]
                if isinstance(series, pd.DataFrame):
                    log.warning(f"Field '{field}' is a DataFrame, flattening...")
                    series = series.iloc[:, 0]  # Take first column
                
                # Handle amount fields - set up slider values
                if re.search(target_key, field, re.IGNORECASE):
                    log.info(f"Found amount field: {field} -- len(series): {len(series)}")
                    cleaned_series = clean_numeric_string(series)
                    log.info(f"Cleaned series: {len(cleaned_series)}")
                    if cleaned_series.notna().any():
                        retDict['slider_name'] = field
                        retDict['slider_min'] = int(cleaned_series.min())
                        retDict['slider_max'] = int(cleaned_series.max())
                # Process unique values for the specific column
                field_values = series.unique()
                if len(field_values) == 1 and pd.notna(field_values[0]):
                    log.info(f"Unique value for field {field} is {field_values[0]}")
                    uniqueFields.append(field)
                    unique_values.append(str(field_values[0]).strip())
                
                valid_fields.append(field)
                    
            except Exception as e:
                log.error(f"Error processing field '{field}': {str(e)}")
                continue
        
        # Update return dictionary
        if unique_values:
            retDict['info'] = ' | '.join(unique_values)
        retDict['fields'] = valid_fields
        retDict['uniqueFields'] = uniqueFields
        
        start_DataAnt(retDict, df)
        return retDict
    
    def _handle_update(self) -> Dict[str, Any]:
        """Handle update action"""
        try:
            pass
        except Exception as e:
            log.traceback(e)
            raise

    def _handle_list(self) -> Dict[str, Any]:
        """Handle list action"""
        try:
            log.info(f"List action")
            # jprint(self.prompt_dict)
            unique_list = []
            filter_list = self.prompt_dict.get('fields', [])
            if not filter_list:
                filter_list = self.df.columns.tolist()
            for f in filter_list:
                unique_list.append((str(f), int(self.df[f].nunique())))
            unique_list.sort(key=lambda x: x[1], reverse=True)
            # jprint(unique_list)
        
            self.prompt_dict['fields'] = unique_list
            log.info(f"Total field: {len(filter_list)}")
            return(self.prompt_dict)
        except Exception as e:
            log.traceback(e)
            raise

    def _handle_raise(self) -> Dict[str, Any]:
        """Handle create action"""
        try:
            #Raise JIRA ticket for failed test cases
            pass
                
        except Exception as e:
            log.traceback(e)
            raise


def runEngine(config_data, prompt_data):
    # jprint(prompt_data)
    log.info(f"Engine started")
    try:
        #Initialize db_file
        db_file = None
        target_key = None
        primary_key = None
        if prompt_data.get('db_file', None) is not None:
            db_file = prompt_data['db_file']
            if not Path(db_file).exists():
                log.error(f"DB file not found: {db_file}, hence using default db file {config_data['db_file']}")
                db_file = config_data['db_file']
                prompt_data['db_file'] = db_file
        else:
            db_file = config_data['db_file']
            prompt_data['db_file'] = db_file
            
        exclude_list = []
        field_list = []
        if prompt_data.get('fields', None) is not None:
            field_list += prompt_data['fields']
        if prompt_data.get('exclude', None) is not None:
            exclude_list += prompt_data['exclude']
        if prompt_data.get('target', None) is not None:
            target_key = prompt_data['target']
        if prompt_data.get('primary', None) is not None:
            primary_key = prompt_data['primary']
            
        filter_list = []
        
        #Load data from db_file
        df = pd.read_csv(db_file, low_memory=False)
        log.info(f"Loaded data from {db_file} -- len(df): {len(df)}")
        #Clean data
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        log.info(f"Cleaned data -- len(df): {len(df)}")
        df = clean_and_transform_data(df)
        
        colList = df.columns.tolist()        
        log.info(f"Length of colList (Fields): {len(colList)}")
        
        if len(field_list) > 0:
            for field in field_list:
                for col in colList:
                    if re.search(field, col, re.IGNORECASE):
                        filter_list.append(col)
        else:
            filter_list = colList

        if len(exclude_list) > 0:
            for exclude in exclude_list:
                filter_list = [
                    filter_item for filter_item in filter_list 
                    if not re.search(exclude, filter_item, re.IGNORECASE)
                ]
        
        df = df[filter_list]
        prompt_data['fields'] = filter_list
        
        log.info(f"Processing action [ {prompt_data['action']} ] -- with dataframe of {len(df)} enties")
        action_handler = ActionHandler(config_data, prompt_data, df)
        
        result = action_handler.process_action()
        return(result)
        
    except Exception as e:
        log.error(f"Engine failed: {e}")
        log.traceback(e)
        log.error(f"Error line no: {sys.exc_info()[2].tb_lineno}")
        return(False, f"Failed to Start Engine... Please re-try the command")
