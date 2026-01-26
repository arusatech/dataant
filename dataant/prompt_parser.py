"""
spaCy-based prompt parser to extract structured data from natural language prompts.
Replaces Gemini AI dependency for prompt parsing.
"""
import re
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from jsonpath_nz import log

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    log.warning("spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")


class PromptParser:
    """
    Parse natural language prompts to extract structured information:
    - action: [run, analysis, update, list, show, filter, update_filter, update_field]
    - db_file: path to database file
    - fields: list of fields to include
    - exclude: list of fields to exclude
    - target: target field name for prediction
    - primary: primary key field name
    """
    
    # Valid actions
    VALID_ACTIONS = ['run', 'analysis', 'update', 'list', 'show', 'filter', 'update_filter', 'update_field']
    
    def __init__(self):
        """Initialize the parser with spaCy model"""
        self.nlp = None
        self.matcher = None
        if SPACY_AVAILABLE:
            try:
                # Try to load the English model
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self._setup_patterns()
            except OSError:
                log.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                log.info("Falling back to rule-based parsing without spaCy")
    
    def _setup_patterns(self):
        """Setup spaCy matcher patterns for action detection"""
        if not self.matcher:
            return
        
        # Pattern for "analyze" or "analysis"
        self.matcher.add("ANALYSIS", [[{"LEMMA": {"IN": ["analyze", "analysis"]}}]])
        # Pattern for "list" or "show"
        self.matcher.add("LIST", [[{"LEMMA": {"IN": ["list", "show", "display"]}}]])
        # Pattern for "run"
        self.matcher.add("RUN", [[{"LEMMA": "run"}]])
        # Pattern for "update"
        self.matcher.add("UPDATE", [[{"LEMMA": "update"}]])
        # Pattern for "filter"
        self.matcher.add("FILTER", [[{"LEMMA": "filter"}]])
    
    def parse(self, prompt: str) -> Dict[str, Any]:
        """
        Parse a natural language prompt and extract structured data
        
        Args:
            prompt: Natural language prompt string
            
        Returns:
            Dictionary with keys: action, db_file, fields, exclude, target, primary
        """
        prompt_lower = prompt.lower().strip()
        prompt_original = prompt.strip()  # Keep original for file path extraction
        
        result = {
            'action': None,
            'db_file': None,
            'fields': None,
            'exclude': None,
            'target': None,
            'primary': None
        }
        
        # Extract action
        result['action'] = self._extract_action(prompt_lower)
        
        # Extract db_file path (use original prompt to preserve case for paths)
        result['db_file'] = self._extract_db_file(prompt_original)
        
        # Extract fields to include
        result['fields'] = self._extract_fields(prompt_lower, prompt_original)
        
        # Extract fields to exclude
        result['exclude'] = self._extract_exclude(prompt_lower)
        
        # Extract target field
        result['target'] = self._extract_target(prompt_lower)
        
        # Extract primary key
        result['primary'] = self._extract_primary(prompt_lower)
        
        return result
    
    def _extract_action(self, prompt_lower: str) -> Optional[str]:
        """Extract action from prompt"""
        # Use spaCy if available
        if self.nlp and self.matcher:
            doc = self.nlp(prompt_lower)
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                if label == "ANALYSIS":
                    return "analysis"
                elif label == "LIST":
                    return "list"
                elif label == "RUN":
                    return "run"
                elif label == "UPDATE":
                    return "update"
                elif label == "FILTER":
                    return "filter"
        
        # Fallback to regex patterns
        action_patterns = {
            'analysis': r'\b(analyze|analysis|analyzing)\b',
            'list': r'\b(list|show|display|view)\b',
            'run': r'\b(run|execute|start)\b',
            'update': r'\b(update|modify|change)\b',
            'filter': r'\b(filter|filtering)\b',
            'show': r'\b(show|display)\b'
        }
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, prompt_lower):
                # Map 'show' to 'list' for consistency
                return 'list' if action == 'show' else action
        
        # Default to 'analysis' if no action found but prompt seems like analysis
        if any(word in prompt_lower for word in ['analyze', 'dataset', 'data', 'predict', 'model']):
            return 'analysis'
        
        return None
    
    def _extract_db_file(self, prompt: str) -> Optional[str]:
        """Extract database file path from prompt"""
        # Pattern for file paths (absolute or relative)
        # Matches paths like: /path/to/file.csv, ./file.csv, file.csv, ~/file.csv
        path_patterns = [
            r'(?:db_file|database|file|path|dataset)\s*[:=]\s*([^\s,]+\.(?:csv|json|parquet|xlsx|db|sqlite))',
            r'([/\.~]?[^\s,]+\.(?:csv|json|parquet|xlsx|db|sqlite))',
            r'["\']([^"\']+\.(?:csv|json|parquet|xlsx|db|sqlite))["\']'
        ]
        
        for pattern in path_patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                file_path = match.group(1) if match.lastindex else match.group(0)
                # Expand ~ to home directory
                if file_path.startswith('~'):
                    file_path = os.path.expanduser(file_path)
                # Check if file exists or is a valid path
                if os.path.exists(file_path) or os.path.isabs(file_path) or file_path.startswith('./'):
                    return file_path
        
        return None
    
    def _extract_fields(self, prompt_lower: str, prompt_original: str = None) -> Optional[List[str]]:
        """Extract fields/columns to include"""
        fields = []
        
        # Pattern for "fields", "columns", "include"
        patterns = [
            r'(?:fields?|columns?|include)\s*[:=]\s*\[([^\]]+)\]',  # fields: [field1, field2]
            r'(?:fields?|columns?|include)\s+(?:are|is|:)\s*([^,]+(?:,\s*(?:and\s+)?[^,]+)*)',  # fields are field1, field2
            r'using\s+([^,]+(?:,\s*(?:and\s+)?[^,]+)*)\s+(?:as\s+)?(?:features?|fields?|columns?)',  # using field1, field2 as features
            r'(?:features?|fields?|columns?)\s+([^,]+(?:,\s*(?:and\s+)?[^,]+)*)',  # features field1, field2
            r'focusing\s+on\s+([^,]+(?:,\s*(?:and\s+)?[^,]+)*)',  # focusing on field1, field2
            r'with\s+([^,]+(?:,\s*(?:and\s+)?[^,]+)*)\s+(?:as\s+)?(?:features?|fields?|columns?)',  # with field1, field2 as features
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                field_str = match.group(1)
                # Split by comma and "and", then clean
                field_list = re.split(r',\s*|\s+and\s+', field_str)
                field_list = [f.strip().strip("'\"") for f in field_list if f.strip()]
                fields.extend(field_list)
        
        # Also look for field names mentioned directly (common field patterns)
        # This is a simple heuristic - can be improved
        if not fields:
            # Look for quoted field names (use original prompt to preserve case)
            search_prompt = prompt_original if prompt_original else prompt_lower
            quoted_fields = re.findall(r'["\']([^"\']+)["\']', search_prompt)
            if quoted_fields:
                fields.extend(quoted_fields)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fields = []
        for field in fields:
            if field.lower() not in seen:
                seen.add(field.lower())
                unique_fields.append(field)
        
        return unique_fields if unique_fields else None
    
    def _extract_exclude(self, prompt_lower: str) -> Optional[List[str]]:
        """Extract fields/columns to exclude"""
        exclude = []
        
        patterns = [
            r'exclude\s*[:=]\s*\[([^\]]+)\]',  # exclude: [field1, field2]
            r'exclude\s+(?:are|is|:)\s*([^,]+(?:,\s*[^,]+)*)',  # exclude are field1, field2
            r'(?:excluding|without)\s+([^,]+(?:,\s*(?:and\s+)?[^,]+)*)',  # excluding field1, field2
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                exclude_str = match.group(1)
                exclude_list = [e.strip().strip("'\"") for e in exclude_str.split(',')]
                exclude.extend(exclude_list)
        
        return exclude if exclude else None
    
    def _extract_target(self, prompt_lower: str) -> Optional[str]:
        """Extract target field for prediction"""
        patterns = [
            r'target\s*(?:key|field|column)?\s*[:=]\s*["\']?([^"\',\s]+)["\']?',  # target: field_name
            r'target\s+(?:is|as|:)\s*["\']?([^"\',\s]+)["\']?',  # target is field_name
            r'predict\s+["\']?([^"\',\s]+)["\']?',  # predict field_name
            r'predicting\s+["\']?([^"\',\s]+)["\']?',  # predicting field_name
            r'(?:predict|target)\s+(?:for|on)\s+["\']?([^"\',\s]+)["\']?',  # predict for field_name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_primary(self, prompt_lower: str) -> Optional[str]:
        """Extract primary key field"""
        patterns = [
            r'primary\s*(?:key|field|column)?\s*[:=]\s*["\']?([^"\',\s]+)["\']?',  # primary: field_name
            r'primary\s+(?:key|is|as|:)\s*["\']?([^"\',\s]+)["\']?',  # primary key is field_name
            r'primary\s+key\s+["\']?([^"\',\s]+)["\']?',  # primary key field_name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        return None


def parse_prompt(prompt: str) -> Dict[str, Any]:
    """
    Convenience function to parse a prompt
    
    Args:
        prompt: Natural language prompt string
        
    Returns:
        Dictionary with parsed information
    """
    parser = PromptParser()
    return parser.parse(prompt)
