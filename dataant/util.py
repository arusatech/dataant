import os
import re
import json
import base64
import google.generativeai as genai
from datetime import datetime
from jsonpath_nz import log , jprint
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional, Dict, Any

#Global setting
#create OUTPUT_DIR if not exists from current working directory
OUTPUT_DIR = os.path.join(os.getcwd(), f'dataant_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, will look for GOOGLE_API_KEY env variable
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY key must be provided either directly from config.json or through environment variable")
        
        # Configure the library
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-pro')
           
    def _create_response_template(self, prompt: str, response: str, status: str = "success", 
                                error: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized JSON response template
        
        Args:
            prompt (str): Original prompt
            response (str): Generated response
            status (str): Status of the generation (success/error)
            error (str, optional): Error message if any
            
        Returns:
            dict: Formatted response dictionary
        """
        return {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "model": "gemini-pro",
                "status": status
            },
            "request": {
                "prompt": prompt,
                "type": "generate_content"
            },
            "response": {
                "content": response if status == "success" else None,
                "error": error if status == "error" else None
            }
        }

    def generate_response(self, prompt: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using Gemini in JSON format
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters for the generation
                     (temperature, top_p, top_k, max_output_tokens, etc.)
        
        Returns:
            dict: JSON formatted response
        """
        try:
            # Generate the response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**kwargs)
            )
            
            # Return formatted response
            return self._create_response_template(prompt, response.text)
            
        except Exception as e:
            return self._create_response_template(
                prompt=prompt,
                response="",
                status="error",
                error=str(e)
            )
    
    def generate_chat_response(self, messages: list, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response in chat mode with JSON format
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the generation
        
        Returns:
            dict: JSON formatted response
        """
        try:
            # Start a chat
            chat = self.model.start_chat()
            
            # Send all messages
            last_prompt = ""
            for message in messages:
                if message['role'] == 'user':
                    last_prompt = message['content']
                    response = chat.send_message(message['content'], **kwargs)
            
            # Return formatted response
            return self._create_response_template(last_prompt, response.text)
            
        except Exception as e:
            return self._create_response_template(
                prompt=messages[-1]['content'] if messages else "",
                response="",
                status="error",
                error=str(e)
            )

class TextCipher:
    '''
    TextCipher class to encrypt and decrypt text
    ''' 
    def __init__(self, salt: bytes = None):
        """
        Initialize TextCipher with optional salt
        
        Args:
            salt (bytes, optional): Salt for key derivation
        """
        self.salt = salt if salt else os.urandom(16)
        
    def _generate_key(self, password: str) -> bytes:
        """
        Generate encryption key from password
        
        Args:
            password (str): Password to derive key from
            
        Returns:
            bytes: Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, text: str, password: str) -> str:
        """
        Encrypt plain text using password
        
        Args:
            text (str): Text to encrypt
            password (str): Password for encryption
            
        Returns:
            CipherResult: Encryption result
        """
        try:
            # Input validation
            if not text or not password:
                return(None)
            
            # Generate key from password
            key = self._generate_key(password)
            cipher_suite = Fernet(key)
            
            # Encrypt
            cipher_text = cipher_suite.encrypt(text.encode())
            
            # Combine salt and cipher text
            combined = base64.urlsafe_b64encode(self.salt + cipher_text)
            
            return(combined.decode())
            
        except Exception as e:
            log.error(f"Encryption failed: {traceback(e)}")
            return None

    def decrypt(self, cipher_text: str, password: str) -> str:
        """
        Decrypt cipher text using password
        
        Args:
            cipher_text (str): Text to decrypt
            password (str): Password for decryption
            
        Returns:
            CipherResult: Decryption result
        """
        try:
            # Input validation
            if not cipher_text or not password:
                return(None)
            
            
            # Decode combined salt and cipher text
            combined = base64.urlsafe_b64decode(cipher_text.encode())
            
            # Extract salt and cipher text
            salt = combined[:16]
            actual_cipher_text = combined[16:]
            
            # Set salt and generate key
            self.salt = salt
            key = self._generate_key(password)
            cipher_suite = Fernet(key)
            
            # Decrypt
            plain_text = cipher_suite.decrypt(actual_cipher_text)
            
            return(plain_text.decode())
            
        except Exception as e:
            log.traceback(e)
            return(None)

def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from content string that may be wrapped in markdown code blocks
    
    Args:
        content (str): Content string containing JSON
        
    Returns:
        Optional[Dict[str, Any]]: Parsed dictionary or None if invalid
    """
    try:
        # Pattern to match JSON content between code blocks
        json_pattern = r'```(?:json)?\n(.*?)\n```'
        
        # Try to find JSON content in code blocks
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            # Extract JSON string from code block
            json_str = match.group(1)
        else:
            # Try parsing the content directly
            json_str = content
            
        # Parse JSON string to dict
        return json.loads(json_str)
        
    except Exception as e:
        log.traceback(e)
        return None 
    
def gemini_client_process(prompt, api_key):
    '''Process the prompt using GeminiClient'''
    promptClient = GeminiClient(api_key=api_key)
    response = promptClient.generate_response(
        prompt,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1000,
        candidate_count=1
    )
    return response