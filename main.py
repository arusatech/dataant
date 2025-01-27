import os
import re
import sys
import json
import getpass
import platform
import argparse
import signal
from jsonpath_nz import log, jprint
from dataant.util import gemini_client_process, TextCipher, extract_json
from dataant.engine import runEngine
from jsonpath_nz import log
import pandas as pd
#Global variables
#Signal handler
GLB_PROMPT = f'''Your task as developer
You will be provided with text delimited by triple backticks
Provide them in JSON format with the following keys:
action, db_file, fields, exclude, target , primary
where
action key = any of the following actions [run, analysis, update, list, show, filter, update_filter, update_field] or default set to None
db_file key = path to the database file or default set to None
fields key = list of fields given from the prompt as list to include in the output or default set to None
exclude key = list of fields given from the prompt as list to exclude from the output or default set to None 
target key = target key is the field name to be predicted or default set to None
primary key = primary key is teh field name given by the user to be used as the primary key of the db or default set to None
'''
def signal_handler(sig, frame):
    '''Signal handler for Ctrl+C'''
    log.info('\n !!!You pressed Ctrl+C , Exiting ...... ')
    sys.exit(0)

def parse_opts(argv):
    """
    Parsing command line argument for tool 'dataant' 
    """
    parser = argparse.ArgumentParser(prog="dataant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--prompt", type=str, nargs='?', default=None , help="Prompt: Generative Prompt for the Date Analytic bot")
    group.add_argument("-f", "--file", type=str, nargs='?', default=None , help="Prompt: provided as a file (use -t to get the prompt template)")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug: Captures debug to the default temp file")
    parser.add_argument("-t", "--template", action="store_true", help="template: Generative Prompt template file")
    
    if len(argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()
    
def main(argv):
    try:
        opts = parse_opts(argv)
        config = None
        log.info(f"Prompt Engineering request -- started")
        prompt = opts.prompt
        if not prompt:
            if opts.file:
                prompt = open(opts.file, 'r').read()
            else:
                return(f"Please provide prompt: to start the bot",False)
        
        if os.path.exists('config.json'):
            with open('config.json', 'r') as file:
                config = json.load(file)
        else:
            log.error(f"!! Error : config.json file NOT FOUND")
            log.info(f"Creating a new config.json file at {os.path.join(os.getcwd(), 'config.json')}")  
            #create a new config.json file
            with open('config.json', 'w') as file:
                json.dump({'api_key': 'None', 'db_file': 'None'}, file)
            return(f'''
                    --- To Start the bot ---
                    provide prompt: 'set api_key <YOUR GOOGLE_API_KEY> '
                    Provide prompt: 'set db_file <YOUR DB_FILE_PATH> '
                    ''',False)
        
        #Prompt starts with 'set' and follows by key and value
        public_key = f"{getpass.getuser()}@{platform.system()}"
        if str(prompt).lower().startswith('set '):
            set_patterns = {
                'colon': r'^set\s+([^\s:]+)\s*:\s*(.+)$',      # set key: value
                'equals': r'^set\s+([^\s=]+)\s*=\s*(.+)$',     # set key=value
                'space': r'^set\s+([^\s:=]+)\s+([^:=].+)$'     # set key value
            }
            key = None
            value = None
            sensitive_keys = ['key', 'secret', 'password', 'pass'] 
            log.info(f"Prompt: {prompt}")
            # Try each pattern
            for pattern_name, pattern in set_patterns.items():
                match = re.match(pattern, prompt, re.IGNORECASE)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    log.info(f"Key: {key} - Value: {value}")
                    #encrypt the sensitive data
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        value = TextCipher().encrypt(value, public_key)
                    break
            if key is None or value is None:
                return(f"Invalid set prompt format. Please use 'set key value' format.",False)
            config[key] = value
            log.info(f"Config: {config}")
            with open('config.json', 'w') as file:
                json.dump(config, file)
            return(f"Config updated successfully with {key} : {value}",True)

        # Check if required config values exist
        if not config.get('api_key'):
            return "Provide prompt: 'set api_key <YOUR GOOGLE_API_KEY>'", False

        if not config.get('db_file'):
            return "Provide prompt: 'set db_file <YOUR DB_FILE_PATH>'", False


        #Generative AI prompts
        api_key = TextCipher().decrypt(config['api_key'], public_key)
        if not api_key:
            return(f"Provide prompt: 'set api_key <YOUR GOOGLE_API_KEY>' ",False)
        prompt = f'''{GLB_PROMPT}
        ```{prompt}```
        '''
        prompt_response = gemini_client_process(prompt, api_key)
        # fmt_print(prompt_response)
        if prompt_response['metadata']['status'] == 'error':
            return(f'''Failed to get response from Gemini [ {prompt_response['response']['error']} ], 
                    Try again by providing valid api_key (set api_key <YOUR GOOGLE_API_KEY>)''',False)
        else:
            json_data = extract_json(prompt_response['response']['content'])
            count= 0
            while not json_data:
                prompt_response = gemini_client_process(prompt, api_key)
                json_data = extract_json(prompt_response['response']['content'])
                count += 1
                if count > 3:
                    return(f"Failed to extract JSON from the prompt response after 3 attempts",False)
                else:
                    log.info(f"Retrying {count} time again... ")

            jprint(json_data)
            msg = runEngine(config, json_data)
            return(msg, True)
        
    except Exception as e:
        log.traceback(e)
        return(f"Error: {e}",False)
    

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    opts = parse_opts(sys.argv)
    log.info(f"Opts: {opts}")
    if opts.template:
        template_sentence = f'''{GLB_PROMPT} ```<YOUR PROMPT>```'''
        template_file = 'prompt_template.txt'   
        with open(template_file, 'w') as file:
            file.write(template_sentence)
        log.info(f"Template file created at {template_file}")
        sys.exit(0)
    try:
        msg,status = main(sys.argv)
        if not status:
            log.error(msg)
            sys.exit(1)
        else:
            jprint(msg)
            sys.exit(0)
            # jprint(msg)
            # if isinstance(msg, dict):
            #     if 'slider_name' in msg.keys():
            #     # log.info(f"Slider name: {msg['slider_name']}")
            #     appDict = msg
            #     #remove duplicate columns
            #     df = df.loc[:, ~df.columns.duplicated()]
            #     start_DataAnt(appDict, df)
                # Create app
    except Exception as e:
        log.critical(f"Error on line number: {sys.exc_info()[-1].tb_lineno}")
        log.traceback(e)
        sys.exit(1)
    